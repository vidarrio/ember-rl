use burn::tensor::backend::AutodiffBackend;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rl_traits::{Environment, Experience, Policy, ReplayBuffer};

use crate::algorithms::dqn::{CircularBuffer, DqnAgent};
use crate::encoding::{DiscreteActionMapper, ObservationEncoder};
use crate::stats::{EpisodeRecord, EvalReport, StatsTracker};
use crate::training::run::TrainingRun;

/// Metrics emitted after every environment step.
///
/// The trainer yields one of these per call to `Iterator::next()`.
/// Use these to log progress, plot curves, or decide when to stop.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Total environment steps taken so far.
    pub total_steps: usize,

    /// Which episode this step belongs to.
    pub episode: usize,

    /// Step index within the current episode.
    pub episode_step: usize,

    /// Reward received this step.
    pub reward: f64,

    /// Cumulative reward in the current episode so far.
    pub episode_reward: f64,

    /// Current ε (exploration rate). `None` if in warm-up.
    pub epsilon: f64,

    /// Whether a gradient update was performed this step.
    pub did_update: bool,

    /// Whether this step ended the episode.
    pub episode_done: bool,

    /// How the episode ended (only meaningful when `episode_done` is `true`).
    pub episode_status: rl_traits::EpisodeStatus,
}

/// The imperative training driver.
///
/// Drives the interaction between an environment and a DQN agent,
/// exposing it as an iterator that yields `StepMetrics` after every step,
/// as well as higher-level `train()` and `eval()` methods.
///
/// # Usage — iterator style (manual control)
///
/// ```rust,ignore
/// let mut trainer = DqnTrainer::new(env, agent, seed);
///
/// for step in trainer.steps().take(50_000) {
///     if step.episode_done {
///         println!("Episode {} reward: {}", step.episode, step.episode_reward);
///     }
/// }
/// ```
///
/// # Usage — imperative style (with `TrainingRun`)
///
/// ```rust,ignore
/// let run = TrainingRun::create("cartpole", "v1")?;
/// let mut trainer = DqnTrainer::new(env, agent, seed).with_run(run);
/// trainer.train(200_000);
/// let report = trainer.eval(20);
/// report.print();
/// ```
pub struct DqnTrainer<E, Enc, Act, B, Buf = CircularBuffer<
    <E as Environment>::Observation,
    <E as Environment>::Action,
>>
where
    E: Environment,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    env: E,
    agent: DqnAgent<E, Enc, Act, B, Buf>,
    rng: SmallRng,

    // Episode state
    current_obs: Option<E::Observation>,
    episode: usize,
    episode_step: usize,
    episode_reward: f64,

    // Optional run tracking
    run: Option<TrainingRun>,
    stats: StatsTracker,

    // Checkpoint policy
    checkpoint_freq: usize,
    keep_checkpoints: usize,

    // For "save best" during eval
    best_eval_reward: f64,
}

impl<E, Enc, Act, B, Buf> DqnTrainer<E, Enc, Act, B, Buf>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B>
        + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    pub fn new(env: E, agent: DqnAgent<E, Enc, Act, B, Buf>, seed: u64) -> Self {
        Self {
            env,
            agent,
            rng: SmallRng::seed_from_u64(seed),
            current_obs: None,
            episode: 0,
            episode_step: 0,
            episode_reward: 0.0,
            run: None,
            stats: StatsTracker::new(),
            checkpoint_freq: 10_000,
            keep_checkpoints: 5,
            best_eval_reward: f64::NEG_INFINITY,
        }
    }

    /// Attach a `TrainingRun` for checkpoint saving and stats persistence.
    pub fn with_run(mut self, run: TrainingRun) -> Self {
        self.run = Some(run);
        self
    }

    /// How often (in steps) to save a numbered checkpoint. Default: 10_000.
    pub fn with_checkpoint_freq(mut self, freq: usize) -> Self {
        self.checkpoint_freq = freq;
        self
    }

    /// How many numbered checkpoints to keep on disk. Default: 5.
    pub fn with_keep_checkpoints(mut self, keep: usize) -> Self {
        self.keep_checkpoints = keep;
        self
    }

    /// Replace the default stats tracker with a custom one.
    pub fn with_stats(mut self, stats: StatsTracker) -> Self {
        self.stats = stats;
        self
    }

    /// Returns an iterator that yields `StepMetrics` after each environment step.
    ///
    /// The iterator is infinite — stop it with `.take(n)` or `break`.
    pub fn steps(&mut self) -> TrainIter<'_, E, Enc, Act, B, Buf> {
        TrainIter { trainer: self }
    }

    /// Access the agent for evaluation or inspection.
    pub fn agent(&self) -> &DqnAgent<E, Enc, Act, B, Buf> {
        &self.agent
    }

    /// Consume the trainer and return the inner agent.
    ///
    /// Useful for converting to a `DqnPolicy` after training:
    /// ```rust,ignore
    /// let policy = trainer.into_agent().into_policy();
    /// ```
    pub fn into_agent(self) -> DqnAgent<E, Enc, Act, B, Buf> {
        self.agent
    }

    /// Access the environment directly.
    pub fn env(&self) -> &E {
        &self.env
    }

    /// Run `n_steps` of training.
    ///
    /// If a `TrainingRun` is attached, saves checkpoints at `checkpoint_freq`
    /// intervals and writes episode records to `train_episodes.jsonl`.
    pub fn train(&mut self, n_steps: usize) {
        let start_steps = self.agent.total_steps();
        let target_steps = start_steps + n_steps;

        loop {
            let metrics = self.step_once();
            let total = metrics.total_steps;

            if metrics.episode_done {
                let record = EpisodeRecord::new(
                    metrics.episode_reward,
                    metrics.episode_step,
                    metrics.episode_status.clone(),
                );
                self.stats.update(&record);
                if let Some(run) = &self.run {
                    let _ = run.log_train_episode(&record);
                }
            }

            // Periodic checkpoint
            if let Some(run) = &mut self.run {
                if total.is_multiple_of(self.checkpoint_freq) {
                    let path = run.checkpoint_path(total);
                    // Strip the .mpk extension — DqnAgent::save appends it
                    let path_no_ext = path.with_extension("");
                    let _ = self.agent.save(&path_no_ext);
                    let _ = self.agent.save(run.latest_checkpoint_path().with_extension(""));
                    let _ = run.prune_checkpoints(self.keep_checkpoints);
                    let _ = run.update_metadata(total, self.episode);
                }
            }

            if total >= target_steps {
                break;
            }
        }
    }

    /// Run `n_episodes` of greedy evaluation and return an `EvalReport`.
    ///
    /// Exploration is disabled (ε = 0). If a `TrainingRun` is attached,
    /// each episode record is written to `eval_episodes.jsonl`.
    /// If the mean reward improves, saves a `best.mpk` checkpoint.
    pub fn eval(&mut self, n_episodes: usize) -> EvalReport {
        let total_steps = self.agent.total_steps();
        let mut eval_stats = StatsTracker::new();
        let mut records = Vec::with_capacity(n_episodes);

        for _ in 0..n_episodes {
            let record = self.run_greedy_episode();
            eval_stats.update(&record);
            records.push(record);
        }

        let summary = eval_stats.summary();
        let mean_reward = summary.get("episode_reward").copied().unwrap_or(f64::NAN);

        // Save best checkpoint
        if mean_reward > self.best_eval_reward {
            self.best_eval_reward = mean_reward;
            if let Some(run) = &self.run {
                let _ = self.agent.save(run.best_checkpoint_path().with_extension(""));
            }
        }

        // Log eval episodes
        if let Some(run) = &self.run {
            for record in &records {
                let _ = run.log_eval_episode(record, total_steps);
            }
        }

        EvalReport::new(total_steps, n_episodes, summary)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Perform one step. Called by `TrainIter::next()`.
    fn step_once(&mut self) -> StepMetrics {
        // Initialise the first episode
        if self.current_obs.is_none() {
            let (obs, _info) = self.env.reset(Some(0));
            self.current_obs = Some(obs);
            self.episode = 0;
            self.episode_step = 0;
            self.episode_reward = 0.0;
        }

        let obs = self.current_obs.clone().unwrap();

        // ε-greedy action selection
        let action = self.agent.act_epsilon_greedy(&obs, &mut self.rng);
        let epsilon = self.agent.epsilon();

        // Step environment
        let result = self.env.step(action.clone());
        let reward = result.reward;
        let done = result.is_done();

        self.episode_reward += reward;
        self.episode_step += 1;

        // Store experience
        let experience = Experience::new(
            obs,
            action,
            reward,
            result.observation.clone(),
            result.status.clone(),
        );
        let did_update = self.agent.observe(experience);

        let metrics = StepMetrics {
            total_steps: self.agent.total_steps(),
            episode: self.episode,
            episode_step: self.episode_step,
            reward,
            episode_reward: self.episode_reward,
            epsilon,
            did_update,
            episode_done: done,
            episode_status: result.status.clone(),
        };

        // Handle episode boundary
        if done {
            let (next_obs, _info) = self.env.reset(None);
            self.current_obs = Some(next_obs);
            self.episode += 1;
            self.episode_step = 0;
            self.episode_reward = 0.0;
        } else {
            self.current_obs = Some(result.observation);
        }

        metrics
    }

    /// Run one full episode greedily (no exploration). Returns the episode record.
    fn run_greedy_episode(&mut self) -> EpisodeRecord {
        let (mut obs, _) = self.env.reset(None);
        let mut total_reward = 0.0;
        let mut length = 0;

        loop {
            let action = self.agent.act(&obs);
            let result = self.env.step(action);
            total_reward += result.reward;
            length += 1;

            if result.is_done() {
                return EpisodeRecord::new(total_reward, length, result.status);
            }
            obs = result.observation;
        }
    }
}

// ── TrainIter ─────────────────────────────────────────────────────────────────

/// The iterator returned by `DqnTrainer::steps()`.
pub struct TrainIter<'a, E, Enc, Act, B, Buf = CircularBuffer<
    <E as Environment>::Observation,
    <E as Environment>::Action,
>>
where
    E: Environment,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    trainer: &'a mut DqnTrainer<E, Enc, Act, B, Buf>,
}

impl<'a, E, Enc, Act, B, Buf> Iterator for TrainIter<'a, E, Enc, Act, B, Buf>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B>
        + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    type Item = StepMetrics;

    fn next(&mut self) -> Option<StepMetrics> {
        // The iterator is infinite — training stops when the caller stops
        // consuming it (e.g. via `.take(n)` or a manual break).
        Some(self.trainer.step_once())
    }
}


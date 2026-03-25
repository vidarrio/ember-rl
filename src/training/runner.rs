use burn::tensor::backend::AutodiffBackend;
use rl_traits::{Environment, Experience};

use crate::algorithms::dqn::{CircularBuffer, DqnAgent};
use crate::encoding::{DiscreteActionMapper, ObservationEncoder};
use crate::stats::{EpisodeRecord, EvalReport, StatsTracker};
use crate::traits::ActMode;
use crate::training::run::TrainingRun;
use crate::training::session::TrainingSession;
use rl_traits::ReplayBuffer;

/// Metrics emitted after every environment step.
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

    /// Current ε (exploration rate).
    pub epsilon: f64,

    /// Whether this step ended the episode.
    pub episode_done: bool,

    /// How the episode ended (only meaningful when `episode_done` is `true`).
    pub episode_status: rl_traits::EpisodeStatus,
}

/// The imperative training driver for DQN.
///
/// A thin convenience wrapper around [`TrainingSession`] that adds an
/// environment and drives the training loop. Use [`TrainingSession`] directly
/// when your loop is owned externally (e.g. Bevy's ECS).
///
/// # Usage -- iterator style
///
/// ```rust,ignore
/// let mut trainer = DqnTrainer::new(env, agent);
///
/// for step in trainer.steps().take(50_000) {
///     if step.episode_done {
///         println!("Episode {} reward: {}", step.episode, step.episode_reward);
///     }
/// }
/// ```
///
/// # Usage -- imperative style with run tracking
///
/// ```rust,ignore
/// let mut trainer = DqnTrainer::new(env, agent)
///     .with_run(TrainingRun::create("cartpole", "v1")?)
///     .with_max_steps(200_000);
///
/// trainer.train();
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
    session: TrainingSession<E, DqnAgent<E, Enc, Act, B, Buf>>,

    // Episode state
    current_obs: Option<E::Observation>,
    episode: usize,
    episode_step: usize,
    episode_reward: f64,
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
    pub fn new(env: E, agent: DqnAgent<E, Enc, Act, B, Buf>) -> Self {
        Self {
            env,
            session: TrainingSession::new(agent),
            current_obs: None,
            episode: 0,
            episode_step: 0,
            episode_reward: 0.0,
        }
    }

    /// Attach a `TrainingRun` for checkpoint saving and JSONL logging.
    pub fn with_run(mut self, run: TrainingRun) -> Self {
        self.session = self.session.with_run(run);
        self
    }

    /// Maximum training steps. `train()` stops when this is reached.
    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.session = self.session.with_max_steps(n);
        self
    }

    /// Checkpoint frequency in steps. Default: 10_000.
    pub fn with_checkpoint_freq(mut self, freq: usize) -> Self {
        self.session = self.session.with_checkpoint_freq(freq);
        self
    }

    /// Number of numbered checkpoints to keep on disk. Default: 3.
    pub fn with_keep_checkpoints(mut self, keep: usize) -> Self {
        self.session = self.session.with_keep_checkpoints(keep);
        self
    }

    /// Replace the default stats tracker.
    pub fn with_stats(mut self, stats: StatsTracker) -> Self {
        self.session = self.session.with_stats(stats);
        self
    }

    /// Returns an iterator that yields `StepMetrics` after each environment step.
    pub fn steps(&mut self) -> TrainIter<'_, E, Enc, Act, B, Buf> {
        TrainIter { trainer: self }
    }

    /// Run until `max_steps` is reached (or forever if not set).
    pub fn train(&mut self) {
        loop {
            self.step_once();
            if self.session.is_done() {
                break;
            }
        }
    }

    /// Run `n_episodes` of greedy evaluation and return an `EvalReport`.
    pub fn eval(&mut self, n_episodes: usize) -> EvalReport {
        let mut eval_stats = StatsTracker::new();
        let mut records = Vec::with_capacity(n_episodes);

        for _ in 0..n_episodes {
            let record = self.run_greedy_episode();
            eval_stats.update(&record);
            self.session.on_eval_episode(&record);
            records.push(record);
        }

        let summary = eval_stats.summary();
        let mean_reward = summary.get("episode_reward").copied().unwrap_or(f64::NAN);
        self.session.maybe_save_best(mean_reward);

        let total_steps = self.session.total_steps();
        EvalReport::new(total_steps, n_episodes, summary)
    }

    /// Read-only access to the underlying session.
    pub fn session(&self) -> &TrainingSession<E, DqnAgent<E, Enc, Act, B, Buf>> {
        &self.session
    }

    /// Consume the trainer and return the inner agent.
    pub fn into_agent(self) -> DqnAgent<E, Enc, Act, B, Buf> {
        self.session.into_agent()
    }

    /// Access the environment directly.
    pub fn env(&self) -> &E {
        &self.env
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn step_once(&mut self) -> StepMetrics {
        if self.current_obs.is_none() {
            let (obs, _) = self.env.reset(Some(0));
            self.current_obs = Some(obs);
            self.episode = 0;
            self.episode_step = 0;
            self.episode_reward = 0.0;
            self.session.on_episode_start();
        }

        let obs = self.current_obs.clone().unwrap();
        let epsilon = self.session.agent().epsilon();
        let action = self.session.act(&obs, ActMode::Explore);

        let result = self.env.step(action.clone());
        let reward = result.reward;
        let done = result.is_done();

        self.episode_reward += reward;
        self.episode_step += 1;

        self.session.observe(Experience::new(
            obs,
            action,
            reward,
            result.observation.clone(),
            result.status.clone(),
        ));

        let metrics = StepMetrics {
            total_steps: self.session.total_steps(),
            episode: self.episode,
            episode_step: self.episode_step,
            reward,
            episode_reward: self.episode_reward,
            epsilon,
            episode_done: done,
            episode_status: result.status.clone(),
        };

        if done {
            self.session.on_episode(
                self.episode_reward,
                self.episode_step,
                result.status,
                self.env.episode_extras(),
            );
            let (next_obs, _) = self.env.reset(None);
            self.current_obs = Some(next_obs);
            self.episode += 1;
            self.episode_step = 0;
            self.episode_reward = 0.0;
        } else {
            self.current_obs = Some(result.observation);
        }

        metrics
    }

    fn run_greedy_episode(&mut self) -> EpisodeRecord {
        let (mut obs, _) = self.env.reset(None);
        let mut total_reward = 0.0;
        let mut length = 0;

        loop {
            let action = self.session.act(&obs, ActMode::Exploit);
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
        Some(self.trainer.step_once())
    }
}

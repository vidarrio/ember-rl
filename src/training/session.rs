use std::collections::HashMap;

use rl_traits::{Environment, Experience};

use crate::stats::{EpisodeRecord, EvalReport, StatsTracker};
use crate::traits::{ActMode, LearningAgent};
use crate::training::run::TrainingRun;


/// Configuration for a `TrainingSession`.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Stop when `total_steps >= max_steps`. Default: `usize::MAX` (no limit).
    pub max_steps: usize,

    /// Save a numbered checkpoint every this many steps. Default: 10_000.
    pub checkpoint_freq: usize,

    /// Number of recent numbered checkpoints to keep on disk. Default: 3.
    pub keep_checkpoints: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_steps: usize::MAX,
            checkpoint_freq: 10_000,
            keep_checkpoints: 3,
        }
    }
}

/// A self-contained, loop-agnostic training coordinator.
///
/// `TrainingSession` wires together a [`LearningAgent`], an optional
/// [`TrainingRun`], and a [`StatsTracker`]. It is driven purely by incoming
/// data -- it does not own a training loop. Feed it experiences and episode
/// boundaries from wherever your loop lives: a plain `for` loop, Bevy's ECS,
/// or anything else.
///
/// # Usage
///
/// ```rust,ignore
/// let session = TrainingSession::new(agent)
///     .with_run(TrainingRun::create("cartpole", "v1")?)
///     .with_max_steps(200_000)
///     .with_checkpoint_freq(10_000);
///
/// // Each environment step:
/// session.observe(experience);
///
/// // Each episode end:
/// session.on_episode(total_reward, steps, status, env_extras);
///
/// if session.is_done() { break; }
/// ```
pub struct TrainingSession<E: Environment, A> {
    agent: A,
    run: Option<TrainingRun>,
    stats: StatsTracker,
    config: SessionConfig,
    best_eval_reward: f64,
    _env: std::marker::PhantomData<E>,
}

impl<E, A> TrainingSession<E, A>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    A: LearningAgent<E>,
{
    /// Create a session with no run attached.
    ///
    /// Stats are tracked in memory but nothing is persisted. Attach a run with
    /// [`with_run`] to enable checkpointing and JSONL logging.
    ///
    /// [`with_run`]: TrainingSession::with_run
    pub fn new(agent: A) -> Self {
        Self {
            agent,
            run: None,
            stats: StatsTracker::new(),
            config: SessionConfig::default(),
            best_eval_reward: f64::NEG_INFINITY,
            _env: std::marker::PhantomData,
        }
    }

    /// Attach a `TrainingRun` for checkpointing and JSONL episode logging.
    pub fn with_run(mut self, run: TrainingRun) -> Self {
        self.run = Some(run);
        self
    }

    /// Maximum number of steps before `is_done()` returns `true`. Default: no limit.
    pub fn with_max_steps(mut self, n: usize) -> Self {
        self.config.max_steps = n;
        self
    }

    /// Checkpoint frequency in steps. Default: 10_000.
    pub fn with_checkpoint_freq(mut self, freq: usize) -> Self {
        self.config.checkpoint_freq = freq;
        self
    }

    /// Number of numbered checkpoints to retain on disk. Default: 3.
    pub fn with_keep_checkpoints(mut self, keep: usize) -> Self {
        self.config.keep_checkpoints = keep;
        self
    }

    /// Replace the default `StatsTracker` with a custom one.
    pub fn with_stats(mut self, stats: StatsTracker) -> Self {
        self.stats = stats;
        self
    }

    // ── Data ingestion ────────────────────────────────────────────────────────

    /// Select an action for `obs` according to `mode`.
    pub fn act(&mut self, obs: &E::Observation, mode: ActMode) -> E::Action {
        self.agent.act(obs, mode)
    }

    /// Record a transition. Checkpoints + prunes if a step milestone is hit.
    pub fn observe(&mut self, experience: Experience<E::Observation, E::Action>) {
        self.agent.observe(experience);

        let total = self.agent.total_steps();
        if total > 0 && total.is_multiple_of(self.config.checkpoint_freq) {
            if let Some(run) = &mut self.run {
                let path = run.checkpoint_path(total).with_extension("");
                let _ = self.agent.save(&path);
                let latest = run.latest_checkpoint_path().with_extension("");
                let _ = self.agent.save(&latest);
                let _ = run.prune_checkpoints(self.config.keep_checkpoints);
            }
        }
    }

    /// Record an episode boundary.
    ///
    /// Merges agent and environment extras into the record, updates stats,
    /// and appends to the training JSONL log (if a run is attached).
    ///
    /// `env_extras` should come from [`crate::traits::EpisodeStats::episode_extras`]
    /// if the environment implements it, or an empty map otherwise.
    pub fn on_episode(
        &mut self,
        total_reward: f64,
        steps: usize,
        status: rl_traits::EpisodeStatus,
        env_extras: HashMap<String, f64>,
    ) {
        let agent_extras = self.agent.episode_extras();
        let record = EpisodeRecord::new(total_reward, steps, status)
            .with_extras(env_extras)
            .with_extras(agent_extras);

        self.stats.update(&record);

        if let Some(run) = &mut self.run {
            let _ = run.log_train_episode(&record);
            let _ = run.update_metadata(self.agent.total_steps(), 0);
        }

        self.agent.on_episode_start();
    }

    /// Signal the start of a new episode (resets per-episode agent aggregators).
    pub fn on_episode_start(&mut self) {
        self.agent.on_episode_start();
    }

    /// Total environment steps observed so far.
    pub fn total_steps(&self) -> usize {
        self.agent.total_steps()
    }

    /// Returns `true` when `total_steps >= max_steps`.
    pub fn is_done(&self) -> bool {
        self.config.max_steps != usize::MAX
            && self.agent.total_steps() >= self.config.max_steps
    }

    // ── Eval ──────────────────────────────────────────────────────────────────

    /// Log an eval episode to the run (if attached).
    pub fn on_eval_episode(&self, record: &EpisodeRecord) {
        if let Some(run) = &self.run {
            let _ = run.log_eval_episode(record, self.agent.total_steps());
        }
    }

    /// Save `best.mpk` if `mean_reward` exceeds the best seen so far.
    pub fn maybe_save_best(&mut self, mean_reward: f64) {
        if mean_reward > self.best_eval_reward {
            self.best_eval_reward = mean_reward;
            if let Some(run) = &self.run {
                let best = run.best_checkpoint_path().with_extension("");
                let _ = self.agent.save(&best);
            }
        }
    }

    // ── Access ────────────────────────────────────────────────────────────────

    /// Read-only access to the agent.
    pub fn agent(&self) -> &A {
        &self.agent
    }

    /// Mutable access to the agent.
    pub fn agent_mut(&mut self) -> &mut A {
        &mut self.agent
    }

    /// Current stats summary.
    pub fn stats_summary(&self) -> HashMap<String, f64> {
        self.stats.summary()
    }

    /// Read-only access to the run (if attached).
    pub fn run(&self) -> Option<&TrainingRun> {
        self.run.as_ref()
    }

    /// Consume the session and return the inner agent.
    pub fn into_agent(self) -> A {
        self.agent
    }

    /// Snapshot the current stats as an `EvalReport`.
    pub fn eval_report(&self, n_episodes: usize) -> EvalReport {
        EvalReport::new(self.agent.total_steps(), n_episodes, self.stats.summary())
    }
}

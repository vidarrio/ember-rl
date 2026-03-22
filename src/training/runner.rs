use burn::tensor::backend::AutodiffBackend;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rl_traits::{Environment, Experience};

use crate::algorithms::dqn::DqnAgent;
use crate::encoding::{DiscreteActionMapper, ObservationEncoder};

/// Metrics emitted after every environment step.
///
/// The runner yields one of these per call to `Iterator::next()`.
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
}

/// The imperative training runner.
///
/// Drives the interaction between an environment and a DQN agent,
/// exposing it as an iterator that yields `StepMetrics` after every step.
///
/// # Usage
///
/// ```rust,ignore
/// let mut runner = DqnRunner::new(env, agent, seed);
///
/// for step in runner.steps().take(50_000) {
///     if step.episode_done {
///         println!("Episode {} reward: {}", step.episode, step.episode_reward);
///     }
/// }
/// ```
///
/// # Why an iterator?
///
/// - You control the loop: add early stopping, custom logging, checkpointing
/// - bevy-gym can drive the same runner one step per ECS tick
/// - No callbacks, no closures, no inversion of control
pub struct DqnRunner<E, Enc, Act, B>
where
    E: Environment,
    B: AutodiffBackend,
{
    env: E,
    agent: DqnAgent<E, Enc, Act, B>,
    rng: SmallRng,

    // Episode state
    current_obs: Option<E::Observation>,
    episode: usize,
    episode_step: usize,
    episode_reward: f64,
}

impl<E, Enc, Act, B> DqnRunner<E, Enc, Act, B>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B>
        + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
{
    pub fn new(env: E, agent: DqnAgent<E, Enc, Act, B>, seed: u64) -> Self {
        Self {
            env,
            agent,
            rng: SmallRng::seed_from_u64(seed),
            current_obs: None,
            episode: 0,
            episode_step: 0,
            episode_reward: 0.0,
        }
    }

    /// Returns an iterator that yields `StepMetrics` after each environment step.
    pub fn steps(&mut self) -> StepIter<'_, E, Enc, Act, B> {
        StepIter { runner: self }
    }

    /// Access the agent for evaluation or inspection.
    pub fn agent(&self) -> &DqnAgent<E, Enc, Act, B> {
        &self.agent
    }

    /// Access the environment directly.
    pub fn env(&self) -> &E {
        &self.env
    }

    /// Perform one step. Called by `StepIter::next()`.
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
}

/// The iterator returned by `DqnRunner::steps()`.
pub struct StepIter<'a, E, Enc, Act, B>
where
    E: Environment,
    B: AutodiffBackend,
{
    runner: &'a mut DqnRunner<E, Enc, Act, B>,
}

impl<'a, E, Enc, Act, B> Iterator for StepIter<'a, E, Enc, Act, B>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B>
        + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
{
    type Item = StepMetrics;

    fn next(&mut self) -> Option<StepMetrics> {
        // The iterator is infinite — training stops when the caller stops
        // consuming it (e.g. via `.take(n)` or a manual break).
        Some(self.runner.step_once())
    }
}

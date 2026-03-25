//! Core training traits for ember-rl.
//!
//! These traits define the composable building blocks that all algorithms
//! and environments plug into. They live in ember-rl (not rl-traits) because
//! they are training machinery, not environment contract.

use std::collections::HashMap;
use std::path::Path;

use rl_traits::{Environment, Experience};

// ── ActMode ───────────────────────────────────────────────────────────────────

/// Controls whether an agent acts to explore or exploit.
///
/// Passed to [`LearningAgent::act`] to select the agent's action strategy.
/// Algorithms interpret this mode internally -- DQN uses epsilon-greedy for
/// `Explore` and greedy argmax for `Exploit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActMode {
    /// Act to explore: use the algorithm's exploration strategy (e.g. ε-greedy).
    Explore,
    /// Act to exploit: select the greedy/best-known action.
    Exploit,
}

// ── Checkpointable ────────────────────────────────────────────────────────────

/// An agent whose weights can be saved to and loaded from disk.
pub trait Checkpointable: Sized {
    /// Save weights to `path` (without extension -- implementations add their own).
    fn save(&self, path: &Path) -> anyhow::Result<()>;

    /// Load weights from `path`, consuming and returning `self`.
    fn load(self, path: &Path) -> anyhow::Result<Self>;
}

// ── LearningAgent ─────────────────────────────────────────────────────────────

/// An agent that can act, learn from experience, and report training stats.
///
/// Implemented by all algorithm agents (`DqnAgent`, future `PpoAgent`, etc.).
/// The agent owns its exploration RNG internally -- no external RNG is needed
/// at call sites.
///
/// # Episode extras
///
/// Algorithms should maintain internal aggregators (e.g. `Mean`, `Std`, `Max`
/// from [`crate::stats`]) over per-step values during each episode, reset them
/// at episode start, and report summaries via [`episode_extras`]. These are
/// merged into [`crate::stats::EpisodeRecord::extras`] automatically by
/// [`crate::training::TrainingSession`].
///
/// Example extras a DQN agent might report:
/// ```text
/// { "epsilon": 0.12, "loss_mean": 0.043, "loss_std": 0.012, "loss_max": 0.21 }
/// ```
///
/// [`episode_extras`]: LearningAgent::episode_extras
pub trait LearningAgent<E: Environment>: Checkpointable {
    /// Select an action for `obs` according to `mode`.
    fn act(&mut self, obs: &E::Observation, mode: ActMode) -> E::Action;

    /// Record a transition and update the agent's internal state.
    fn observe(&mut self, experience: Experience<E::Observation, E::Action>);

    /// Total number of `observe` calls since construction.
    fn total_steps(&self) -> usize;

    /// Per-episode aggregates of step-level values, reported at episode end.
    ///
    /// The default implementation returns an empty map. Algorithms override
    /// this to expose training dynamics (loss statistics, epsilon, etc.).
    fn episode_extras(&self) -> HashMap<String, f64> {
        HashMap::new()
    }

    /// Called by [`crate::training::TrainingSession`] at the start of each
    /// episode so the agent can reset its per-episode aggregators.
    fn on_episode_start(&mut self) {}
}


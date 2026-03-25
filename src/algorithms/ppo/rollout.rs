use rand::seq::SliceRandom;
use rand::rngs::SmallRng;

/// One stored transition from the rollout.
pub struct Transition<O> {
    pub obs: O,
    pub action: usize,
    pub reward: f32,
    pub done: bool,       // true if the episode ended on this step
    pub value: f32,       // critic estimate at obs (from act() forward pass)
    pub log_prob: f32,    // log π(action | obs) under the policy at collection time
}

/// Flat minibatch ready for the PPO update.
///
/// All vecs are the same length (= batch_size). `obs` is kept generic so
/// we can encode it lazily in the agent's update step rather than storing
/// pre-encoded tensors (which would require a Backend reference here).
pub struct Batch<O> {
    pub obs: Vec<O>,
    pub actions: Vec<usize>,
    pub old_log_probs: Vec<f32>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}

/// On-policy rollout buffer.
///
/// Collects exactly `capacity` transitions from the current policy, then
/// computes GAE advantages and exposes shuffled minibatches for training.
/// Cleared after each PPO update cycle.
pub struct RolloutBuffer<O> {
    transitions: Vec<Transition<O>>,
    capacity: usize,

    // Computed by compute_gae(); empty until then.
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

impl<O: Clone> RolloutBuffer<O> {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            capacity,
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }

    pub fn push(&mut self, t: Transition<O>) {
        self.transitions.push(t);
    }

    pub fn is_full(&self) -> bool {
        self.transitions.len() >= self.capacity
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Compute GAE advantages and discounted returns.
    ///
    /// `bootstrap_value` is the critic's estimate for the state *after* the
    /// last stored transition -- used to bootstrap the final advantage.
    /// Pass `0.0` if the last transition ended an episode (no future value).
    pub fn compute_gae(&mut self, bootstrap_value: f32, gamma: f32, lambda: f32) {
        let n = self.transitions.len();
        self.advantages = vec![0.0; n];
        self.returns = vec![0.0; n];

        let mut last_gae = 0.0f32;
        let mut next_value = bootstrap_value;

        for i in (0..n).rev() {
            let t = &self.transitions[i];
            let mask = if t.done { 0.0 } else { 1.0 };
            let delta = t.reward + gamma * next_value * mask - t.value;
            last_gae = delta + gamma * lambda * mask * last_gae;
            self.advantages[i] = last_gae;
            self.returns[i] = last_gae + t.value;
            next_value = t.value;
        }
    }

    /// Normalize advantages to zero mean, unit variance.
    ///
    /// Standard practice for PPO -- stabilises learning.
    pub fn normalize_advantages(&mut self) {
        let n = self.advantages.len() as f32;
        let mean = self.advantages.iter().sum::<f32>() / n;
        let var = self.advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt() + 1e-8;
        for a in &mut self.advantages {
            *a = (*a - mean) / std;
        }
    }

    /// Yield shuffled minibatches of size `batch_size`.
    ///
    /// Must be called after `compute_gae()`. Panics if GAE hasn't been run.
    pub fn minibatches(&self, batch_size: usize, rng: &mut SmallRng) -> Vec<Batch<O>> {
        assert!(!self.advantages.is_empty(), "call compute_gae() before minibatches()");

        let n = self.transitions.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(rng);

        indices
            .chunks(batch_size)
            .filter(|chunk| chunk.len() == batch_size) // drop incomplete last chunk
            .map(|chunk| Batch {
                obs:           chunk.iter().map(|&i| self.transitions[i].obs.clone()).collect(),
                actions:       chunk.iter().map(|&i| self.transitions[i].action).collect(),
                old_log_probs: chunk.iter().map(|&i| self.transitions[i].log_prob).collect(),
                advantages:    chunk.iter().map(|&i| self.advantages[i]).collect(),
                returns:       chunk.iter().map(|&i| self.returns[i]).collect(),
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.transitions.clear();
        self.advantages.clear();
        self.returns.clear();
    }

    /// Value estimate stored for the last transition (used as bootstrap fallback).
    #[allow(dead_code)]
    pub fn last_value(&self) -> f32 {
        self.transitions.last().map(|t| t.value).unwrap_or(0.0)
    }

    /// Whether the last stored transition ended an episode.
    pub fn last_done(&self) -> bool {
        self.transitions.last().map(|t| t.done).unwrap_or(false)
    }
}

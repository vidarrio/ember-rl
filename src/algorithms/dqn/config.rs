/// Configuration for a DQN agent.
///
/// All hyperparameters live here. Pass this to `DqnAgent::new()`.
/// The defaults reflect standard DQN practice suitable for moderately
/// complex environments. Simple environments like CartPole will want
/// smaller buffer/warmup values and faster epsilon decay.
#[derive(Debug, Clone)]
pub struct DqnConfig {
    /// Discount factor γ. Controls how much future rewards are valued.
    /// Typical values: 0.95–0.999. Default: 0.99
    pub gamma: f64,

    /// Learning rate for the Adam optimiser.
    /// Default: 1e-4
    pub learning_rate: f64,

    /// Number of experiences sampled per gradient update.
    /// Default: 32
    pub batch_size: usize,

    /// Maximum number of experiences in the replay buffer.
    /// Oldest are overwritten when full.
    /// Default: 100_000
    pub buffer_capacity: usize,

    /// Minimum number of experiences collected before training begins.
    /// During warm-up, actions are sampled randomly.
    /// Must be >= batch_size. Default: 10_000
    pub min_replay_size: usize,

    /// Number of steps between hard target network updates.
    ///
    /// The target network is a frozen copy of the online network used to
    /// compute stable TD targets. Updating it too frequently causes
    /// instability; too rarely slows learning. Default: 1_000
    pub target_update_freq: usize,

    /// Hidden layer sizes for the Q-network.
    ///
    /// The network architecture is:
    /// `obs_size -> hidden[0] -> hidden[1] -> ... -> num_actions`
    /// All hidden layers use ReLU activations.
    /// Default: [128, 128]
    pub hidden_sizes: Vec<usize>,

    /// Starting epsilon for ε-greedy exploration.
    /// At step 0, actions are random with this probability.
    /// Default: 1.0
    pub epsilon_start: f64,

    /// Final epsilon after decay is complete.
    /// Default: 0.05
    pub epsilon_end: f64,

    /// Number of steps over which epsilon decays linearly from
    /// `epsilon_start` to `epsilon_end`. Default: 50_000
    pub epsilon_decay_steps: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            learning_rate: 1e-4,
            batch_size: 32,
            buffer_capacity: 100_000,
            min_replay_size: 10_000,
            target_update_freq: 1_000,
            hidden_sizes: vec![128, 128],
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 50_000,
        }
    }
}

impl DqnConfig {
    /// Compute the current epsilon given the number of elapsed steps.
    ///
    /// Decays linearly from `epsilon_start` to `epsilon_end` over
    /// `epsilon_decay_steps`, then stays flat.
    pub fn epsilon_at(&self, step: usize) -> f64 {
        if step >= self.epsilon_decay_steps {
            return self.epsilon_end;
        }
        let progress = step as f64 / self.epsilon_decay_steps as f64;
        self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_at_zero() {
        let config = DqnConfig::default();
        assert_eq!(config.epsilon_at(0), config.epsilon_start);
    }

    #[test]
    fn epsilon_at_end() {
        let config = DqnConfig::default();
        assert_eq!(config.epsilon_at(config.epsilon_decay_steps), config.epsilon_end);
    }

    #[test]
    fn epsilon_past_end_is_clamped() {
        let config = DqnConfig::default();
        assert_eq!(config.epsilon_at(config.epsilon_decay_steps * 10), config.epsilon_end);
    }

    #[test]
    fn epsilon_midpoint() {
        let config = DqnConfig::default();
        let mid = config.epsilon_decay_steps / 2;
        let expected = (config.epsilon_start + config.epsilon_end) / 2.0;
        let actual = config.epsilon_at(mid);
        assert!((actual - expected).abs() < 1e-6);
    }
}

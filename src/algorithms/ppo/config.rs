/// Hyperparameters for the PPO algorithm.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PpoConfig {
    // -- Rollout collection --
    /// Steps collected per environment before each update.
    /// Total rollout size = n_steps * n_envs.
    pub n_steps: usize,

    /// Number of parallel environments feeding this agent.
    /// Must match the number of envs in your BevyGymPlugin / training loop.
    pub n_envs: usize,

    // -- Update --
    /// Number of gradient epochs over each rollout. Typical: 4-10.
    pub n_epochs: usize,

    /// Minibatch size for each gradient step. Must divide n_steps * n_envs evenly.
    pub batch_size: usize,

    pub learning_rate: f64,

    // -- PPO objective --
    /// Clipping range for the probability ratio. Typical: 0.1-0.3.
    pub clip_epsilon: f64,

    /// Weight on the value function loss. Typical: 0.5.
    pub value_loss_coef: f64,

    /// Weight on the entropy bonus (encourages exploration). Typical: 0.01.
    pub entropy_coef: f64,

    // -- Returns / advantages --
    /// Discount factor.
    pub gamma: f64,

    /// GAE smoothing parameter. 1.0 = full Monte Carlo, 0.0 = TD(0).
    pub gae_lambda: f64,

    // -- Network --
    /// Hidden layer sizes for the shared trunk.
    pub hidden_sizes: Vec<usize>,

    /// Clip gradient norm to this value. Set to 0.0 to disable.
    pub max_grad_norm: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            n_steps: 128,
            n_envs: 1,
            n_epochs: 4,
            batch_size: 64,
            learning_rate: 2.5e-4,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            hidden_sizes: vec![64, 64],
            max_grad_norm: 0.5,
        }
    }
}

impl PpoConfig {
    /// Total number of transitions collected before each update.
    pub fn rollout_size(&self) -> usize {
        self.n_steps * self.n_envs
    }
}

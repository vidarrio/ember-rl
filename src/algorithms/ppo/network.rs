use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Relu};

/// Shared-trunk actor-critic network.
///
/// Architecture: fully-connected trunk with ReLU activations, then two
/// independent linear heads:
/// - **Actor head**: outputs one logit per discrete action (`[batch, n_actions]`)
/// - **Critic head**: outputs a single state-value estimate (`[batch, 1]`)
///
/// Both heads read from the final trunk activation, so they share a common
/// representation while keeping their output scales independent.
#[derive(Module, Debug)]
pub struct ActorCriticNetwork<B: Backend> {
    trunk: Vec<Linear<B>>,
    actor_head: Linear<B>,
    critic_head: Linear<B>,
    activation: Relu,
}

impl<B: Backend> ActorCriticNetwork<B> {
    /// Build the network.
    ///
    /// `layer_sizes` is the trunk: `[obs_size, hidden_0, hidden_1, ...]`.
    /// The actor and critic heads are added on top automatically.
    pub fn new(layer_sizes: &[usize], n_actions: usize, device: &B::Device) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least obs_size and one hidden layer");

        let trunk = layer_sizes
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();

        let trunk_out = *layer_sizes.last().unwrap();
        let actor_head = LinearConfig::new(trunk_out, n_actions).init(device);
        let critic_head = LinearConfig::new(trunk_out, 1).init(device);

        Self { trunk, actor_head, critic_head, activation: Relu::new() }
    }

    /// Forward pass.
    ///
    /// Returns `(action_logits, value)`:
    /// - `action_logits`: `[batch, n_actions]` -- raw logits, apply softmax for probabilities
    /// - `value`: `[batch]` -- state-value estimates
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let mut out = x;
        for layer in &self.trunk {
            out = self.activation.forward(layer.forward(out));
        }
        let logits = self.actor_head.forward(out.clone());
        let value_2d = self.critic_head.forward(out); // [batch, 1]
        let batch = value_2d.dims()[0];
        let value = value_2d.reshape([batch]);         // [batch]
        (logits, value)
    }
}

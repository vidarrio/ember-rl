use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Relu};

/// A feedforward Q-network.
///
/// Maps observations to Q-values: `[batch, obs_size] -> [batch, num_actions]`
///
/// Architecture: fully connected layers with ReLU activations between them.
/// Layer sizes are determined at construction from `DqnConfig::hidden_sizes`.
///
/// This is a Burn `Module`, meaning it owns its parameters and can be
/// serialised, cloned for the target network, and updated by an optimiser.
///
/// # Target network
///
/// DQN requires two copies of the network: the online network (updated every
/// step) and the target network (periodically synced from online). Burn's
/// `Module::clone()` gives us the target network for free -- it performs a
/// deep clone of all parameters.
#[derive(Module, Debug)]
pub struct QNetwork<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
}

impl<B: Backend> QNetwork<B> {
    /// Build a Q-network with the given layer sizes.
    ///
    /// `layer_sizes` should be the full sequence from input to output:
    /// `[obs_size, hidden_0, hidden_1, ..., num_actions]`
    pub fn new(layer_sizes: &[usize], device: &B::Device) -> Self {
        assert!(layer_sizes.len() >= 2, "need at least input and output sizes");

        let layers = layer_sizes
            .windows(2)
            .map(|w| LinearConfig::new(w[0], w[1]).init(device))
            .collect();

        Self {
            layers,
            activation: Relu::new(),
        }
    }

    /// Forward pass: observation batch -> Q-values.
    ///
    /// Input:  `[batch_size, obs_size]`
    /// Output: `[batch_size, num_actions]`
    ///
    /// ReLU is applied between all layers except the final output layer,
    /// which is linear (Q-values are unbounded).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let last_idx = self.layers.len() - 1;
        let mut out = x;

        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(out);
            if i < last_idx {
                out = self.activation.forward(out);
            }
        }

        out
    }
}

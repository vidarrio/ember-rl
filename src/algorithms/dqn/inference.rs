use std::marker::PhantomData;
use std::path::Path;

use burn::prelude::*;
use burn::record::{CompactRecorder, RecorderError};
use rl_traits::{Environment, Policy};

use crate::encoding::{DiscreteActionMapper, ObservationEncoder};
use super::config::DqnConfig;
use super::network::QNetwork;

/// A DQN agent in inference-only mode.
///
/// Holds just the Q-network, encoder, and action mapper -- no optimizer,
/// no replay buffer, no exploration. Requires only `B: Backend` (not
/// `AutodiffBackend`), so it can run on plain `NdArray` without any
/// autodiff overhead.
///
/// Load from a checkpoint saved by `DqnAgent::save`:
///
/// ```rust,ignore
/// use burn::backend::NdArray;
///
/// let policy = DqnPolicy::<CartPoleEnv, _, _, NdArray>::new(
///     VecEncoder::new(4),
///     UsizeActionMapper::new(2),
///     &config,
///     device,
/// )
/// .load("cartpole_dqn")?;
/// ```
pub struct DqnPolicy<E, Enc, Act, B: Backend> {
    net: QNetwork<B>,
    encoder: Enc,
    action_mapper: Act,
    device: B::Device,
    _env: PhantomData<E>,
}

impl<E, Enc, Act, B> DqnPolicy<E, Enc, Act, B>
where
    E: Environment,
    Enc: ObservationEncoder<E::Observation, B>,
    Act: DiscreteActionMapper<E::Action>,
    B: Backend,
{
    /// Build an uninitialised policy with the given architecture.
    ///
    /// The network weights are random until `load` is called.
    pub fn new(encoder: Enc, action_mapper: Act, config: &DqnConfig, device: B::Device) -> Self {
        let obs_size = encoder.obs_size();
        let num_actions = action_mapper.num_actions();

        let mut layer_sizes = vec![obs_size];
        layer_sizes.extend_from_slice(&config.hidden_sizes);
        layer_sizes.push(num_actions);

        let net = QNetwork::new(&layer_sizes, &device);

        Self {
            net,
            encoder,
            action_mapper,
            device,
            _env: PhantomData,
        }
    }

    /// Create a policy directly from a pre-built network.
    ///
    /// Used by `DqnAgent::into_policy()` to convert a trained agent into
    /// an inference-only policy without touching the disk.
    pub fn from_network(net: QNetwork<B>, encoder: Enc, action_mapper: Act, device: B::Device) -> Self {
        Self {
            net,
            encoder,
            action_mapper,
            device,
            _env: PhantomData,
        }
    }

    /// Load network weights from a checkpoint file.
    ///
    /// The checkpoint must have been saved with `DqnAgent::save` (`.mpk` format). The
    /// architecture (hidden sizes) must match exactly.
    pub fn load(mut self, path: impl AsRef<Path>) -> Result<Self, RecorderError> {
        self.net = self
            .net
            .load_file(path.as_ref().to_path_buf(), &CompactRecorder::new(), &self.device)?;
        Ok(self)
    }
}

impl<E, Enc, Act, B> Policy<E::Observation, E::Action> for DqnPolicy<E, Enc, Act, B>
where
    E: Environment,
    Enc: ObservationEncoder<E::Observation, B>,
    Act: DiscreteActionMapper<E::Action>,
    B: Backend,
{
    fn act(&self, obs: &E::Observation) -> E::Action {
        let obs_tensor = self.encoder.encode(obs, &self.device).unsqueeze_dim(0);
        let q_values = self.net.forward(obs_tensor);
        let best_idx = q_values
            .argmax(1)
            .into_data()
            .to_vec::<i64>()
            .unwrap()[0] as usize;
        self.action_mapper.index_to_action(best_idx)
    }
}

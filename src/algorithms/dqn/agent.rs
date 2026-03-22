use std::path::Path;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::record::{CompactRecorder, RecorderError};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rl_traits::{Environment, Experience, Policy};

use crate::encoding::{DiscreteActionMapper, ObservationEncoder};
use super::config::DqnConfig;
use super::network::QNetwork;
use super::replay::CircularBuffer;
use rl_traits::ReplayBuffer;

/// A DQN agent.
///
/// Implements ε-greedy action selection, experience replay, and TD learning
/// with a target network. Generic over:
///
/// - `E`: the environment type (must satisfy `rl_traits::Environment`)
/// - `Enc`: the observation encoder (converts `E::Observation` to tensors)
/// - `Act`: the action mapper (converts `E::Action` to/from integer indices)
/// - `B`: the Burn backend (e.g. `NdArray`, `Wgpu`)
/// - `Buf`: the replay buffer (defaults to `CircularBuffer` — swap for PER etc.)
///
/// # Usage
///
/// ```rust,ignore
/// let agent = DqnAgent::new(encoder, action_mapper, config, device, seed);
/// ```
///
/// Then hand it to `DqnRunner`, which drives the training loop.
pub struct DqnAgent<E, Enc, Act, B, Buf = CircularBuffer<
    <E as Environment>::Observation,
    <E as Environment>::Action,
>>
where
    E: Environment,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    // Network pair
    online_net: QNetwork<B>,
    target_net: QNetwork<B::InnerBackend>,

    // Optimiser
    optimiser: OptimizerAdaptor<Adam, QNetwork<B>, B>,

    // Experience replay
    buffer: Buf,

    // Encoding
    encoder: Enc,
    action_mapper: Act,

    // Config and runtime state
    config: DqnConfig,
    device: B::Device,
    total_steps: usize,

    rng: SmallRng,

    _env: std::marker::PhantomData<E>,
}

impl<E, Enc, Act, B> DqnAgent<E, Enc, Act, B>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B> + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
{
    /// Create a new agent using the default `CircularBuffer` replay buffer.
    ///
    /// Buffer capacity is taken from `config.buffer_capacity`.
    pub fn new(encoder: Enc, action_mapper: Act, config: DqnConfig, device: B::Device, seed: u64) -> Self {
        let buffer = CircularBuffer::new(config.buffer_capacity);
        Self::new_with_buffer(encoder, action_mapper, config, device, seed, buffer)
    }
}

impl<E, Enc, Act, B, Buf> DqnAgent<E, Enc, Act, B, Buf>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B> + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    /// Create a new agent with a custom replay buffer.
    ///
    /// Use this to swap in prioritised experience replay or any other
    /// `ReplayBuffer` implementation in place of the default `CircularBuffer`.
    pub fn new_with_buffer(
        encoder: Enc,
        action_mapper: Act,
        config: DqnConfig,
        device: B::Device,
        seed: u64,
        buffer: Buf,
    ) -> Self {
        let obs_size = <Enc as ObservationEncoder<E::Observation, B>>::obs_size(&encoder);
        let num_actions = action_mapper.num_actions();

        // Build layer sizes: obs_size -> hidden... -> num_actions
        let mut layer_sizes = vec![obs_size];
        layer_sizes.extend_from_slice(&config.hidden_sizes);
        layer_sizes.push(num_actions);

        let online_net = QNetwork::new(&layer_sizes, &device);
        let target_net = QNetwork::new(&layer_sizes, &device.clone());

        let optimiser = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<B, QNetwork<B>>();

        Self {
            online_net,
            target_net,
            optimiser,
            buffer,
            encoder,
            action_mapper,
            device: device.clone(),
            total_steps: 0,
            config,
            rng: SmallRng::seed_from_u64(seed),
            _env: std::marker::PhantomData,
        }
    }

    /// Store a transition in the replay buffer and potentially run a gradient update.
    ///
    /// Called by the runner after every environment step. Returns `true` if
    /// a gradient update was performed this step.
    pub fn observe(&mut self, experience: Experience<E::Observation, E::Action>) -> bool {
        self.buffer.push(experience);
        self.total_steps += 1;

        // Sync target network periodically
        if self.total_steps.is_multiple_of(self.config.target_update_freq) {
            self.sync_target();
        }

        // Don't train until warm-up is complete
        if !self.buffer.ready_for(self.config.batch_size) {
            return false;
        }
        if self.buffer.len() < self.config.min_replay_size {
            return false;
        }

        self.train_step();
        true
    }

    /// The current exploration probability.
    pub fn epsilon(&self) -> f64 {
        self.config.epsilon_at(self.total_steps)
    }

    /// Total environment steps observed so far.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Select an action using ε-greedy policy.
    pub fn act_epsilon_greedy(&self, obs: &E::Observation, rng: &mut impl Rng) -> E::Action {
        let epsilon = self.epsilon();
        if rng.gen::<f64>() < epsilon {
            // Random action — sample by index and convert
            let idx = rng.gen_range(0..self.action_mapper.num_actions());
            self.action_mapper.index_to_action(idx)
        } else {
            self.act(obs)
        }
    }

    /// Save the online network weights to a file.
    ///
    /// Uses Burn's `CompactRecorder` (MessagePack format). The recorder appends
    /// its own extension to the path, so `save("run/cartpole")` produces
    /// `run/cartpole.mpk`.
    ///
    /// Only the online network weights are saved — the target network,
    /// replay buffer, and optimizer state are not included. This is sufficient
    /// for inference. To resume training, call `load` followed by
    /// `set_total_steps` to restore the correct epsilon.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), RecorderError> {
        self.online_net
            .clone()
            .save_file(path.as_ref().to_path_buf(), &CompactRecorder::new())
            .map(|_| ())
    }

    /// Load network weights from a file into this agent.
    ///
    /// Loads into the online network and immediately syncs the target network.
    /// Takes `self` by value and returns the updated agent so you can chain
    /// with the constructor:
    ///
    /// ```rust,ignore
    /// let agent = DqnAgent::new(...).load("run/cartpole")?;
    /// ```
    pub fn load(mut self, path: impl AsRef<Path>) -> Result<Self, RecorderError> {
        self.online_net = self
            .online_net
            .load_file(path.as_ref().to_path_buf(), &CompactRecorder::new(), &self.device)?;
        self.target_net = self.online_net.valid();
        Ok(self)
    }

    /// Convert this trained agent into an inference-only `DqnPolicy`.
    ///
    /// Strips all training state (optimizer, buffer, RNG) and downcasts the
    /// network to `B::InnerBackend` (no autodiff). Use this when training is
    /// complete and you want a lightweight policy for evaluation or deployment.
    ///
    /// ```rust,ignore
    /// let policy = runner.into_agent().into_policy();
    /// let action = policy.act(&obs);
    /// ```
    pub fn into_policy(self) -> super::inference::DqnPolicy<E, Enc, Act, B::InnerBackend> {
        super::inference::DqnPolicy::from_network(
            self.online_net.valid(),
            self.encoder,
            self.action_mapper,
            self.device,
        )
    }

    /// Override the internal step counter.
    ///
    /// Useful when resuming training — restores epsilon to the correct value
    /// for the point in training where the checkpoint was saved.
    pub fn set_total_steps(&mut self, steps: usize) {
        self.total_steps = steps;
    }

    /// Sync target network weights from the online network.
    ///
    /// Hard update: copies all parameters exactly. The target network
    /// provides stable TD targets — if it updated every step it would
    /// chase itself and training would diverge.
    fn sync_target(&mut self) {
        // Burn's valid() converts an autodiff module to its inner (non-diff) counterpart.
        self.target_net = self.online_net.valid();
    }

    /// One gradient update step.
    fn train_step(&mut self) {
        let batch = self.buffer.sample(self.config.batch_size, &mut self.rng);

        let batch_size = batch.len();

        // Encode observations and next observations
        let obs_batch: Vec<_> = batch.iter().map(|e| &e.observation).cloned().collect();
        let next_obs_batch: Vec<_> = batch.iter().map(|e| &e.next_observation).cloned().collect();

        let obs_tensor = self.encoder.encode_batch(&obs_batch, &self.device);
        // next_obs is encoded on the inner (non-autodiff) backend — no gradients needed
        let next_obs_tensor = self.encoder
            .encode_batch(&next_obs_batch, &self.device.clone());

        // Action indices, rewards, bootstrap masks
        let action_indices: Vec<usize> = batch.iter()
            .map(|e| self.action_mapper.action_to_index(&e.action))
            .collect();

        let rewards: Vec<f32> = batch.iter()
            .map(|e| e.reward as f32)
            .collect();

        let masks: Vec<f32> = batch.iter()
            .map(|e| e.bootstrap_mask() as f32)
            .collect();

        let rewards_t: Tensor<B, 1> = Tensor::from_floats(rewards.as_slice(), &self.device);
        let masks_t: Tensor<B, 1> = Tensor::from_floats(masks.as_slice(), &self.device);

        // Target Q-values (no gradients — computed on the target network)
        let next_q_values = self.target_net.forward(next_obs_tensor);
        let max_next_q: Tensor<B::InnerBackend, 1> = next_q_values.max_dim(1).squeeze::<1>();
        let max_next_q_autodiff: Tensor<B, 1> = Tensor::from_inner(max_next_q);

        // TD target: r + γ * mask * max_a Q_target(s', a)
        let targets = rewards_t + masks_t * max_next_q_autodiff * self.config.gamma as f32;

        // Online Q-values for the actions actually taken
        let q_values = self.online_net.forward(obs_tensor);
        let action_indices_t = Tensor::<B, 1, Int>::from_ints(
            action_indices.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice(),
            &self.device,
        );
        // Gather Q-values at the taken actions: q_values[i, action_indices[i]]
        let q_taken = q_values
            .gather(1, action_indices_t.reshape([batch_size, 1]))
            .squeeze::<1>();

        // Huber loss (more robust to outlier rewards than MSE)
        let loss = HuberLossConfig::new(1.0).init().forward(q_taken, targets.detach(), Reduction::Mean);

        // Gradient update
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online_net);
        self.online_net = self.optimiser.step(
            self.config.learning_rate,
            self.online_net.clone(),
            grads,
        );
    }
}

impl<E, Enc, Act, B, Buf> Policy<E::Observation, E::Action> for DqnAgent<E, Enc, Act, B, Buf>
where
    E: Environment,
    Enc: ObservationEncoder<E::Observation, B>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    /// Greedy action selection (no exploration).
    ///
    /// Use this for evaluation. For training, use `act_epsilon_greedy`.
    fn act(&self, obs: &E::Observation) -> E::Action {
        let obs_tensor = self.encoder.encode(obs, &self.device).unsqueeze_dim(0);
        let q_values = self.online_net.forward(obs_tensor);
        let best_action_idx: usize = q_values
            .argmax(1)
            .into_data()
            .to_vec::<i64>()
            .unwrap()[0] as usize;
        self.action_mapper.index_to_action(best_action_idx)
    }
}

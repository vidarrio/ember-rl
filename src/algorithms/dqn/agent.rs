use std::collections::HashMap;
use std::path::Path;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::record::CompactRecorder;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rl_traits::{Environment, Experience, Policy};

use crate::encoding::{DiscreteActionMapper, ObservationEncoder};
use crate::stats::{Aggregator, Max, Mean, Std};
use crate::traits::{ActMode, Checkpointable, LearningAgent};
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
/// - `Buf`: the replay buffer (defaults to `CircularBuffer` -- swap for PER etc.)
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

    // Two decoupled RNGs -- sharing one causes correlated sampling/exploration.
    explore_rng: SmallRng,  // drives ε-greedy action selection
    sample_rng: SmallRng,   // drives replay buffer sampling

    // Per-episode loss aggregators (reset each episode via on_episode_start)
    ep_loss_mean: Mean,
    ep_loss_std: Std,
    ep_loss_max: Max,

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
            // Offset the two seeds so they produce independent streams.
            explore_rng: SmallRng::seed_from_u64(seed),
            sample_rng: SmallRng::seed_from_u64(seed.wrapping_add(0x9e37_79b9_7f4a_7c15)),
            ep_loss_mean: Mean::default(),
            ep_loss_std: Std::default(),
            ep_loss_max: Max::default(),
            _env: std::marker::PhantomData,
        }
    }

    /// The current exploration probability.
    pub fn epsilon(&self) -> f64 {
        self.config.epsilon_at(self.total_steps)
    }

    /// Override the internal step counter (use when resuming training).
    pub fn set_total_steps(&mut self, steps: usize) {
        self.total_steps = steps;
    }

    /// Convert this trained agent into an inference-only `DqnPolicy`.
    pub fn into_policy(self) -> super::inference::DqnPolicy<E, Enc, Act, B::InnerBackend> {
        super::inference::DqnPolicy::from_network(
            self.online_net.valid(),
            self.encoder,
            self.action_mapper,
            self.device,
        )
    }

    fn sync_target(&mut self) {
        self.target_net = self.online_net.valid();
    }

    /// One gradient update step. Returns the scalar loss value.
    fn train_step(&mut self) -> f64 {
        let batch = self.buffer.sample(self.config.batch_size, &mut self.sample_rng);
        let batch_size = batch.len();

        let obs_batch: Vec<_> = batch.iter().map(|e| &e.observation).cloned().collect();
        let next_obs_batch: Vec<_> = batch.iter().map(|e| &e.next_observation).cloned().collect();

        let obs_tensor = self.encoder.encode_batch(&obs_batch, &self.device);
        let next_obs_tensor = self.encoder.encode_batch(&next_obs_batch, &self.device.clone());

        let action_indices: Vec<usize> = batch.iter()
            .map(|e| self.action_mapper.action_to_index(&e.action))
            .collect();
        let rewards: Vec<f32> = batch.iter().map(|e| e.reward as f32).collect();
        let masks: Vec<f32> = batch.iter().map(|e| e.bootstrap_mask() as f32).collect();

        let rewards_t: Tensor<B, 1> = Tensor::from_floats(rewards.as_slice(), &self.device);
        let masks_t: Tensor<B, 1> = Tensor::from_floats(masks.as_slice(), &self.device);

        let next_q_values = self.target_net.forward(next_obs_tensor);
        let max_next_q: Tensor<B::InnerBackend, 1> = next_q_values.max_dim(1).squeeze::<1>();
        let max_next_q_autodiff: Tensor<B, 1> = Tensor::from_inner(max_next_q);

        let targets = rewards_t + masks_t * max_next_q_autodiff * self.config.gamma as f32;

        let q_values = self.online_net.forward(obs_tensor);
        let action_indices_t = Tensor::<B, 1, Int>::from_ints(
            action_indices.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice(),
            &self.device,
        );
        let q_taken = q_values
            .gather(1, action_indices_t.reshape([batch_size, 1]))
            .squeeze::<1>();

        let loss = HuberLossConfig::new(1.0)
            .init()
            .forward(q_taken, targets.detach(), Reduction::Mean);

        let loss_val = loss.clone().into_scalar().elem::<f64>();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.online_net);
        self.online_net = self.optimiser.step(
            self.config.learning_rate,
            self.online_net.clone(),
            grads,
        );

        loss_val
    }
}

// ── Checkpointable ────────────────────────────────────────────────────────────

impl<E, Enc, Act, B, Buf> Checkpointable for DqnAgent<E, Enc, Act, B, Buf>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B> + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    fn save(&self, path: &Path) -> anyhow::Result<()> {
        self.online_net
            .clone()
            .save_file(path.to_path_buf(), &CompactRecorder::new())
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!(e))
    }

    fn load(mut self, path: &Path) -> anyhow::Result<Self> {
        self.online_net = self.online_net
            .load_file(path.to_path_buf(), &CompactRecorder::new(), &self.device)
            .map_err(|e| anyhow::anyhow!(e))?;
        self.target_net = self.online_net.valid();
        Ok(self)
    }
}

// ── LearningAgent ─────────────────────────────────────────────────────────────

impl<E, Enc, Act, B, Buf> LearningAgent<E> for DqnAgent<E, Enc, Act, B, Buf>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B> + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    fn act(&mut self, obs: &E::Observation, mode: ActMode) -> E::Action {
        match mode {
            ActMode::Exploit => {
                // Greedy argmax via the inner (non-autodiff) network -- no gradient
                // bookkeeping overhead, same weights, same device type.
                let obs_tensor = ObservationEncoder::<E::Observation, B::InnerBackend>::encode(
                    &self.encoder, obs, &self.device,
                ).unsqueeze_dim(0);
                let q_values = self.online_net.valid().forward(obs_tensor);
                let idx: usize = q_values
                    .argmax(1)
                    .into_data()
                    .to_vec::<i64>()
                    .unwrap()[0] as usize;
                self.action_mapper.index_to_action(idx)
            }
            ActMode::Explore => {
                let epsilon = self.config.epsilon_at(self.total_steps);
                if self.explore_rng.gen::<f64>() < epsilon {
                    let idx = self.explore_rng.gen_range(0..self.action_mapper.num_actions());
                    self.action_mapper.index_to_action(idx)
                } else {
                    let obs_tensor = self.encoder.encode(obs, &self.device).unsqueeze_dim(0);
                    let q_values = self.online_net.forward(obs_tensor);
                    let idx: usize = q_values
                        .argmax(1)
                        .into_data()
                        .to_vec::<i64>()
                        .unwrap()[0] as usize;
                    self.action_mapper.index_to_action(idx)
                }
            }
        }
    }

    fn observe(&mut self, experience: Experience<E::Observation, E::Action>) {
        self.buffer.push(experience);
        self.total_steps += 1;

        if self.total_steps.is_multiple_of(self.config.target_update_freq) {
            self.sync_target();
        }

        if self.buffer.ready_for(self.config.batch_size)
            && self.buffer.len() >= self.config.min_replay_size
        {
            let loss = self.train_step();
            self.ep_loss_mean.update(loss);
            self.ep_loss_std.update(loss);
            self.ep_loss_max.update(loss);
        }
    }

    fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn episode_extras(&self) -> HashMap<String, f64> {
        [
            ("epsilon".to_string(),    self.epsilon()),
            ("loss_mean".to_string(),  self.ep_loss_mean.value()),
            ("loss_std".to_string(),   self.ep_loss_std.value()),
            ("loss_max".to_string(),   self.ep_loss_max.value()),
        ]
        .into_iter()
        .filter(|(_, v)| v.is_finite())
        .collect()
    }

    fn on_episode_start(&mut self) {
        self.ep_loss_mean.reset();
        self.ep_loss_std.reset();
        self.ep_loss_max.reset();
    }
}

// ── Policy (greedy inference) ─────────────────────────────────────────────────

impl<E, Enc, Act, B, Buf> Policy<E::Observation, E::Action> for DqnAgent<E, Enc, Act, B, Buf>
where
    E: Environment,
    Enc: ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
    Buf: ReplayBuffer<E::Observation, E::Action>,
{
    fn act(&self, obs: &E::Observation) -> E::Action {
        let obs_tensor = self.encoder.encode(obs, &self.device).unsqueeze_dim(0);
        let q_values = self.online_net.valid().forward(obs_tensor);
        let idx: usize = q_values
            .argmax(1)
            .into_data()
            .to_vec::<i64>()
            .unwrap()[0] as usize;
        self.action_mapper.index_to_action(idx)
    }
}

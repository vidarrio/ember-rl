use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::Path;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::record::CompactRecorder;
use rand::{SeedableRng};
use rand::rngs::SmallRng;
use rl_traits::{Environment, Experience};

use crate::encoding::{DiscreteActionMapper, ObservationEncoder};
use crate::stats::{Aggregator, Mean};
use crate::traits::{ActMode, Checkpointable, LearningAgent};
use super::config::PpoConfig;
use super::network::ActorCriticNetwork;
use super::rollout::{RolloutBuffer, Transition};

/// A PPO agent with discrete actions.
///
/// Implements clipped-surrogate PPO with GAE advantage estimation and an
/// actor-critic network. Generic over environment, encoder, action mapper,
/// and Burn backend -- the same pattern as `DqnAgent`.
///
/// **On-policy**: experience is collected into a rollout buffer, used for
/// `n_epochs` gradient passes, then discarded. The buffer size is
/// `n_steps * n_envs`; an update fires automatically when it fills.
///
/// # Parallel environments
///
/// Set `PpoConfig::n_envs` to match the number of environments feeding
/// this agent (e.g. `bevy-gym`'s `NUM_ENVS`). All envs contribute to the
/// same rollout buffer; the update fires after `n_steps` ticks.
pub struct PpoAgent<E, Enc, Act, B>
where
    E: Environment,
    B: AutodiffBackend,
{
    model: ActorCriticNetwork<B>,
    optimiser: OptimizerAdaptor<Adam, ActorCriticNetwork<B>, B>,

    rollout: RolloutBuffer<E::Observation>,

    encoder: Enc,
    action_mapper: Act,

    config: PpoConfig,
    device: B::Device,
    total_steps: usize,

    // FIFO cache populated by act() and consumed by observe().
    // Each act() call pushes (log_prob, value); observe() pops one entry.
    // This works because bevy-gym (and DqnTrainer) call act then observe
    // in matched pairs within the same tick / loop iteration.
    pending: VecDeque<(f32, f32)>,

    update_rng: SmallRng,

    // Per-update stats (reset after each PPO update)
    ep_policy_loss: Mean,
    ep_value_loss: Mean,
    ep_entropy: Mean,
    ep_approx_kl: Mean,

    _env: std::marker::PhantomData<E>,
}

impl<E, Enc, Act, B> PpoAgent<E, Enc, Act, B>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B>
        + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
{
    pub fn new(encoder: Enc, action_mapper: Act, config: PpoConfig, device: B::Device, seed: u64) -> Self {
        let obs_size = <Enc as ObservationEncoder<E::Observation, B>>::obs_size(&encoder);
        let n_actions = action_mapper.num_actions();

        let mut layer_sizes = vec![obs_size];
        layer_sizes.extend_from_slice(&config.hidden_sizes);

        let model = ActorCriticNetwork::new(&layer_sizes, n_actions, &device);
        let optimiser = AdamConfig::new().with_epsilon(1e-5).init::<B, ActorCriticNetwork<B>>();

        let rollout_size = config.rollout_size();

        Self {
            model,
            optimiser,
            rollout: RolloutBuffer::new(rollout_size),
            encoder,
            action_mapper,
            config,
            device,
            total_steps: 0,
            pending: VecDeque::new(),
            update_rng: SmallRng::seed_from_u64(seed),
            ep_policy_loss: Mean::default(),
            ep_value_loss: Mean::default(),
            ep_entropy: Mean::default(),
            ep_approx_kl: Mean::default(),
            _env: std::marker::PhantomData,
        }
    }

    // ---- Forward pass (no grad) ----

    fn forward_inference(
        &self,
        obs: &E::Observation,
    ) -> (usize, f32, f32) {
        // Use valid() to avoid tracking gradients during collection.
        let obs_t = ObservationEncoder::<E::Observation, B::InnerBackend>::encode(
            &self.encoder, obs, &self.device,
        ).unsqueeze_dim(0);

        let model_valid = self.model.valid();
        let (logits, value) = model_valid.forward(obs_t);

        // Softmax -> categorical sample
        let probs = burn::tensor::activation::softmax(logits.clone(), 1); // [1, n_actions]
        let probs_vec: Vec<f32> = probs
            .squeeze::<1>()
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        // Sample action proportional to probabilities
        let action_idx = sample_categorical(&probs_vec, &mut rand::thread_rng());

        let log_prob = probs_vec[action_idx].ln();
        let v: f32 = value.into_data().to_vec::<f32>().unwrap()[0];

        (action_idx, log_prob, v)
    }

    fn forward_inference_greedy(&self, obs: &E::Observation) -> usize {
        let obs_t = ObservationEncoder::<E::Observation, B::InnerBackend>::encode(
            &self.encoder, obs, &self.device,
        ).unsqueeze_dim(0);

        let (logits, _) = self.model.valid().forward(obs_t);
        logits
            .argmax(1)
            .into_data()
            .to_vec::<i64>()
            .unwrap()[0] as usize
    }

    // ---- Bootstrap value for GAE ----

    fn compute_bootstrap_value(&self, next_obs: &E::Observation) -> f32 {
        let obs_t = ObservationEncoder::<E::Observation, B::InnerBackend>::encode(
            &self.encoder, next_obs, &self.device,
        ).unsqueeze_dim(0);
        let (_, value) = self.model.valid().forward(obs_t);
        value.into_data().to_vec::<f32>().unwrap()[0]
    }

    // ---- PPO update ----

    fn run_update(&mut self, bootstrap_value: f32) {
        let gamma = self.config.gamma as f32;
        let lambda = self.config.gae_lambda as f32;
        self.rollout.compute_gae(bootstrap_value, gamma, lambda);
        self.rollout.normalize_advantages();

        for _ in 0..self.config.n_epochs {
            let batches = self.rollout.minibatches(self.config.batch_size, &mut self.update_rng);
            for batch in batches {
                let (pl, vl, ent, kl) = self.update_step(&batch);
                self.ep_policy_loss.update(pl);
                self.ep_value_loss.update(vl);
                self.ep_entropy.update(ent);
                self.ep_approx_kl.update(kl);
            }
        }

        self.rollout.clear();
    }

    fn update_step(
        &mut self,
        batch: &super::rollout::Batch<E::Observation>,
    ) -> (f64, f64, f64, f64) {
        let bs = batch.obs.len();
        let obs_t = self.encoder.encode_batch(&batch.obs, &self.device);

        let (logits, values) = self.model.forward(obs_t);

        // Probabilities and log-probabilities
        let probs = burn::tensor::activation::softmax(logits, 1);  // [bs, n_actions]
        let log_probs_all = probs.clone().log();                    // [bs, n_actions]

        // Gather log_prob for the taken actions
        let action_idx_t = Tensor::<B, 1, Int>::from_ints(
            batch.actions.iter().map(|&a| a as i32).collect::<Vec<_>>().as_slice(),
            &self.device,
        );
        let new_log_probs = log_probs_all.clone()
            .gather(1, action_idx_t.reshape([bs, 1]))
            .squeeze::<1>(); // [bs]

        // Old log probs (from collection time)
        let old_log_probs_t: Tensor<B, 1> = Tensor::from_floats(
            batch.old_log_probs.as_slice(), &self.device,
        );

        // Advantages and returns
        let advantages_t: Tensor<B, 1> = Tensor::from_floats(
            batch.advantages.as_slice(), &self.device,
        );
        let returns_t: Tensor<B, 1> = Tensor::from_floats(
            batch.returns.as_slice(), &self.device,
        );

        // PPO clipped surrogate loss
        let ratio = (new_log_probs.clone() - old_log_probs_t.clone().detach()).exp();
        let clip_eps = self.config.clip_epsilon as f32;
        let surr1 = ratio.clone() * advantages_t.clone().detach();
        let surr2 = ratio.clone().clamp(1.0 - clip_eps, 1.0 + clip_eps)
            * advantages_t.detach();
        let policy_loss = -surr1.min_pair(surr2).mean();

        // Value loss (MSE)
        let diff = values - returns_t.detach();
        let value_loss = (diff.clone() * diff).mean()
            * self.config.value_loss_coef as f32;

        // Entropy bonus
        let entropy = -(probs.clone() * log_probs_all.clone()).sum_dim(1).mean();
        let entropy_loss = entropy.clone() * (-self.config.entropy_coef as f32);

        let total_loss = policy_loss.clone() + value_loss.clone() + entropy_loss;

        // Scalar stats (detach before backward)
        let pl_val = policy_loss.clone().into_scalar().elem::<f64>();
        let vl_val = value_loss.clone().into_scalar().elem::<f64>();
        let ent_val = entropy.clone().into_scalar().elem::<f64>();

        // Approximate KL (for monitoring)
        let approx_kl = (old_log_probs_t.detach() - new_log_probs.detach())
            .mean()
            .into_scalar()
            .elem::<f64>();

        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimiser.step(self.config.learning_rate, self.model.clone(), grads);

        (pl_val, vl_val, ent_val, approx_kl)
    }
}

// ---- Checkpointable ----

impl<E, Enc, Act, B> Checkpointable for PpoAgent<E, Enc, Act, B>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B> + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
{
    fn save(&self, path: &Path) -> anyhow::Result<()> {
        self.model
            .clone()
            .save_file(path.to_path_buf(), &CompactRecorder::new())
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!(e))
    }

    fn load(mut self, path: &Path) -> anyhow::Result<Self> {
        self.model = self.model
            .load_file(path.to_path_buf(), &CompactRecorder::new(), &self.device)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(self)
    }
}

// ---- LearningAgent ----

impl<E, Enc, Act, B> LearningAgent<E> for PpoAgent<E, Enc, Act, B>
where
    E: Environment,
    E::Observation: Clone + Send + Sync + 'static,
    E::Action: Clone + Send + Sync + 'static,
    Enc: ObservationEncoder<E::Observation, B> + ObservationEncoder<E::Observation, B::InnerBackend>,
    Act: DiscreteActionMapper<E::Action>,
    B: AutodiffBackend,
{
    fn act(&mut self, obs: &E::Observation, mode: ActMode) -> E::Action {
        match mode {
            ActMode::Exploit => {
                let idx = self.forward_inference_greedy(obs);
                // Push dummy pending entry so observe() can safely pop
                self.pending.push_back((0.0, 0.0));
                self.action_mapper.index_to_action(idx)
            }
            ActMode::Explore => {
                let (idx, log_prob, value) = self.forward_inference(obs);
                self.pending.push_back((log_prob, value));
                self.action_mapper.index_to_action(idx)
            }
        }
    }

    fn observe(&mut self, experience: Experience<E::Observation, E::Action>) {
        self.total_steps += 1;

        let (log_prob, value) = self.pending.pop_front().unwrap_or((0.0, 0.0));
        let action = self.action_mapper.action_to_index(&experience.action);
        let done = !matches!(experience.status, rl_traits::EpisodeStatus::Continuing);

        self.rollout.push(Transition {
            obs: experience.observation,
            action,
            reward: experience.reward as f32,
            done,
            value,
            log_prob,
        });

        if self.rollout.is_full() {
            // Bootstrap: if the last step ended an episode, future value is 0;
            // otherwise estimate it from the next observation.
            let bootstrap = if self.rollout.last_done() {
                0.0
            } else {
                self.compute_bootstrap_value(&experience.next_observation)
            };
            self.run_update(bootstrap);
        }
    }

    fn total_steps(&self) -> usize {
        self.total_steps
    }

    fn episode_extras(&self) -> HashMap<String, f64> {
        [
            ("policy_loss".to_string(), self.ep_policy_loss.value()),
            ("value_loss".to_string(),  self.ep_value_loss.value()),
            ("entropy".to_string(),     self.ep_entropy.value()),
            ("approx_kl".to_string(),   self.ep_approx_kl.value()),
        ]
        .into_iter()
        .filter(|(_, v)| v.is_finite())
        .collect()
    }

    fn on_episode_start(&mut self) {
        self.ep_policy_loss.reset();
        self.ep_value_loss.reset();
        self.ep_entropy.reset();
        self.ep_approx_kl.reset();
    }
}

// ---- Categorical sampling ----

fn sample_categorical(probs: &[f32], rng: &mut impl rand::Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

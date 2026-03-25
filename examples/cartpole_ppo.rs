// CartPole-v1 with PPO.
//
// Demonstrates the PPO agent with the ember-rl training infrastructure.
// A single agent trains across 1 environment (set N_ENVS in PpoConfig to match).
//
//   cargo run --example cartpole_ppo --features envs --release

use burn::backend::{Autodiff, NdArray};
use ember_rl::{
    algorithms::ppo::{PpoAgent, PpoConfig},
    encoding::{UsizeActionMapper, VecEncoder},
    envs::cartpole::CartPoleEnv,
    training::{TrainingRun, TrainingSession},
    traits::ActMode,
};
use rl_traits::{Environment, EpisodeStatus};

type B = Autodiff<NdArray>;

const MAX_STEPS: usize = 500_000;
const LOG_INTERVAL: usize = 10_000;
const N_ENVS: usize = 1;

fn main() {
    let config = PpoConfig {
        n_steps: 128,
        n_envs: N_ENVS,
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
    };

    let agent = PpoAgent::<CartPoleEnv, _, _, B>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        config,
        Default::default(),
        42,
    );

    let run = TrainingRun::create("cartpole_ppo", "v1")
        .expect("failed to create training run");
    println!("Run: {}", run.dir().display());

    let mut session = TrainingSession::new(agent)
        .with_run(run)
        .with_max_steps(MAX_STEPS)
        .with_checkpoint_freq(50_000);

    let mut env = CartPoleEnv::new();
    let (mut obs, _) = env.reset(Some(0));

    let mut episode_reward = 0.0f64;
    let mut episode_steps = 0usize;
    let mut total_episodes = 0usize;
    let mut reward_window: std::collections::VecDeque<f64> = std::collections::VecDeque::with_capacity(100);
    let mut last_logged = 0usize;

    println!("Training PPO on CartPole-v1 ({MAX_STEPS} steps)...\n");

    while !session.is_done() {
        let action = session.act(&obs, ActMode::Explore);
        let result = env.step(action.clone());

        episode_reward += result.reward;
        episode_steps += 1;

        session.observe(rl_traits::Experience::new(
            obs,
            action,
            result.reward,
            result.observation.clone(),
            result.status.clone(),
        ));

        let done = !matches!(result.status, EpisodeStatus::Continuing);
        obs = if done {
            let (next_obs, _) = env.reset(None);
            session.on_episode(
                episode_reward,
                episode_steps,
                result.status,
                std::collections::HashMap::new(),
            );

            if reward_window.len() == 100 { reward_window.pop_front(); }
            reward_window.push_back(episode_reward);
            total_episodes += 1;
            episode_reward = 0.0;
            episode_steps = 0;
            next_obs
        } else {
            result.observation
        };

        let steps = session.total_steps();
        let bucket = steps / LOG_INTERVAL;
        if bucket > last_logged {
            last_logged = bucket;
            let mean: f64 = if reward_window.is_empty() { f64::NAN }
                else { reward_window.iter().sum::<f64>() / reward_window.len() as f64 };
            println!("step {:>7}  ep {:>5}  mean reward (last {}): {:>6.1}",
                steps, total_episodes, reward_window.len(), mean);
        }
    }

    println!("\nDone. Checkpoints in run directory.");
}

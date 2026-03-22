//! CartPole-v1 trained with DQN.
//!
//! Run with:  cargo run --example cartpole --features envs --release

use burn::backend::{Autodiff, NdArray};
use ember_rl::{
    algorithms::dqn::{DqnAgent, DqnConfig},
    encoding::{UsizeActionMapper, VecEncoder},
    envs::cartpole::CartPoleEnv,
    training::DqnRunner,
};

type B = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    let config = DqnConfig {
        gamma: 0.99,
        learning_rate: 1e-3,
        batch_size: 64,
        buffer_capacity: 10_000,
        min_replay_size: 1_000,
        target_update_freq: 200,
        hidden_sizes: vec![64, 64],
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay_steps: 10_000,
    };

    let agent = DqnAgent::<CartPoleEnv, _, _, B>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        config,
        device,
        2,
    );
    let mut runner = DqnRunner::new(CartPoleEnv::new(), agent, 3);

    for step in runner.steps().take(100_000) {
        if step.episode_done {
            println!(
                "ep {:>4}  steps {:>3}  reward {:>6.1}  ε {:.3}",
                step.episode, step.episode_step, step.episode_reward, step.epsilon,
            );
        }
    }
}

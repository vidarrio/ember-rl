//! CartPole-v1 — train or evaluate a DQN agent.
//!
//! # Train (default)
//!
//!   cargo run --example cartpole --features envs --release
//!
//! Trains for 100 000 steps, saving checkpoints to
//! `runs/cartpole/v1/<timestamp>/`.
//! Runs a 20-episode greedy eval at the end and prints the report.
//!
//! # Eval from a saved run
//!
//!   cargo run --example cartpole --features envs --release -- --eval runs/cartpole/v1
//!
//! Loads `best.mpk` from the latest run under that path and evaluates for
//! 20 episodes.

use std::env;

use burn::backend::{Autodiff, NdArray};
use ember_rl::{
    algorithms::dqn::{DqnAgent, DqnConfig, DqnPolicy},
    encoding::{UsizeActionMapper, VecEncoder},
    envs::cartpole::CartPoleEnv,
    training::{DqnTrainer, TrainingRun},
};
use rl_traits::{Environment, Policy};

type B = Autodiff<NdArray>;
type InferB = NdArray;

const TRAIN_STEPS: usize = 100_000;
const EVAL_EPISODES: usize = 20;
const CHECKPOINT_FREQ: usize = 10_000;

fn cartpole_config() -> DqnConfig {
    DqnConfig {
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
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if let Some(pos) = args.iter().position(|a| a == "--eval") {
        let run_path = args
            .get(pos + 1)
            .expect("--eval requires a path argument, e.g. --eval runs/cartpole/v1");
        eval_mode(run_path);
    } else {
        train_mode();
    }
}

fn train_mode() {
    println!("Training DQN on CartPole-v1 for {TRAIN_STEPS} steps...\n");

    let device = Default::default();
    let config = cartpole_config();

    let agent = DqnAgent::<CartPoleEnv, _, _, B>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        config.clone(),
        device,
        42,
    );

    let run = TrainingRun::create("cartpole", "v1")
        .expect("failed to create training run");
    run.write_config(&(&config, VecEncoder::new(4), UsizeActionMapper::new(2)))
        .expect("failed to write config");

    println!("Run directory: {}", run.dir().display());

    let mut trainer = DqnTrainer::new(CartPoleEnv::new(), agent)
        .with_run(run)
        .with_checkpoint_freq(CHECKPOINT_FREQ)
        .with_keep_checkpoints(3);

    for step in trainer.steps().take(TRAIN_STEPS) {
        if step.episode_done {
            println!(
                "ep {:>4}  steps {:>3}  reward {:>6.1}  ε {:.3}",
                step.episode, step.episode_step, step.episode_reward, step.epsilon,
            );
        }
    }

    println!("\nRunning {EVAL_EPISODES}-episode eval...");
    let report = trainer.eval(EVAL_EPISODES);
    report.print();
}

fn eval_mode(run_path: &str) {
    println!("Loading checkpoint from {run_path}...\n");

    let run = TrainingRun::resume(run_path)
        .unwrap_or_else(|e| panic!("failed to resume run at {run_path}: {e}"));

    println!("Run: {} {} (step {})", run.metadata.name, run.metadata.version, run.metadata.total_steps);

    let config = cartpole_config();
    let device: <InferB as burn::prelude::Backend>::Device = Default::default();

    let policy = DqnPolicy::<CartPoleEnv, _, _, InferB>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        &config,
        device,
    )
    .load(run.best_checkpoint_path().with_extension(""))
    .expect("failed to load best.mpk — make sure the run has been trained");

    println!("Running {EVAL_EPISODES} greedy episodes...\n");

    let mut env = CartPoleEnv::new();
    let mut total_reward = 0.0;

    for ep in 1..=EVAL_EPISODES {
        let (mut obs, _) = env.reset(Some(ep as u64));
        let mut episode_reward = 0.0;
        let mut steps = 0;

        loop {
            let action = policy.act(&obs);
            let result = env.step(action);
            episode_reward += result.reward;
            steps += 1;
            if result.is_done() { break; }
            obs = result.observation;
        }

        total_reward += episode_reward;
        println!("ep {:>2}  steps {:>3}  reward {:>6.1}", ep, steps, episode_reward);
    }

    println!("\nmean reward: {:.1}", total_reward / EVAL_EPISODES as f64);
}

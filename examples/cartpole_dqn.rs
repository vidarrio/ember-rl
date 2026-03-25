//! CartPole-v1 -- train or evaluate a DQN agent.
//!
//! This example walks through the full ember-rl workflow:
//!   1. Configure a DQN agent
//!   2. Attach a named training run for checkpointing and logging
//!   3. Train with the iterator-style loop
//!   4. Evaluate the best saved checkpoint
//!
//! # Train (saves checkpoints to runs/cartpole/v1/<timestamp>/)
//!
//!   cargo run --example cartpole --features envs --release
//!
//! # Eval from the latest saved run
//!
//!   cargo run --example cartpole --features envs --release -- --eval runs/cartpole/v1

use std::env;

use burn::backend::{Autodiff, NdArray};
use ember_rl::{
    algorithms::dqn::{DqnAgent, DqnConfig, DqnPolicy},
    encoding::{UsizeActionMapper, VecEncoder},
    envs::cartpole::CartPoleEnv,
    training::{DqnTrainer, TrainingRun},
};
use rl_traits::{Environment, Policy};

// Burn requires two backend types:
//   B      -- training backend, wraps NdArray with automatic differentiation
//   InferB -- inference backend, plain NdArray without autodiff overhead
//
// When you switch to GPU, swap NdArray for e.g. Wgpu and update both lines.
type B = Autodiff<NdArray>;
type InferB = NdArray;

const TRAIN_STEPS: usize = 100_000;
const EVAL_EPISODES: usize = 20;
const CHECKPOINT_FREQ: usize = 10_000;

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

    // DqnConfig holds all hyperparameters. DqnConfig::default() is a reasonable
    // starting point; here we set them explicitly for clarity.
    let config = DqnConfig {
        gamma: 0.99,
        learning_rate: 3e-4,
        batch_size: 64,
        buffer_capacity: 100_000,
        // Don't start learning until the buffer has this many transitions.
        min_replay_size: 1_000,
        // Copy the online network to the target network every N steps.
        target_update_freq: 500,
        hidden_sizes: vec![64, 64],
        // Epsilon decays linearly from 1.0 → 0.01 over the first 10k steps,
        // then stays flat. Controls the explore/exploit trade-off.
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay_steps: 10_000,
    };

    // VecEncoder converts a Vec<f32> observation into a Burn tensor.
    // The argument is the observation size -- CartPole has 4 state variables.
    //
    // UsizeActionMapper maps between action indices (0, 1) and the usize
    // action type. The argument is the number of discrete actions.
    //
    // The turbofish `<CartPoleEnv, _, _, B>` fixes the environment and training
    // backend; the encoder and mapper types are inferred from the arguments.
    let agent = DqnAgent::<CartPoleEnv, _, _, B>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        config.clone(),
        device,
        42, // RNG seed
    );

    // TrainingRun creates and manages the run directory:
    //   runs/cartpole/v1/<YYYYMMDD_HHMMSS>/
    //     metadata.json          -- name, version, step counts, timestamps
    //     config.json            -- hyperparameters saved below
    //     checkpoints/           -- periodic .mpk snapshots + best.mpk
    //     train_episodes.jsonl   -- one record per episode (reward, length, extras)
    //     eval_episodes.jsonl    -- eval episodes tagged with step count
    let run = TrainingRun::create("cartpole", "v1")
        .expect("failed to create training run");
    // Persist the config so you can reproduce this run later.
    run.write_config(&(&config, VecEncoder::new(4), UsizeActionMapper::new(2)))
        .expect("failed to write config");

    println!("Run directory: {}", run.dir().display());

    // DqnTrainer owns the environment and drives the training loop.
    // with_run() attaches the TrainingRun for checkpointing and JSONL logging.
    let mut trainer = DqnTrainer::new(CartPoleEnv::new(), agent)
        .with_run(run)
        .with_checkpoint_freq(CHECKPOINT_FREQ)
        .with_keep_checkpoints(3); // keep the 3 most recent numbered checkpoints

    // trainer.steps() returns an iterator that steps the environment and the
    // agent in lockstep. Each item is a StepMetrics snapshot for that step.
    for step in trainer.steps().take(TRAIN_STEPS) {
        if step.episode_done {
            println!(
                "ep {:>4}  steps {:>3}  reward {:>6.1}  ε {:.3}",
                step.episode, step.episode_step, step.episode_reward, step.epsilon,
            );
        }
    }

    // Run greedy evaluation (epsilon = 0) and print a summary.
    // Saves best.mpk if the mean reward beats the previous best.
    println!("\nRunning {EVAL_EPISODES}-episode eval...");
    let report = trainer.eval(EVAL_EPISODES);
    report.print();
}

fn eval_mode(run_path: &str) {
    println!("Loading checkpoint from {run_path}...\n");

    // resume() accepts a run directory at any level of specificity:
    //   runs/cartpole/v1/<exact timestamp>   -- load that specific run
    //   runs/cartpole/v1                     -- load the most recent run
    //   runs/cartpole                        -- load the most recent version and run
    let run = TrainingRun::resume(run_path)
        .unwrap_or_else(|e| panic!("failed to resume run at {run_path}: {e}"));

    println!("Run: {} {} (step {})", run.metadata.name, run.metadata.version, run.metadata.total_steps);

    let config = DqnConfig {
        gamma: 0.99,
        learning_rate: 3e-4,
        batch_size: 64,
        buffer_capacity: 100_000,
        min_replay_size: 1_000,
        target_update_freq: 500,
        hidden_sizes: vec![64, 64],
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay_steps: 10_000,
    };

    // For inference, use InferB (plain NdArray) -- no autodiff, lower overhead.
    // The Device type annotation is required because Rust can't infer it from
    // Default::default() alone.
    let device: <InferB as burn::prelude::Backend>::Device = Default::default();

    // DqnPolicy is the inference-only counterpart to DqnAgent. It holds only
    // the frozen network weights -- no replay buffer, no optimizer, no epsilon.
    let policy = DqnPolicy::<CartPoleEnv, _, _, InferB>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        &config,
        device,
    )
    // Loads best.mpk -- the checkpoint with the highest mean eval reward.
    .load(run.best_checkpoint_path().with_extension(""))
    .expect("failed to load best.mpk -- run training first");

    println!("Running {EVAL_EPISODES} greedy episodes...\n");

    let mut env = CartPoleEnv::new();
    let mut total_reward = 0.0;

    for ep in 1..=EVAL_EPISODES {
        let (mut obs, _) = env.reset(Some(ep as u64));
        let mut episode_reward = 0.0;
        let mut steps = 0;

        loop {
            // policy.act() always picks the greedy action (no exploration).
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

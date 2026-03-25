//! Reinforcement learning algorithms powered by Burn, built on rl-traits.
//!
//! `ember-rl` is the algorithm layer in the stack:
//!
//! ```text
//! rl-traits     →  core traits (Environment, Agent, Policy, ...)
//! ember-rl      →  algorithm implementations using Burn (this crate)
//! bevy-gym      →  Bevy ECS plugin for visualisation and parallelisation
//! ```
//!
//! # Quick start
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use ember_rl::{
//!     algorithms::dqn::{DqnAgent, DqnConfig},
//!     encoding::{VecEncoder, UsizeActionMapper},
//!     training::DqnRunner,
//! };
//!
//! type B = NdArray;
//!
//! let env = CartPoleEnv::new();
//! let config = DqnConfig::default();
//! let encoder = VecEncoder::new(4);
//! let action_mapper = UsizeActionMapper::new(2);
//! let device = Default::default();
//!
//! let agent = DqnAgent::<_, _, _, B>::new(encoder, action_mapper, config, device);
//! let mut runner = DqnRunner::new(env, agent, 42);
//!
//! for step in runner.steps().take(50_000) {
//!     if step.episode_done {
//!         println!("Episode {} | reward: {:.1}", step.episode, step.episode_reward);
//!     }
//! }
//! ```

pub mod algorithms;
pub mod encoding;
pub mod stats;
pub mod traits;
pub mod training;

#[cfg(feature = "envs")]
pub mod envs;

#[cfg(feature = "dashboard")]
pub mod dashboard;

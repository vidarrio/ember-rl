//! CartPole-v1 reference environment.
//!
//! A pole balanced on a cart. The agent pushes the cart left or right to keep
//! the pole upright. Episodes terminate when the pole falls past 12° or the
//! cart leaves ±2.4 units; they are truncated at 500 steps (CartPole-v1).
//!
//! Matches the Gymnasium `CartPole-v1` specification exactly.
//!
//! # Spaces
//!
//! - **Observation**: `Vec<f32>` -- `[x, ẋ, θ, θ̇]`
//! - **Action**: `usize` -- `0` = push left, `1` = push right
//! - **Reward**: `1.0` per step

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rl_traits::{Environment, EpisodeStatus, StepResult};

/// CartPole-v1 environment.
pub struct CartPoleEnv {
    state: [f64; 4], // [x, x_dot, theta, theta_dot]
    steps: usize,
    rng: SmallRng,
}

impl CartPoleEnv {
    pub fn new() -> Self {
        Self {
            state: [0.0; 4],
            steps: 0,
            rng: SmallRng::seed_from_u64(0),
        }
    }

    fn obs(&self) -> Vec<f32> {
        self.state.map(|x| x as f32).to_vec()
    }

    fn is_terminated(&self) -> bool {
        let [x, _, theta, _] = self.state;
        x.abs() > 2.4 || theta.abs() > 0.20943951 // 12 degrees
    }
}

impl Default for CartPoleEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CartPoleEnv {
    type Observation = Vec<f32>;
    type Action = usize;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Vec<f32>, ()) {
        if let Some(s) = seed {
            self.rng = SmallRng::seed_from_u64(s);
        }
        self.state = std::array::from_fn(|_| self.rng.gen_range(-0.05..0.05));
        self.steps = 0;
        (self.obs(), ())
    }

    fn step(&mut self, action: usize) -> StepResult<Vec<f32>, ()> {
        const GRAVITY: f64 = 9.8;
        const MASSCART: f64 = 1.0;
        const MASSPOLE: f64 = 0.1;
        const TOTAL_MASS: f64 = MASSCART + MASSPOLE;
        const LENGTH: f64 = 0.5;
        const POLEMASS_LENGTH: f64 = MASSPOLE * LENGTH;
        const FORCE_MAG: f64 = 10.0;
        const TAU: f64 = 0.02;

        let [x, x_dot, theta, theta_dot] = self.state;
        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };

        let costheta = theta.cos();
        let sintheta = theta.sin();
        let temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;
        let thetaacc = (GRAVITY * sintheta - costheta * temp)
            / (LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS));
        let xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

        self.state = [
            x + TAU * x_dot,
            x_dot + TAU * xacc,
            theta + TAU * theta_dot,
            theta_dot + TAU * thetaacc,
        ];
        self.steps += 1;

        let status = if self.is_terminated() {
            EpisodeStatus::Terminated
        } else if self.steps >= 500 {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };

        StepResult::new(self.obs(), 1.0, status, ())
    }

    fn sample_action(&self, rng: &mut impl Rng) -> usize {
        rng.gen_range(0..2)
    }
}

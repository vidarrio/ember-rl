use burn::prelude::*;

/// Converts environment observations into Burn tensors.
///
/// This is the primary bridge between rl-traits' generic world and Burn's
/// tensor world. Users implement this for their specific observation type —
/// for CartPole it's 4 floats stacked into a 1D tensor; for Atari it would
/// be image preprocessing.
///
/// # Why this is separate from `Environment`
///
/// `rl-traits` deliberately knows nothing about tensors or ML backends.
/// This trait lives in ember-rl as the adapter layer. A user can implement
/// the same `Environment` for both headless training (with this encoder)
/// and Bevy visualisation (with no encoder at all).
///
/// # Batching
///
/// `encode_batch` has a default implementation that calls `encode` in a loop,
/// which is correct but slow. Override it with a vectorised implementation
/// if your observation type allows it — which for simple flat observations
/// (like CartPole) it always does.
pub trait ObservationEncoder<O, B: Backend> {
    /// The number of features in the encoded observation vector.
    ///
    /// Used to determine the Q-network's input layer size automatically.
    fn obs_size(&self) -> usize;

    /// Encode a single observation into a 1D tensor of shape `[obs_size]`.
    fn encode(&self, obs: &O, device: &B::Device) -> Tensor<B, 1>;

    /// Encode a batch of observations into a 2D tensor of shape `[batch, obs_size]`.
    ///
    /// The default implementation calls `encode` in a loop and stacks results.
    /// Override with a vectorised implementation for performance.
    fn encode_batch(&self, obs: &[O], device: &B::Device) -> Tensor<B, 2> {
        let encoded: Vec<Tensor<B, 1>> = obs.iter()
            .map(|o| self.encode(o, device))
            .collect();
        Tensor::stack(encoded, 0)
    }
}

/// Maps between environment action types and integer indices.
///
/// DQN is a discrete-action algorithm. The Q-network outputs one Q-value
/// per action, indexed 0..N. This trait bridges that integer world and the
/// environment's `Action` type, which may be an enum or something richer.
///
/// # Example
///
/// ```rust
/// use ember_rl::encoding::DiscreteActionMapper;
/// // CartPole: action is just usize (push left = 0, push right = 1)
/// struct CartPoleActions;
/// impl DiscreteActionMapper<usize> for CartPoleActions {
///     fn num_actions(&self) -> usize { 2 }
///     fn action_to_index(&self, action: &usize) -> usize { *action }
///     fn index_to_action(&self, index: usize) -> usize { index }
/// }
/// ```
pub trait DiscreteActionMapper<A> {
    /// Total number of discrete actions available.
    ///
    /// Determines the Q-network's output layer size.
    fn num_actions(&self) -> usize;

    /// Convert an action to its integer index.
    ///
    /// Used when storing experience — we record the index, not the action.
    fn action_to_index(&self, action: &A) -> usize;

    /// Convert an integer index to an action.
    ///
    /// Used when the Q-network selects an action — it returns an argmax
    /// index that we convert back to the environment's action type.
    fn index_to_action(&self, index: usize) -> A;
}

/// A trivial encoder for environments whose observations are already `Vec<f32>`.
///
/// Useful for getting something running quickly without boilerplate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VecEncoder {
    size: usize,
}

impl VecEncoder {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl<B: Backend> ObservationEncoder<Vec<f32>, B> for VecEncoder {
    fn obs_size(&self) -> usize {
        self.size
    }

    fn encode(&self, obs: &Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
        Tensor::from_floats(obs.as_slice(), device)
    }

    fn encode_batch(&self, obs: &[Vec<f32>], device: &B::Device) -> Tensor<B, 2> {
        let flat: Vec<f32> = obs.iter().flat_map(|o| o.iter().copied()).collect();
        let batch = obs.len();
        Tensor::<B, 1>::from_floats(flat.as_slice(), device)
            .reshape([batch, self.size])
    }
}

/// A trivial action mapper for environments whose actions are plain `usize`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UsizeActionMapper {
    num_actions: usize,
}

impl UsizeActionMapper {
    pub fn new(num_actions: usize) -> Self {
        Self { num_actions }
    }
}

impl DiscreteActionMapper<usize> for UsizeActionMapper {
    fn num_actions(&self) -> usize {
        self.num_actions
    }

    fn action_to_index(&self, action: &usize) -> usize {
        *action
    }

    fn index_to_action(&self, index: usize) -> usize {
        index
    }
}

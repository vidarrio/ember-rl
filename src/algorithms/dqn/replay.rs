use rand::Rng;
use rl_traits::{Experience, ReplayBuffer};

/// A fixed-capacity circular replay buffer.
///
/// Stores experience tuples and overwrites the oldest when full.
/// Implements `rl_traits::ReplayBuffer` so it's usable by any algorithm
/// that depends on that trait, not just DQN.
///
/// # Implementation notes
///
/// Uses a `Vec` pre-allocated to `capacity` with a write cursor that wraps
/// around. This avoids any allocation after construction and gives O(1) push
/// and O(batch_size) sample.
pub struct CircularBuffer<O, A> {
    storage: Vec<Experience<O, A>>,
    capacity: usize,
    cursor: usize,
    len: usize,
}

impl<O: Clone + Send + Sync, A: Clone + Send + Sync> CircularBuffer<O, A> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "buffer capacity must be > 0");
        Self {
            storage: Vec::with_capacity(capacity),
            capacity,
            cursor: 0,
            len: 0,
        }
    }
}

impl<O, A> ReplayBuffer<O, A> for CircularBuffer<O, A>
where
    O: Clone + Send + Sync,
    A: Clone + Send + Sync,
{
    fn push(&mut self, experience: Experience<O, A>) {
        if self.storage.len() < self.capacity {
            self.storage.push(experience);
        } else {
            self.storage[self.cursor] = experience;
        }
        self.cursor = (self.cursor + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<Experience<O, A>> {
        assert!(
            batch_size <= self.len,
            "cannot sample {batch_size} from buffer of size {}",
            self.len
        );
        (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..self.len);
                self.storage[idx].clone()
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl_traits::EpisodeStatus;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn make_exp(reward: f64) -> Experience<f32, usize> {
        Experience::new(0.0f32, 0usize, reward, 0.0f32, EpisodeStatus::Continuing)
    }

    #[test]
    fn push_and_len() {
        let mut buf: CircularBuffer<f32, usize> = CircularBuffer::new(4);
        assert!(buf.is_empty());
        buf.push(make_exp(1.0));
        buf.push(make_exp(2.0));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn overwrites_when_full() {
        let mut buf: CircularBuffer<f32, usize> = CircularBuffer::new(3);
        buf.push(make_exp(1.0));
        buf.push(make_exp(2.0));
        buf.push(make_exp(3.0));
        buf.push(make_exp(4.0)); // overwrites slot 0
        assert_eq!(buf.len(), 3);
        assert!(buf.is_full());
    }

    #[test]
    fn sample_returns_correct_batch_size() {
        let mut buf: CircularBuffer<f32, usize> = CircularBuffer::new(10);
        for i in 0..10 {
            buf.push(make_exp(i as f64));
        }
        let mut rng = SmallRng::seed_from_u64(42);
        let batch = buf.sample(4, &mut rng);
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn ready_for_respects_warmup() {
        let mut buf: CircularBuffer<f32, usize> = CircularBuffer::new(100);
        assert!(!buf.ready_for(64));
        for i in 0..64 {
            buf.push(make_exp(i as f64));
        }
        assert!(buf.ready_for(64));
    }
}

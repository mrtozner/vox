//! Ring buffer for accumulating audio samples into fixed-size frames.

/// A ring buffer that accumulates audio samples and provides access
/// to recent audio history.
pub struct AudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    len: usize,
}

impl AudioBuffer {
    /// Create a buffer that holds `duration_secs` of audio at `sample_rate`.
    pub fn new(duration_secs: f32, sample_rate: u32) -> Self {
        let capacity = (duration_secs * sample_rate as f32) as usize;
        Self {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            len: 0,
        }
    }

    /// Push samples into the ring buffer.
    pub fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }
        self.len = (self.len + samples.len()).min(self.capacity);
    }

    /// Get the last N seconds of audio as a contiguous Vec.
    pub fn last_seconds(&self, seconds: f32, sample_rate: u32) -> Vec<f32> {
        let requested = (seconds * sample_rate as f32) as usize;
        let available = requested.min(self.len);
        if available == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(available);
        // Start position: write_pos - available, wrapped around
        let start = if self.write_pos >= available {
            self.write_pos - available
        } else {
            self.capacity - (available - self.write_pos)
        };

        for i in 0..available {
            result.push(self.buffer[(start + i) % self.capacity]);
        }
        result
    }

    /// Get all buffered audio and reset.
    pub fn drain(&mut self) -> Vec<f32> {
        let result = self.last_seconds(self.len as f32, 1); // 1 Hz trick: len samples
        self.clear();
        result
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_retrieve() {
        let mut buf = AudioBuffer::new(1.0, 16000);
        let samples: Vec<f32> = (0..4800).map(|i| i as f32 / 4800.0).collect();
        buf.push(&samples);

        let last = buf.last_seconds(0.1, 16000); // 1600 samples
        assert_eq!(last.len(), 1600);
        // Should be the last 1600 samples pushed
        assert!((last[0] - 3200.0 / 4800.0).abs() < 1e-6);
    }

    #[test]
    fn wraps_around() {
        let mut buf = AudioBuffer::new(0.1, 100); // 10 samples
        buf.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        buf.push(&[11.0, 12.0, 13.0]); // wraps around

        let all = buf.last_seconds(0.1, 100);
        assert_eq!(all.len(), 10);
        assert_eq!(
            all,
            vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
        );
    }

    #[test]
    fn drain_clears() {
        let mut buf = AudioBuffer::new(1.0, 100);
        buf.push(&[1.0, 2.0, 3.0]);
        let drained = buf.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(buf.last_seconds(1.0, 100).len(), 0);
    }

    #[test]
    fn empty_buffer_returns_empty() {
        let buf = AudioBuffer::new(1.0, 16000);
        let result = buf.last_seconds(0.5, 16000);
        assert!(result.is_empty());
    }

    #[test]
    fn request_more_than_available() {
        let mut buf = AudioBuffer::new(1.0, 16000);
        buf.push(&[1.0, 2.0, 3.0]);
        let result = buf.last_seconds(10.0, 16000);
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn multiple_wraps() {
        let mut buf = AudioBuffer::new(0.05, 100); // 5 samples
        for i in 0..20 {
            buf.push(&[i as f32]);
        }
        let result = buf.last_seconds(0.05, 100);
        assert_eq!(result, vec![15.0, 16.0, 17.0, 18.0, 19.0]);
    }

    #[test]
    fn drain_then_push() {
        let mut buf = AudioBuffer::new(1.0, 100);
        buf.push(&[1.0, 2.0, 3.0]);
        buf.drain();
        buf.push(&[4.0, 5.0]);
        let result = buf.last_seconds(1.0, 100);
        assert_eq!(result.len(), 2);
        assert_eq!(result, vec![4.0, 5.0]);
    }

    #[test]
    fn zero_duration_request() {
        let mut buf = AudioBuffer::new(1.0, 16000);
        buf.push(&[1.0, 2.0, 3.0]);
        let result = buf.last_seconds(0.0, 16000);
        assert!(result.is_empty());
    }

    #[test]
    fn exact_capacity_fill() {
        let mut buf = AudioBuffer::new(0.1, 100); // 10 samples
        buf.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let result = buf.last_seconds(0.1, 100);
        assert_eq!(result.len(), 10);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[9], 10.0);
    }
}

//! Re-framing of an audio stream into fixed-size frames.

/// Splits a stream of arbitrarily sized sample chunks into frames of exactly
/// `frame_size` samples.
///
/// Samples that do not fill a frame are kept and prepended to the next push,
/// so nothing is dropped at chunk boundaries. VAD backends such as Silero
/// need exact frames (512 samples), while capture devices and resamplers
/// deliver whatever chunk size they like.
#[derive(Debug, Clone)]
pub struct FrameAssembler {
    frame_size: usize,
    residual: Vec<f32>,
}

impl FrameAssembler {
    /// Create an assembler producing frames of `frame_size` samples (minimum 1).
    pub fn new(frame_size: usize) -> Self {
        let frame_size = frame_size.max(1);
        Self {
            frame_size,
            residual: Vec::with_capacity(frame_size * 2),
        }
    }

    /// Frame size in samples.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Append samples to the stream.
    pub fn push(&mut self, samples: &[f32]) {
        self.residual.extend_from_slice(samples);
    }

    /// Take the next complete frame, if one is available.
    pub fn pop(&mut self) -> Option<Vec<f32>> {
        if self.residual.len() < self.frame_size {
            return None;
        }
        Some(self.residual.drain(..self.frame_size).collect())
    }

    /// Number of buffered samples that do not yet form a complete frame
    /// (after all complete frames have been popped).
    pub fn pending(&self) -> usize {
        self.residual.len()
    }

    /// Drop all buffered samples.
    pub fn clear(&mut self) {
        self.residual.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reframes_480_into_512_without_loss() {
        let input: Vec<f32> = (0..480 * 16).map(|i| i as f32).collect();
        let mut asm = FrameAssembler::new(512);
        let mut out = Vec::new();

        for chunk in input.chunks(480) {
            asm.push(chunk);
            while let Some(frame) = asm.pop() {
                assert_eq!(frame.len(), 512);
                out.extend(frame);
            }
        }

        // 16 * 480 = 15 * 512: every sample comes out, in order.
        assert_eq!(asm.pending(), 0);
        assert_eq!(out, input);
    }

    #[test]
    fn keeps_residual_until_frame_is_full() {
        let mut asm = FrameAssembler::new(512);
        asm.push(&[0.0; 500]);
        assert!(asm.pop().is_none());
        assert_eq!(asm.pending(), 500);
        asm.push(&[0.0; 12]);
        assert!(asm.pop().is_some());
        assert_eq!(asm.pending(), 0);
    }

    #[test]
    fn large_push_yields_multiple_frames() {
        let mut asm = FrameAssembler::new(4);
        asm.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(asm.pop().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(asm.pop().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
        assert!(asm.pop().is_none());
        assert_eq!(asm.pending(), 1);
        asm.clear();
        assert_eq!(asm.pending(), 0);
    }

    #[test]
    fn zero_frame_size_is_clamped() {
        assert_eq!(FrameAssembler::new(0).frame_size(), 1);
    }
}

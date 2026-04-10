//! ARM NEON-optimized operations for Raspberry Pi

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// ARM NEON-optimized dot product (if available)
#[cfg(target_arch = "aarch64")]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(4), "length must be multiple of 4 for NEON");

    unsafe {
        let mut sum = vdupq_n_f32(0.0);

        for i in (0..a.len()).step_by(4) {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            sum = vfmaq_f32(sum, va, vb); // sum += va * vb
        }

        // Horizontal sum of the 4 lanes
        let sum_arr = [
            vgetq_lane_f32(sum, 0),
            vgetq_lane_f32(sum, 1),
            vgetq_lane_f32(sum, 2),
            vgetq_lane_f32(sum, 3),
        ];

        sum_arr.iter().sum()
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    // Fallback: standard dot product
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_neon() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = dot_product_neon(&a, &b);
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0; // = 70.0

        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_zeros() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let result = dot_product_neon(&a, &b);
        assert!((result - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_larger() {
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];

        let result = dot_product_neon(&a, &b);
        let expected = 32.0; // 16 * 1.0 * 2.0

        assert!((result - expected).abs() < 1e-5);
    }
}

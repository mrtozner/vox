//! Quantized neural network layers for memory-efficient inference on edge devices.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder};

/// INT8 quantized linear layer
///
/// Stores weights as int8 (1 byte) with scale/zero-point for dequantization.
/// Memory: 1/2 of BF16 (int8 vs 16-bit)
pub struct QuantizedLinear {
    /// Quantized weights (int8)
    weights: Tensor,
    /// Scale factors for dequantization
    scales: Tensor,
    /// Zero points for dequantization
    zero_points: Tensor,
    /// Optional bias (kept in original dtype)
    bias: Option<Tensor>,
}

impl QuantizedLinear {
    /// Quantize an existing linear layer to INT8
    pub fn from_linear(linear: &Linear, device: &Device) -> Result<Self> {
        // Get original weights (assume BF16 or F32)
        let weights = linear.weight();

        // Convert to F32 for quantization
        let weights_f32 = weights.to_dtype(DType::F32)?;

        // Per-tensor quantization (simpler, faster)
        // scale = (max - min) / 255
        // zero_point = round(-min / scale)
        let flat = weights_f32.flatten_all()?;
        let min = flat.min(0)?.to_scalar::<f32>()?;
        let max = flat.max(0)?.to_scalar::<f32>()?;
        let scale = (max - min) / 255.0;
        let zero_point = (-min / scale).round();

        // Quantize: q = round(x / scale + zero_point)
        let quantized = weights_f32
            .affine((1.0 / scale) as f64, zero_point as f64)?
            .round()?
            .clamp(0.0, 255.0)?
            .to_dtype(DType::U8)?;

        Ok(Self {
            weights: quantized,
            scales: Tensor::from_slice(&[scale], 1, device)?,
            zero_points: Tensor::from_slice(&[zero_point], 1, device)?,
            bias: linear.bias().cloned(),
        })
    }

    /// Forward pass with runtime dequantization
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Dequantize: w = (q - zero_point) * scale
        let weights_f32 = self
            .weights
            .to_dtype(DType::F32)?
            .broadcast_sub(&self.zero_points)?
            .broadcast_mul(&self.scales)?;

        // Standard linear: y = xW^T + b
        let out = x.matmul(&weights_f32.t()?)?;

        if let Some(bias) = &self.bias {
            Ok(out.broadcast_add(bias)?)
        } else {
            Ok(out)
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let weights_bytes = self.weights.elem_count(); // 1 byte per element (int8)
        let scales_bytes = self.scales.elem_count() * 4; // f32
        let zero_points_bytes = self.zero_points.elem_count() * 4; // f32
        let bias_bytes = self.bias.as_ref().map(|b| b.elem_count() * 4).unwrap_or(0);

        weights_bytes + scales_bytes + zero_points_bytes + bias_bytes
    }
}

/// Helper to quantize a single linear layer by name
pub fn quantize_linear_layer(
    vb: &VarBuilder,
    layer_name: &str,
    in_dim: usize,
    out_dim: usize,
    device: &Device,
) -> Result<QuantizedLinear> {
    // Load the linear layer
    let linear = candle_nn::linear(in_dim, out_dim, vb.pp(layer_name))?;

    // Quantize it
    QuantizedLinear::from_linear(&linear, device)
}

/// Helper to quantize all linear layers in a model
pub fn quantize_model_layers(linears: &[Linear], device: &Device) -> Result<Vec<QuantizedLinear>> {
    linears
        .iter()
        .map(|linear| QuantizedLinear::from_linear(linear, device))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_quantization_memory_savings() {
        // Original BF16 layer: 1024x1024 = 2MB
        // Quantized INT8: 1024x1024 = 1MB (50% savings)
        let device = Device::Cpu;

        // Create test linear layer
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let linear = candle_nn::linear(1024, 1024, vb.pp("test")).unwrap();

        // Quantize
        let q_linear = QuantizedLinear::from_linear(&linear, &device).unwrap();

        // Check memory savings
        let original_bytes = 1024 * 1024 * 2; // BF16 = 2 bytes
        let quantized_bytes = q_linear.memory_bytes();

        let savings_percent = (1.0 - (quantized_bytes as f64 / original_bytes as f64)) * 100.0;
        assert!(savings_percent > 45.0); // At least 45% savings
        assert!(savings_percent < 55.0); // At most 55% savings
    }

    #[test]
    fn test_quantized_linear_forward() {
        let device = Device::Cpu;

        // Create test linear layer (4x4)
        let vb = VarBuilder::zeros(DType::F32, &device);
        let linear = candle_nn::linear(4, 4, vb.pp("test")).unwrap();

        // Quantize
        let q_linear = QuantizedLinear::from_linear(&linear, &device).unwrap();

        // Test forward pass
        let input = Tensor::ones(&[2, 4], DType::F32, &device).unwrap();
        let output = q_linear.forward(&input).unwrap();

        // Output shape should be [2, 4]
        assert_eq!(output.dims(), &[2, 4]);
    }
}

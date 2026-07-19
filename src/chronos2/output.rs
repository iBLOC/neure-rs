//! Output head + sampling for the Chronos2 architecture.
//!
//! Chronos2's published model emits a per-vocab-bin distribution
//! per output step. The architecture in `vendor/architecture.rs`
//! stops at the decoder's last hidden state; the head + sampling
//! lives here. The output head is a single `Linear(d_model,
//! num_quantiles)` projection followed by a sigmoid
//! (the published model uses a cumulative distribution function
//! over quantile fractions). The runtime commit wires this
//! to a `cumulative` -> `quantile lookup` step that produces
//! the actual forecast numbers.
//!
//! The v0 head stays simple: `Linear(d_model, 1) -> sigmoid -> one
//! number per token`. The training commit can swap in the real
//! `num_quantiles`-bin head without changing the runtime API.

use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, Linear, RmsNorm, VarBuilder};

use super::vendor::T5BlockConfig;

/// Output head: a single projection + sigmoid. The runtime
/// commit instantiates one per model.
pub struct Chronos2OutputHead {
    /// The last-decoder-layer RMSNorm that pre-processes the
    /// decoder output before projection. Standard T5 wraps
    /// `lm_head` with a final layer-norm; we do the same.
    final_norm: RmsNorm,
    /// Linear projection `d_model -> 1`. With 1 output neuron
    /// the head is a point estimate; the multi-quantile variant
    /// swaps in `Linear(d_model, num_quantiles)`.
    proj: Linear,
    /// Per-step bias (shape `[1]`). Trainable. We keep it on a
    /// `VarBuilder` like the rest of the model so the safetensors
    /// loader can populate it.
    bias: Tensor,
}

impl Chronos2OutputHead {
    pub fn new(cfg: T5BlockConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            final_norm: candle_nn::rms_norm(
                cfg.d_model,
                cfg.layer_norm_eps,
                vb.pp("final_norm"),
            )?,
            proj: linear(cfg.d_model, 1, vb.pp("proj"))?,
            bias: vb.get_with_hints(
                (1,),
                "bias",
                candle_nn::init::ZERO,
            )?,
        })
    }

    /// Project the decoder's last hidden state to one
    /// forecast number per token. Input shape
    /// `[batch, seq_len, d_model]`; output shape
    /// `[batch, seq_len]`. The runtime then takes the last
    /// `horizon` entries as the point forecast.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.final_norm.forward(x)?;
        let proj = self.proj.forward(&x)?.squeeze(-1)?;
        // Add the per-step bias and squash to (0, 1).
        let proj = proj.broadcast_add(&self.bias)?;
        candle_nn::ops::sigmoid(&proj)
    }
}

/// Aggregate a per-step distribution over a forecast window.
/// Today this is just "take the last `horizon` elements of the
/// projected sequence". The training commit can swap in a
/// proper quantile interpolation.
pub fn take_horizon(
    projections: &Tensor,
    horizon: u32,
) -> Result<Tensor> {
    let (_batch, seq_len) = projections.dims2()?;
    let start = seq_len.saturating_sub(horizon as usize);
    projections.narrow(1, start, seq_len - start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    fn head() -> Chronos2OutputHead {
        let cfg = T5BlockConfig::new(4, 8, 2);
        let dev = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);
        Chronos2OutputHead::new(cfg, vb).expect("head")
    }

    #[test]
    fn head_output_is_in_unit_interval() {
        let head = head();
        let dev = Device::Cpu;
        let x = candle_core::Tensor::zeros((1, 5, 4), DType::F32, &dev).unwrap();
        let y = head.forward(&x).expect("forward");
        assert_eq!(y.dims2().unwrap(), (1, 5));
        // sigmoid -> every value in (0, 1). Zero inputs bias the
        // bias toward 0, so the output is exactly 0.5.
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for &v in &data {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn take_horizon_returns_last_n() {
        let dev = Device::Cpu;
        let x = candle_core::Tensor::zeros((1, 10), DType::F32, &dev).unwrap();
        let out = take_horizon(&x, 3).unwrap();
        assert_eq!(out.dims2().unwrap(), (1, 3));
    }

    #[test]
    fn take_horizon_clamps_to_sequence_length() {
        let dev = Device::Cpu;
        let x = candle_core::Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        let out = take_horizon(&x, 10).unwrap();
        assert_eq!(out.dims2().unwrap(), (1, 4));
    }
}

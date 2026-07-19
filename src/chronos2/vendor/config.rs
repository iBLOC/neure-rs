//! Hyperparameters shared by the encoder and decoder transformer
//! blocks. Both the encoder and decoder of a T5-style model
//! take the same `d_model` / `d_ff` / `num_heads`; the depth
//! differs (typically 12 + 12, or 6 + 6 for the Chronos-Bolt
//! "small" variant) but is owned by the architecture that
//! composes the blocks, not by the block itself.

#[derive(Debug, Clone)]
pub struct T5BlockConfig {
    /// Token-embedding dimension. Standard T5 uses 512; Chronos2
    /// uses 768 in the encoder and 1024 in the decoder.
    pub d_model: usize,
    /// Feed-forward intermediate size. T5 convention: `d_ff = 4 *
    /// d_model`. Stored explicitly because Chronos2 sometimes
    /// uses a different ratio.
    pub d_ff: usize,
    /// Number of attention heads. Heads must divide `d_model`
    /// evenly; the build-time check in `T5Block::new` catches
    /// misconfiguration.
    pub num_heads: usize,
    /// Dropout probability in attention + FFN. 0.0 in
    /// inference-only builds.
    pub dropout: f32,
    /// Epsilon for RMSNorm. T5 uses 1e-6.
    pub layer_norm_eps: f64,
}

impl T5BlockConfig {
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        Self {
            d_model,
            d_ff,
            num_heads,
            dropout: 0.0,
            layer_norm_eps: 1e-6,
        }
    }

    /// Validate the head count divides `d_model` evenly. Returns
    /// `Err` with a human-readable reason if not.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 || self.num_heads == 0 || self.d_ff == 0 {
            return Err("d_model / num_heads / d_ff must all be > 0".to_string());
        }
        if self.d_model % self.num_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                self.d_model, self.num_heads
            ));
        }
        Ok(())
    }

    /// Standard T5 ratio `d_ff = 4 * d_model` for the small
    /// variants. Callers can override by setting `d_ff` after
    /// construction.
    pub fn t5_default(d_model: usize, num_heads: usize) -> Self {
        let mut s = Self::new(d_model, 4 * d_model, num_heads);
        s.dropout = 0.1;
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_divisible() {
        let cfg = T5BlockConfig::new(512, 2048, 8);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_divisible() {
        let cfg = T5BlockConfig::new(513, 2048, 8);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero() {
        let cfg = T5BlockConfig {
            d_model: 0,
            d_ff: 0,
            num_heads: 0,
            dropout: 0.0,
            layer_norm_eps: 1e-6,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn t5_default_uses_4x_ff_ratio() {
        let cfg = T5BlockConfig::t5_default(512, 8);
        assert_eq!(cfg.d_ff, 2048);
        assert_eq!(cfg.dropout, 0.1);
    }
}

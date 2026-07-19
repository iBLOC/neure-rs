//! Vendor tree for the Chronos2 candle port.
//!
//! The goal of this vendor tree is to host the T5-style transformer
//! blocks that Chronos2's encoder/decoder are built from. The blocks
//! are deliberately generic — a Chronos2-specific forward pass
//! composes them in a particular way; the blocks themselves do not
//! know about time series, tokenization, or sampling. The two
//! downstream commits (architecture + runtime) wire the actual
//! data flow.
//!
//! ## Layout
//!
//! - `mod.rs` — re-exports
//! - `config.rs` — hyperparameter struct (layers, d_model, heads, …)
//! - `embeddings.rs` — token + sinusoidal positional embeddings
//! - `transformer.rs` — generic T5 block (multi-head attention + FFN
//!   + RMSNorm + residual). The "block" is a single transformer
//!   layer; Chronos2 stacks N of them in the encoder and M in the
//!   decoder.
//!
//! ## What this vendor tree is NOT
//!
//! - It is not a drop-in Amazon Chronos2 implementation. The
//!   architecture-completion commits wire the actual model.
//! - It does not load safetensors. That's a runtime concern; see
//!   `src/chronos2/candle_runtime.rs` (next commit).
//! - It does not implement masking. The mask shape is T5-specific
//!   (causal vs padding); the architecture-completion commit wires
//!   it.

pub mod architecture;
pub mod config;
pub mod embeddings;
pub mod transformer;

pub use architecture::{T5CrossBlock, T5Decoder, T5Encoder, T5EncoderDecoder};
pub use config::T5BlockConfig;
pub use embeddings::{T5Embeddings, T5EmbeddingsConfig};
pub use transformer::{T5Block, T5BlockOutput};

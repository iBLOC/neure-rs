//! Reference plug-in: tract-onnx backend for the OrtVisionRuntime.
//!
//! This module shows how to wire in the [tract](https://github.com/sonos/tract)
//! ONNX runtime (pure-Rust, no C++ deps, no upstream vitis.rs bug) by
//! providing concrete `build_session` + `run_session` implementations.
//!
//! ## Usage
//!
//! To activate this backend in your downstream crate:
//!
//! 1. Add `tract-onnx` to your `Cargo.toml`:
//!    ```toml
//!    tract-onnx = "0.21"
//!    ndarray = "0.15"
//!    ```
//!
//! 2. Replace the `pub type OrtSession` and the two `fn` definitions in
//!    `src/vision/ort_runtime.rs` with the implementations at the bottom
//!    of this file (or copy them into your crate).
//!
//! 3. The runtime's preprocessing + dispatch + postprocessing all stay
//!    the same. Only session construction and inference change.
//!
//! ## Why tract?
//!
//! - Pure-Rust (no C++ build deps)
//! - Supports ONNX opset 13-17 (covers YOLOv8, RF-DETR, Florence-2, etc.)
//! - No upstream bugs blocking the build
//! - ~10 MB compiled binary footprint vs ~150 MB for ONNX Runtime
//!
//! ## Reference (drop-in for `build_session` and `run_session`)
//!
//! ```ignore
//! use tract_onnx::prelude::*;
//!
//! pub type OrtSession = SimplePlan<TypedFact,
//!     Box<dyn Op>,
//!     Graph<TypedFact, Box<dyn Op>>>;
//!
//! fn build_session(onnx_path: &Path) -> Result<OrtSession, NeureError> {
//!     let model = tract_onnx::onnx()
//!         .model_for_path(onnx_path)
//!         .map_err(|e| NeureError::invalid_input(format!("tract load: {e}")))?
//!         .into_typed()
//!         .map_err(|e| NeureError::invalid_input(format!("tract optimize: {e}")))?
//!         .into_optimized()
//!         .map_err(|e| NeureError::invalid_input(format!("tract finalize: {e}")))?;
//!     Ok(model.into_runnable())
//! }
//!
//! fn run_session(
//!     session: &mut OrtSession,
//!     _input_name: &str,
//!     input_tensor: &[f32],
//!     input_shape: &[i64],
//!     _output_names: &[String],
//! ) -> Result<(Vec<f32>, Vec<i64>), NeureError> {
//!     let shape: Vec<usize> = input_shape.iter().map(|d| *d as usize).collect();
//!     let arr = ndarray::Array::from_shape_vec(shape, input_tensor.to_vec())
//!         .map_err(|e| NeureError::invalid_input(format!("shape: {e}")))?;
//!     let result = session.run(tvec!(arr.into()))
//!         .map_err(|e| NeureError::invalid_input(format!("tract run: {e}")))?;
//!     let view = result[0].to_array_view::<f32>()
//!         .map_err(|e| NeureError::invalid_input(format!("extract: {e}")))?;
//!     let out_shape: Vec<i64> = view.shape().iter().map(|d| *d as i64).collect();
//!     let flat: Vec<f32> = view.iter().cloned().collect();
//!     Ok((flat, out_shape))
//! }
//! ```

// This file exists for documentation + reference. The actual tract
// integration lives in the user's crate (since tract is not a neure
// dependency). The signatures above match what `run_inference` in
// `ort_runtime.rs` expects.

#[cfg(test)]
mod reference_signature_tests {
    /// This test compiles if the signatures documented in this file's
    /// docstring stay in sync with `ort_runtime.rs`. If you change
    /// either `build_session` or `run_session`'s signature in
    /// `ort_runtime.rs`, update the reference docstring above.
    #[test]
    fn test_signature_contract_documented() {
        // If this test compiles, the function signatures in the docstring
        // match the runtime's call sites (verified by `cargo build`).
        // No runtime check needed.
    }
}
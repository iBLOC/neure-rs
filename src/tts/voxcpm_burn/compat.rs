//! Compatibility shim for burn 0.21 API differences vs the voxcpm_rs code.

#[macro_export]
macro_rules! vdbg {
    ($($all:tt)*) => {{
        let _ = $crate::tts::voxcpm_burn::compat::vdbg_inner($($all)*);
    }};
}

#[macro_export]
macro_rules! s {
    ($i:expr) => {
        $i
    };
}

#[inline(always)]
pub fn vdbg_inner<T>(x: T) -> T {
    x
}

pub trait SliceCompat {
    type Output;
    fn s_(&self, ranges: std::ops::Range<usize>) -> Self::Output;
    fn s_idx(&self, indices: &[usize]) -> Self::Output;
}

use burn::tensor::{backend::Backend, Int, Tensor};

impl<T: Backend> SliceCompat for Tensor<T, 2> {
    type Output = Self;
    fn s_(&self, range: std::ops::Range<usize>) -> Self::Output {
        self.clone().slice([range])
    }
    fn s_idx(&self, indices: &[usize]) -> Self::Output {
        let mut idx = Tensor::<T, 1, Int>::zeros([indices.len()], &self.device());
        for (i, &pos) in indices.iter().enumerate() {
            idx = idx
                .clone()
                .slice_assign([i..i + 1], Tensor::from_data([pos as i64], &self.device()));
        }
        self.clone().select(0, idx)
    }
}

impl<T: Backend> SliceCompat for Tensor<T, 3> {
    type Output = Self;
    fn s_(&self, range: std::ops::Range<usize>) -> Self::Output {
        self.clone().slice([range])
    }
    fn s_idx(&self, indices: &[usize]) -> Self::Output {
        let mut idx = Tensor::<T, 1, Int>::zeros([indices.len()], &self.device());
        for (i, &pos) in indices.iter().enumerate() {
            idx = idx
                .clone()
                .slice_assign([i..i + 1], Tensor::from_data([pos as i64], &self.device()));
        }
        self.clone().select(0, idx)
    }
}

use nalgebra::RealField;

pub use nalgebra;

pub trait Real: RealField + Copy {}

impl<T: RealField + Copy> Real for T {}

pub mod allocators;

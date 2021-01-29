pub mod allocators;
pub mod assembly;
pub mod connectivity;
pub mod element;
pub mod error;
pub mod mesh;
pub mod model;
pub mod procedural;
pub mod quadrature;
pub mod reorder;
pub mod space;
pub mod util;
pub mod vtk;

pub mod geometry {
    pub use fenris_geometry::*;
}

pub mod optimize {
    pub use fenris_optimize::*;
}

#[cfg(feature = "proptest")]
pub mod proptest;

mod mesh_convert;
mod space_impl;

pub extern crate nalgebra;
pub extern crate nested_vec;
pub extern crate vtkio;

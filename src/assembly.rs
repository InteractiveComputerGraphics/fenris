pub mod global;
pub mod local;

use crate::quadrature::Quadrature;
use local::ElementMatrixTransformation;

use nalgebra::allocator::Allocator;
use nalgebra::DMatrixSliceMut;
use nalgebra::{DefaultAllocator, DimName, Scalar, U1};

/// Lookup table mapping elements to quadrature rules.
pub trait QuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
{
    type QuadratureRule: Quadrature<T, GeometryDim>;

    fn quadrature_for_element(&self, element_index: usize) -> Self::QuadratureRule;
}

impl<'a, T, GeometryDim, F, Q> QuadratureTable<T, GeometryDim> for F
where
    F: 'a + Fn(usize) -> Q,
    Q: 'a + Quadrature<T, GeometryDim>,
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
{
    type QuadratureRule = Q;

    fn quadrature_for_element(&self, element_index: usize) -> Self::QuadratureRule {
        self(element_index)
    }
}

/// Convenience wrapper to turn a single quadrature into a quadrature table.
///
/// More precisely, this implies that the same quadrature rule will be used for every
/// element.
///
/// Note that the given quadrature will be cloned, so it's often more useful to wrap
/// a reference to a quadrature than letting the quadrature be cloned for each element.
#[derive(Copy, Clone, Debug)]
pub struct UniformQuadratureTable<Q>(pub Q);

impl<T, GeometryDim, Q> QuadratureTable<T, GeometryDim> for UniformQuadratureTable<Q>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
    Q: Clone + Quadrature<T, GeometryDim>,
{
    type QuadratureRule = Q;

    fn quadrature_for_element(&self, _element_index: usize) -> Self::QuadratureRule {
        self.0.clone()
    }
}

/// Leaves the given element matrix unaltered.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NoTransformation;

impl<T: Scalar> ElementMatrixTransformation<T> for NoTransformation {
    fn transform_element_matrix(&self, _element_matrix: &mut DMatrixSliceMut<T>) {
        // Do nothing
    }
}

// TODO: Write tests for distribute_local_to_global

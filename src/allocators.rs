//! Helper traits for collecting element allocator trait bounds.

use crate::element::{
    ConnectivityGeometryDim, ConnectivityReferenceDim, ElementConnectivity,
};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Scalar, U1};

/// Helper trait to make specifying bounds on generic functions working with the
/// `ReferenceFiniteElement` trait easier.
pub trait ReferenceFiniteElementAllocator<T, ReferenceDim>:
Allocator<T, ReferenceDim, ReferenceDim>
+ Allocator<T, ReferenceDim, U1>
+ Allocator<T, U1, ReferenceDim>
// For representing the indices of the nodes
+ Allocator<(usize, usize), ReferenceDim>
where
    T: Scalar,
    ReferenceDim: DimName,
{

}

/// Helper trait to make specifying bounds on generic functions working with the
/// `FiniteElement` trait easier.
pub trait FiniteElementAllocator<T, GeometryDim, ReferenceDim>:
ReferenceFiniteElementAllocator<T, ReferenceDim>
+ Allocator<T, GeometryDim>
+ Allocator<T, GeometryDim, GeometryDim>
+ Allocator<T, GeometryDim, ReferenceDim>
+ Allocator<T, GeometryDim, U1>
+ Allocator<T, U1, GeometryDim>
+ Allocator<T, ReferenceDim, ReferenceDim>
+ Allocator<T, ReferenceDim, U1>
+ Allocator<T, U1, ReferenceDim>
+ Allocator<T, ReferenceDim, GeometryDim>
// For representing the indices of the nodes
+ Allocator<(usize, usize), GeometryDim>
+ Allocator<(usize, usize), ReferenceDim>
    where
        T: Scalar,
        GeometryDim: DimName,
        ReferenceDim: DimName,
{

}

/// Helper trait to make specifying bounds on generic functions working with the
/// `FiniteElement` trait easier, for elements whose geometry dimension and reference element
/// dimension coincide.
pub trait VolumeFiniteElementAllocator<T, GeometryDim>:
    FiniteElementAllocator<T, GeometryDim, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
{
}

/// Helper trait to simplify specifying bounds on generic functions that need to
/// construct element (mass/stiffness) matrices when working with the `FiniteElement` trait.
pub trait FiniteElementMatrixAllocator<T, SolutionDim, GeometryDim>:
    VolumeFiniteElementAllocator<T, GeometryDim>
    + Allocator<T, SolutionDim, GeometryDim>
    + Allocator<T, SolutionDim, SolutionDim>
    + Allocator<T, GeometryDim, SolutionDim>
    + Allocator<(usize, usize), SolutionDim>
where
    T: Scalar,
    GeometryDim: DimName,
    SolutionDim: DimName,
{
}

impl<T, ReferenceDim> ReferenceFiniteElementAllocator<T, ReferenceDim>
    for DefaultAllocator
where
    T: Scalar,
    ReferenceDim: DimName,
    DefaultAllocator:
        Allocator<T, ReferenceDim, ReferenceDim>
        + Allocator<T, ReferenceDim, U1>
        + Allocator<T, U1, ReferenceDim>
        + Allocator<(usize, usize), ReferenceDim>,
{
}

impl<T, GeometryDim, ReferenceDim>
    FiniteElementAllocator<T, GeometryDim, ReferenceDim> for DefaultAllocator
where
    T: Scalar,
    GeometryDim: DimName,
    ReferenceDim: DimName,
    DefaultAllocator: ReferenceFiniteElementAllocator<T, ReferenceDim>
        + Allocator<T, GeometryDim>
        + Allocator<T, U1>
        + Allocator<T, GeometryDim, GeometryDim>
        + Allocator<T, GeometryDim, ReferenceDim>
        + Allocator<T, GeometryDim, U1>
        + Allocator<T, U1, GeometryDim>
        + Allocator<T, ReferenceDim, ReferenceDim>
        + Allocator<T, ReferenceDim, U1>
        + Allocator<T, U1, ReferenceDim>
        + Allocator<T, ReferenceDim, GeometryDim>
        + Allocator<(usize, usize), GeometryDim>
        + Allocator<(usize, usize), ReferenceDim>,
{
}

impl<T, SolutionDim, GeometryDim>
    FiniteElementMatrixAllocator<T, SolutionDim, GeometryDim> for DefaultAllocator
where
    T: Scalar,
    GeometryDim: DimName,
    SolutionDim: DimName,
    DefaultAllocator: VolumeFiniteElementAllocator<T, GeometryDim>
        + Allocator<T, SolutionDim, GeometryDim>
        + Allocator<T, SolutionDim, SolutionDim>
        + Allocator<T, GeometryDim, SolutionDim>
        + Allocator<(usize, usize), SolutionDim>,
{
}

impl<T, GeometryDim> VolumeFiniteElementAllocator<T, GeometryDim>
    for DefaultAllocator
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: FiniteElementAllocator<T, GeometryDim, GeometryDim>,
{
}

pub trait ElementConnectivityAllocator<T, Connectivity>:
    FiniteElementAllocator<
    T,
    ConnectivityGeometryDim<T, Connectivity>,
    ConnectivityReferenceDim<T, Connectivity>,
>
where
    T: Scalar,
    Connectivity: ElementConnectivity<T>,
    DefaultAllocator: FiniteElementAllocator<
        T,
        ConnectivityGeometryDim<T, Connectivity>,
        ConnectivityReferenceDim<T, Connectivity>,
    >,
{
}

impl<T, C> ElementConnectivityAllocator<T, C> for DefaultAllocator
where
    T: Scalar,
    C: ElementConnectivity<T>,
    DefaultAllocator: FiniteElementAllocator<
        T,
        ConnectivityGeometryDim<T, C>,
        ConnectivityReferenceDim<T, C>,
    >,
{
}

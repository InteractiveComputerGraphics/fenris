//! Helper traits for allocator trait bounds.
use crate::element::{ConnectivityGeometryDim, ConnectivityReferenceDim, ElementConnectivity};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Scalar, U1};

pub trait ElementConnectivityAllocator<T, Connectivity>:
    BiDimAllocator<T, ConnectivityGeometryDim<T, Connectivity>, ConnectivityReferenceDim<T, Connectivity>>
where
    T: Scalar,
    Connectivity: ElementConnectivity<T>,
    DefaultAllocator:
        BiDimAllocator<T, ConnectivityGeometryDim<T, Connectivity>, ConnectivityReferenceDim<T, Connectivity>>,
{
}

impl<T, C> ElementConnectivityAllocator<T, C> for DefaultAllocator
where
    T: Scalar,
    C: ElementConnectivity<T>,
    DefaultAllocator: BiDimAllocator<T, ConnectivityGeometryDim<T, C>, ConnectivityReferenceDim<T, C>>,
{
}

pub use fenris_traits::allocators::*;
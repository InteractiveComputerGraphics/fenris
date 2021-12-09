//! Helper traits for collecting element allocator trait bounds.

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

// TODO: The SmallDimAllocator, BiDimAllocator and TriDimAllocator classes should make
// many/most of the other allocators redundant, I think?
// TODO: Rename to `UniDimAllocator` or similar? Or just `DimAllocator`?
pub trait SmallDimAllocator<T: Scalar, D: DimName>:
Allocator<T, D> + Allocator<T, D, D> + Allocator<T, U1, D>
// Used for various functionality like decompositions
+ Allocator<usize, D>
+ Allocator<(usize, usize), D>
{}

impl<T, D> SmallDimAllocator<T, D> for DefaultAllocator
where
    T: Scalar,
    D: DimName,
    DefaultAllocator:
        Allocator<T, D> + Allocator<T, D, D> + Allocator<T, U1, D> + Allocator<usize, D> + Allocator<(usize, usize), D>,
{
}

pub trait BiDimAllocator<T: Scalar, D1: DimName, D2: DimName>:
    SmallDimAllocator<T, D1> + SmallDimAllocator<T, D2> + Allocator<T, D1, D2> + Allocator<T, D2, D1>
{
}

impl<T: Scalar, D1: DimName, D2: DimName> BiDimAllocator<T, D1, D2> for DefaultAllocator where
    DefaultAllocator: SmallDimAllocator<T, D1> + SmallDimAllocator<T, D2> + Allocator<T, D1, D2> + Allocator<T, D2, D1>
{
}

pub trait TriDimAllocator<T: Scalar, D1: DimName, D2: DimName, D3: DimName>:
    BiDimAllocator<T, D1, D2> + BiDimAllocator<T, D1, D3> + BiDimAllocator<T, D2, D3>
{
}

impl<T: Scalar, D1: DimName, D2: DimName, D3: DimName> TriDimAllocator<T, D1, D2, D3> for DefaultAllocator where
    DefaultAllocator: BiDimAllocator<T, D1, D2> + BiDimAllocator<T, D1, D3> + BiDimAllocator<T, D2, D3>
{
}

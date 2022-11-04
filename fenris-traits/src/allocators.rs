//! Helper traits for allocator trait bounds.
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Scalar, U1};

/// An allocator for a single dimension.
pub trait DimAllocator<T: Scalar, D: DimName>:
Allocator<T, D>
+ Allocator<T, D, D>
+ Allocator<T, U1, D>
// Used for various functionality like decompositions
+ Allocator<usize, D>
+ Allocator<(usize, usize), D>
// Provide allocators for built-in types so that we don't need separate traits to use these
// if we already have Allocator<T, D>
+ Allocator<f32, D>
+ Allocator<f64, D>
+ Allocator<i8, D>
+ Allocator<i32, D>
+ Allocator<i64, D>
+ Allocator<u8, D>
+ Allocator<u16, D>
+ Allocator<u32, D>
+ Allocator<u64, D>
+ Allocator<isize, D>
+ Allocator<bool, D>
{}

impl<T, D> DimAllocator<T, D> for DefaultAllocator
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, D, D>
        + Allocator<T, U1, D>
        + Allocator<usize, D>
        + Allocator<(usize, usize), D>
        + Allocator<f32, D>
        + Allocator<f64, D>
        + Allocator<i8, D>
        + Allocator<i32, D>
        + Allocator<i64, D>
        + Allocator<u8, D>
        + Allocator<u16, D>
        + Allocator<u32, D>
        + Allocator<u64, D>
        + Allocator<isize, D>
        + Allocator<bool, D>,
{
}

/// An allocator for two dimensions.
pub trait BiDimAllocator<T: Scalar, D1: DimName, D2: DimName>:
    DimAllocator<T, D1> + DimAllocator<T, D2> + Allocator<T, D1, D2> + Allocator<T, D2, D1>
{
}

impl<T: Scalar, D1: DimName, D2: DimName> BiDimAllocator<T, D1, D2> for DefaultAllocator where
    DefaultAllocator: DimAllocator<T, D1> + DimAllocator<T, D2> + Allocator<T, D1, D2> + Allocator<T, D2, D1>
{
}

/// An allocator for three dimensions.
pub trait TriDimAllocator<T: Scalar, D1: DimName, D2: DimName, D3: DimName>:
    BiDimAllocator<T, D1, D2> + BiDimAllocator<T, D1, D3> + BiDimAllocator<T, D2, D3>
{
}

impl<T: Scalar, D1: DimName, D2: DimName, D3: DimName> TriDimAllocator<T, D1, D2, D3> for DefaultAllocator where
    DefaultAllocator: BiDimAllocator<T, D1, D2> + BiDimAllocator<T, D1, D3> + BiDimAllocator<T, D2, D3>
{
}

pub trait QuadDimAllocator<T: Scalar, D1: DimName, D2: DimName, D3: DimName, D4: DimName>:
    TriDimAllocator<T, D1, D2, D3>
    + TriDimAllocator<T, D1, D2, D4>
    + TriDimAllocator<T, D1, D3, D4>
    + TriDimAllocator<T, D2, D3, D4>
{
}

impl<T: Scalar, D1: DimName, D2: DimName, D3: DimName, D4: DimName> QuadDimAllocator<T, D1, D2, D3, D4>
    for DefaultAllocator
where
    DefaultAllocator: TriDimAllocator<T, D1, D2, D3>
        + TriDimAllocator<T, D1, D2, D4>
        + TriDimAllocator<T, D1, D3, D4>
        + TriDimAllocator<T, D2, D3, D4>,
{
}

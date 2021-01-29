use crate::allocators::ElementConnectivityAllocator;
use crate::element::{ConnectivityGeometryDim, ElementConnectivity, ElementForSpace};
use crate::geometry::GeometryCollection;
use nalgebra::{DefaultAllocator, Point, Scalar};

pub trait FiniteElementSpace<T>
where
    T: Scalar,
    DefaultAllocator: ElementConnectivityAllocator<T, Self::Connectivity>,
{
    type Connectivity: ElementConnectivity<T>;

    fn vertices(&self) -> &[Point<T, ConnectivityGeometryDim<T, Self::Connectivity>>];

    fn num_connectivities(&self) -> usize;

    fn get_connectivity(&self, index: usize) -> Option<&Self::Connectivity>;

    fn get_element(&self, index: usize) -> Option<ElementForSpace<T, Self>> {
        self.get_connectivity(index)
            .map(|conn| conn.element(self.vertices()).unwrap())
    }
}

/// A finite element space whose elements can be seen as a collection of geometric entities.
///
/// This trait essentially functions as a marker trait for finite element spaces which can
/// also be interpreted as a collection of geometry objects, with a 1:1 correspondence between
/// elements and geometries.
pub trait GeometricFiniteElementSpace<'a, T>:
    FiniteElementSpace<T> + GeometryCollection<'a>
where
    T: Scalar,
    DefaultAllocator: ElementConnectivityAllocator<T, Self::Connectivity>,
{
}

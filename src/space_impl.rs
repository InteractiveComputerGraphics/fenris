use crate::allocators::ElementConnectivityAllocator;
use crate::connectivity::CellConnectivity;
use crate::element::ElementConnectivity;
use crate::mesh::{Mesh};
use crate::model::NodalModel;
use crate::space::{FiniteElementSpace, GeometricFiniteElementSpace};
use nalgebra::{DefaultAllocator, DimName, Point, Scalar};

impl<T, D, C> FiniteElementSpace<T> for Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: ElementConnectivity<T, GeometryDim = D>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
    type Connectivity = C;

    fn vertices(&self) -> &[Point<T, D>] {
        self.vertices()
    }

    fn num_connectivities(&self) -> usize {
        self.connectivity().len()
    }

    fn get_connectivity(&self, index: usize) -> Option<&Self::Connectivity> {
        self.connectivity().get(index)
    }
}

impl<'a, T, D, C> GeometricFiniteElementSpace<'a, T> for Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: CellConnectivity<T, D> + ElementConnectivity<T, GeometryDim = D>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
}

impl<T, D, C> FiniteElementSpace<T> for NodalModel<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: ElementConnectivity<T, GeometryDim = D>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
    type Connectivity = C;

    fn vertices(&self) -> &[Point<T, D>] {
        self.vertices()
    }

    fn num_connectivities(&self) -> usize {
        self.connectivity().len()
    }

    fn get_connectivity(&self, index: usize) -> Option<&Self::Connectivity> {
        self.connectivity().get(index)
    }
}

impl<'a, T, D, C> GeometricFiniteElementSpace<'a, T> for NodalModel<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: CellConnectivity<T, D> + ElementConnectivity<T, GeometryDim = D>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
}

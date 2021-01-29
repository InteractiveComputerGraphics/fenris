use crate::allocators::ElementConnectivityAllocator;
use crate::connectivity::CellConnectivity;
use crate::element::ElementConnectivity;
use crate::mesh::{ClosedSurfaceMesh2d, Mesh, Mesh2d};
use crate::model::NodalModel;
use crate::space::{FiniteElementSpace, GeometricFiniteElementSpace};
use nalgebra::{DefaultAllocator, DimName, Point, Point2, RealField, Scalar, U2};

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

impl<T, Connectivity> FiniteElementSpace<T> for ClosedSurfaceMesh2d<T, Connectivity>
where
    T: RealField,
    Connectivity: ElementConnectivity<T, GeometryDim = U2>,
    DefaultAllocator: ElementConnectivityAllocator<T, Connectivity>,
{
    type Connectivity = <Mesh2d<T, Connectivity> as FiniteElementSpace<T>>::Connectivity;

    fn vertices(&self) -> &[Point2<T>] {
        self.mesh().vertices()
    }

    fn num_connectivities(&self) -> usize {
        self.mesh().connectivity().len()
    }

    fn get_connectivity(&self, index: usize) -> Option<&Self::Connectivity> {
        self.mesh().connectivity().get(index)
    }
}

impl<'a, T, C> GeometricFiniteElementSpace<'a, T> for ClosedSurfaceMesh2d<T, C>
where
    T: RealField,
    C: CellConnectivity<T, U2> + ElementConnectivity<T, GeometryDim = U2>,
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

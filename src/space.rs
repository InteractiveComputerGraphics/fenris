use crate::allocators::{FiniteElementAllocator};
use crate::element::{
    MatrixSliceMut,
};
use crate::geometry::GeometryCollection;
use crate::nalgebra::{Dynamic, MatrixMN, U1};
use crate::SmallDim;
use nalgebra::{DefaultAllocator, Point, Scalar};

/// Describes the connectivity of elements in a finite element space.
pub trait FiniteElementConnectivity {
    fn num_elements(&self) -> usize;

    fn num_nodes(&self) -> usize;

    fn element_node_count(&self, element_index: usize) -> usize;

    fn populate_element_nodes(&self, nodes: &mut [usize], element_index: usize);
}

/// The "new" FiniteElementSpace trait. Currently playground for new design
pub trait FiniteElementSpace2<T: Scalar>: FiniteElementConnectivity
where
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    type GeometryDim: SmallDim;
    type ReferenceDim: SmallDim;

    fn populate_element_basis(
        &self,
        element_index: usize,
        basis_values: MatrixSliceMut<T, U1, Dynamic>,
        reference_coords: &Point<T, Self::ReferenceDim>,
    );

    fn populate_element_gradients(
        &self,
        element_index: usize,
        gradients: MatrixSliceMut<T, Self::ReferenceDim, Dynamic>,
        reference_coords: &Point<T, Self::ReferenceDim>,
    );

    /// Compute the Jacobian of the transformation from the reference element to the given
    /// element at the given reference coordinates.
    fn element_reference_jacobian(
        &self,
        element_index: usize,
        reference_coords: &Point<T, Self::ReferenceDim>,
    ) -> MatrixMN<T, Self::GeometryDim, Self::ReferenceDim>;

    /// Maps reference coordinates to physical coordinates in the element.
    fn map_element_reference_coords(
        &self,
        element_index: usize,
        reference_coords: &Point<T, Self::ReferenceDim>,
    ) -> Point<T, Self::GeometryDim>;

    /// The diameter of the finite element.
    ///
    /// The diameter of a finite element is defined as the largest distance between any two
    /// points in the element, i.e.
    ///  h = min |x - y| for x, y in K
    /// where K is the element and h is the diameter.
    fn diameter(&self, element_index: usize) -> T;
}

/// A finite element space where `GeometryDim == ReferenceDim`.
pub trait VolumetricFiniteElementSpace<T>:
    FiniteElementSpace2<T, GeometryDim = <Self as FiniteElementSpace2<T>>::ReferenceDim>
where
    T: Scalar,
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

impl<T, S> VolumetricFiniteElementSpace<T> for S
where
    T: Scalar,
    S: FiniteElementSpace2<T, GeometryDim = <Self as FiniteElementSpace2<T>>::ReferenceDim>,
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

/// A finite element space whose elements can be seen as a collection of geometric entities.
///
/// This trait essentially functions as a marker trait for finite element spaces which can
/// also be interpreted as a collection of geometry objects, with a 1:1 correspondence between
/// elements and geometries.
pub trait GeometricFiniteElementSpace<'a, T>:
    FiniteElementSpace2<T> + GeometryCollection<'a>
where
    T: Scalar,
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

use crate::allocators::BiDimAllocator;
use crate::element::{FiniteElement, ReferenceFiniteElement};
use crate::geometry::GeometryCollection;
use crate::nalgebra::{Dynamic, MatrixSliceMut, OMatrix};
use crate::SmallDim;
use nalgebra::{DefaultAllocator, OPoint, Scalar};

/// Describes the connectivity of elements in a finite element space.
pub trait FiniteElementConnectivity {
    fn num_elements(&self) -> usize;

    fn num_nodes(&self) -> usize;

    fn element_node_count(&self, element_index: usize) -> usize;

    fn populate_element_nodes(&self, nodes: &mut [usize], element_index: usize);
}

/// The "new" FiniteElementSpace trait. Currently playground for new design
pub trait FiniteElementSpace<T: Scalar>: FiniteElementConnectivity
where
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    type GeometryDim: SmallDim;
    type ReferenceDim: SmallDim;

    fn populate_element_basis(
        &self,
        element_index: usize,
        basis_values: &mut [T],
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    );

    fn populate_element_gradients(
        &self,
        element_index: usize,
        gradients: MatrixSliceMut<T, Self::ReferenceDim, Dynamic>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    );

    /// Compute the Jacobian of the transformation from the reference element to the given
    /// element at the given reference coordinates.
    fn element_reference_jacobian(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim>;

    /// Maps reference coordinates to physical coordinates in the element.
    fn map_element_reference_coords(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OPoint<T, Self::GeometryDim>;

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
    FiniteElementSpace<T, GeometryDim = <Self as FiniteElementSpace<T>>::ReferenceDim>
where
    T: Scalar,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

impl<T, S> VolumetricFiniteElementSpace<T> for S
where
    T: Scalar,
    S: FiniteElementSpace<T, GeometryDim = <Self as FiniteElementSpace<T>>::ReferenceDim>,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

/// A finite element space whose elements can be seen as a collection of geometric entities.
///
/// This trait essentially functions as a marker trait for finite element spaces which can
/// also be interpreted as a collection of geometry objects, with a 1:1 correspondence between
/// elements and geometries.
pub trait GeometricFiniteElementSpace<'a, T>: FiniteElementSpace<T> + GeometryCollection<'a>
where
    T: Scalar,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

/// A convenience wrapper for producing a [`FiniteElement`] from an indexed element in a
/// [`FiniteElementSpace`].
#[derive(Debug)]
pub struct ElementInSpace<'a, Space> {
    space: &'a Space,
    element_index: usize,
}

impl<'a, Space> ElementInSpace<'a, Space> {
    pub fn from_space_and_element_index(space: &'a Space, element_index: usize) -> Self {
        Self { space, element_index }
    }
}

impl<'a, T, Space> ReferenceFiniteElement<T> for ElementInSpace<'a, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
{
    type ReferenceDim = Space::ReferenceDim;

    fn num_nodes(&self) -> usize {
        self.space.element_node_count(self.element_index)
    }

    fn populate_basis(&self, basis_values: &mut [T], reference_coords: &OPoint<T, Self::ReferenceDim>) {
        self.space
            .populate_element_basis(self.element_index, basis_values, reference_coords)
    }

    fn populate_basis_gradients(
        &self,
        basis_gradients: MatrixSliceMut<T, Self::ReferenceDim, Dynamic>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) {
        self.space
            .populate_element_gradients(self.element_index, basis_gradients, reference_coords)
    }
}

impl<'a, T, Space> FiniteElement<T> for ElementInSpace<'a, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
{
    type GeometryDim = Space::GeometryDim;

    fn reference_jacobian(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim> {
        self.space
            .element_reference_jacobian(self.element_index, reference_coords)
    }

    fn map_reference_coords(&self, reference_coords: &OPoint<T, Self::ReferenceDim>) -> OPoint<T, Self::GeometryDim> {
        self.space
            .map_element_reference_coords(self.element_index, reference_coords)
    }

    fn diameter(&self) -> T {
        self.space.diameter(self.element_index)
    }
}

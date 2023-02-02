use crate::allocators::ElementConnectivityAllocator;
use crate::connectivity::CellConnectivity;
use crate::element::{
    BoundsForElement, ClosestPoint, ClosestPointInElement, ElementConnectivity, FiniteElement, ReferenceFiniteElement,
};
use crate::mesh::Mesh;
use crate::nalgebra::{Dyn, MatrixViewMut, OMatrix};
use crate::space::{
    BoundsForElementInSpace, ClosestPointInElementInSpace, FiniteElementConnectivity, FiniteElementSpace,
    GeometricFiniteElementSpace,
};
use crate::SmallDim;
use fenris_geometry::AxisAlignedBoundingBox;
use fenris_traits::allocators::BiDimAllocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Scalar};

impl<T, D, C> FiniteElementConnectivity for Mesh<T, D, C>
where
    T: Scalar,
    C: ElementConnectivity<T, GeometryDim = D>,
    D: SmallDim,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
    fn num_elements(&self) -> usize {
        self.connectivity().len()
    }

    fn num_nodes(&self) -> usize {
        self.vertices().len()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .vertex_indices()
            .len()
    }

    fn populate_element_nodes(&self, nodes: &mut [usize], element_index: usize) {
        let indices = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .vertex_indices();
        assert_eq!(
            indices.len(),
            nodes.len(),
            "Incompatible slice length for node population"
        );
        nodes.copy_from_slice(&indices);
    }
}

impl<'a, T, D, C> GeometricFiniteElementSpace<'a, T> for Mesh<T, D, C>
where
    T: Scalar,
    D: SmallDim,
    C: CellConnectivity<T, D> + ElementConnectivity<T, GeometryDim = D>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
}

impl<T, D, C> FiniteElementSpace<T> for Mesh<T, D, C>
where
    T: Scalar,
    D: SmallDim,
    C: ElementConnectivity<T, GeometryDim = D>,
    C::ReferenceDim: SmallDim,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
    type GeometryDim = D;
    type ReferenceDim = C::ReferenceDim;

    fn populate_element_basis(
        &self,
        element_index: usize,
        basis_values: &mut [T],
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        assert_eq!(
            basis_values.len(),
            element.num_nodes(),
            "Incompatible slice shape for basis values"
        );
        element.populate_basis(basis_values, &reference_coords)
    }

    fn populate_element_gradients(
        &self,
        element_index: usize,
        gradients: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        assert_eq!(
            gradients.shape(),
            (C::ReferenceDim::dim(), element.num_nodes()),
            "Incompatible slice shape for basis gradients"
        );
        element.populate_basis_gradients(gradients, &reference_coords)
    }

    fn element_reference_jacobian(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim> {
        // TODO: Avoid this repetition
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        element.reference_jacobian(&reference_coords)
    }

    fn map_element_reference_coords(
        &self,
        element_index: usize,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OPoint<T, Self::GeometryDim> {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        OPoint::from(element.map_reference_coords(&reference_coords))
    }

    fn diameter(&self, element_index: usize) -> T {
        let element = self
            .connectivity()
            .get(element_index)
            .expect("Element index out of bounds")
            .element(self.vertices())
            .unwrap();
        element.diameter()
    }
}

impl<T, D, C> ClosestPointInElementInSpace<T> for Mesh<T, D, C>
where
    T: Scalar,
    D: SmallDim,
    C: ElementConnectivity<T, GeometryDim = D>,
    C::Element: ClosestPointInElement<T>,
    DefaultAllocator: BiDimAllocator<T, C::GeometryDim, C::ReferenceDim>,
{
    fn closest_point_in_element(
        &self,
        element_index: usize,
        p: &OPoint<T, Self::GeometryDim>,
    ) -> ClosestPoint<T, Self::ReferenceDim> {
        let conn = &self.connectivity()[element_index];
        conn.element(self.vertices()).unwrap().closest_point(p)
    }
}

impl<T, D, C> BoundsForElementInSpace<T> for Mesh<T, D, C>
where
    T: Scalar,
    D: SmallDim,
    C: ElementConnectivity<T, GeometryDim = D>,
    C::Element: BoundsForElement<T>,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
    fn bounds_for_element(&self, element_index: usize) -> AxisAlignedBoundingBox<T, Self::GeometryDim> {
        let conn = &self.connectivity()[element_index];
        conn.element(self.vertices()).unwrap().element_bounds()
    }
}

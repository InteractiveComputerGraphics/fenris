use crate::allocators::ElementConnectivityAllocator;
use crate::connectivity::CellConnectivity;
use crate::element::{ElementConnectivity, FiniteElement, ReferenceFiniteElement};
use crate::mesh::Mesh;
use crate::nalgebra::{Dynamic, MatrixSliceMut, OMatrix};
use crate::space::{FiniteElementConnectivity, FiniteElementSpace, GeometricFiniteElementSpace};
use crate::SmallDim;
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
        gradients: MatrixSliceMut<T, Self::ReferenceDim, Dynamic>,
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

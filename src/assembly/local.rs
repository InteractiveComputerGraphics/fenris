use crate::allocators::FiniteElementMatrixAllocator;
use crate::connectivity::Connectivity;
use crate::element::VolumetricFiniteElement;
use crate::mesh::Mesh;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{
    DMatrix, DMatrixSliceMut, DefaultAllocator, DimMin, DimName, RealField, Scalar, U1,
};
use crate::nalgebra::{DVectorSliceMut, Point};
use crate::quadrature::Quadrature;
use crate::SmallDim;

mod elliptic;
mod source;

pub use elliptic::*;
pub use source::*;

pub trait ElementConnectivityAssembler {
    fn solution_dim(&self) -> usize;

    fn num_elements(&self) -> usize;

    fn num_nodes(&self) -> usize;

    fn element_node_count(&self, element_index: usize) -> usize;

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize);
}

impl<T, D, C> ElementConnectivityAssembler for Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
    fn solution_dim(&self) -> usize {
        1
    }

    fn num_elements(&self) -> usize {
        self.connectivity().len()
    }

    fn num_nodes(&self) -> usize {
        self.vertices().len()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.connectivity()[element_index].vertex_indices().len()
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        output.copy_from_slice(self.connectivity()[element_index].vertex_indices());
    }
}

pub trait ElementMatrixAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_matrix_into(
        &self,
        element_index: usize,
        output: DMatrixSliceMut<T>,
    ) -> eyre::Result<()>;

    fn as_connectivity_assembler(&self) -> &dyn ElementConnectivityAssembler;
}

/// Assemble the generalized element matrix for the given element.
///
/// TODO: Allow non-uniform density. Possible API: Take callback that gets both index of quadrature
/// point and position of quadrature point...? That way one can i.e. associate density with
/// a particular quadrature point (or return an element-wise constant) or use an analytic function.
#[allow(non_snake_case)]
pub fn assemble_generalized_element_mass<T, SolutionDim, Element, Q>(
    mut element_matrix: DMatrixSliceMut<T>,
    element: &Element,
    density: T,
    quadrature: &Q,
) where
    T: RealField,
    Element: VolumetricFiniteElement<T>,
    Element::GeometryDim: DimMin<Element::GeometryDim, Output = Element::GeometryDim>,
    SolutionDim: DimName,
    Q: Quadrature<T, Element::GeometryDim>,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Element::GeometryDim>,
{
    // TODO: Avoid allocation!!!
    let mut m_element_scalar = DMatrix::zeros(element.num_nodes(), element.num_nodes());
    let mut basis_values = vec![T::zero(); element.num_nodes()];

    let weights = quadrature.weights();
    let points = quadrature.points();

    let num_nodes = element.num_nodes();
    let sol_dim = SolutionDim::dim();

    for (&w, xi) in weights.iter().zip(points) {
        let J = element.reference_jacobian(xi);
        let J_det = J.determinant();

        element.populate_basis(&mut basis_values, xi);
        let phi = &basis_values;

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                // Product of shape functions
                m_element_scalar[(i, j)] += density * J_det.abs() * w * phi[i] * phi[j];
            }
        }
    }

    let skip_shape = (sol_dim.saturating_sub(1), sol_dim.saturating_sub(1));
    for i in 0..sol_dim {
        element_matrix
            .slice_with_steps_mut((i, i), (num_nodes, num_nodes), skip_shape)
            .copy_from(&m_element_scalar);
    }
}

pub trait ElementVectorAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        output: DVectorSliceMut<T>,
    ) -> eyre::Result<()>;
}

/// Lookup table mapping elements to quadrature rules.
///
/// TODO: Eventually replace the existing trait with this one
pub trait QuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
{
    type Data: Default + Clone;

    fn element_quadrature_size(&self, element_index: usize) -> usize;

    fn populate_element_data(&self, element_index: usize, data: &mut [Self::Data]);

    fn populate_element_quadrature(
        &self,
        element_index: usize,
        points: &mut [Point<T, GeometryDim>],
        weights: &mut [T],
    );

    fn populate_element_quadrature_and_data(
        &self,
        element_index: usize,
        points: &mut [Point<T, GeometryDim>],
        weights: &mut [T],
        data: &mut [Self::Data],
    ) {
        self.populate_element_quadrature(element_index, points, weights);
        self.populate_element_data(element_index, data);
    }
}

#[derive(Debug)]
pub struct UniformQuadratureTable<T, GeometryDim, Data = ()>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    points: Vec<Point<T, GeometryDim>>,
    weights: Vec<T>,
    data: Vec<Data>,
}

impl<T, GeometryDim> UniformQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_and_weights(points: Vec<Point<T, GeometryDim>>, weights: Vec<T>) -> Self {
        let data = vec![(); points.len()];
        Self::from_points_weights_and_data(points, weights, data)
    }
}

impl<T, GeometryDim, Data> UniformQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_weights_and_data(
        points: Vec<Point<T, GeometryDim>>,
        weights: Vec<T>,
        data: Vec<Data>,
    ) -> Self {
        let msg = "Points, weights and data must have the same length.";
        assert_eq!(points.len(), weights.len(), "{}", msg);
        assert_eq!(points.len(), data.len(), "{}", msg);
        Self {
            points,
            weights,
            data,
        }
    }
}

impl<T, GeometryDim, Data> QuadratureTable<T, GeometryDim>
    for UniformQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: SmallDim,
    Data: Clone + Default,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    type Data = Data;

    fn element_quadrature_size(&self, _element_index: usize) -> usize {
        self.points.len()
    }

    fn populate_element_data(&self, _element_index: usize, data: &mut [Self::Data]) {
        assert_eq!(data.len(), self.data.len());
        data.clone_from_slice(&self.data);
    }

    fn populate_element_quadrature(
        &self,
        _element_index: usize,
        points: &mut [Point<T, GeometryDim>],
        weights: &mut [T],
    ) {
        assert_eq!(points.len(), self.points.len());
        assert_eq!(weights.len(), self.weights.len());
        points.clone_from_slice(&self.points);
        weights.clone_from_slice(&self.weights);
    }
}

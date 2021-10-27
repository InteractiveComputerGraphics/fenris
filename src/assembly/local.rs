use crate::connectivity::Connectivity;
use crate::mesh::Mesh;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{DMatrixSliceMut, DefaultAllocator, DimName, Scalar, U1};
use crate::nalgebra::{DVectorSliceMut, OPoint};
use crate::quadrature::QuadraturePair;
use crate::util::NestedVec;
use crate::SmallDim;
use itertools::izip;
use serde::{Deserialize, Serialize};

mod elliptic;
mod mass;
mod source;

pub use elliptic::*;
pub use mass::*;
pub use source::*;

pub trait ElementConnectivityAssembler {
    fn solution_dim(&self) -> usize;

    fn num_elements(&self) -> usize;

    fn num_nodes(&self) -> usize;

    fn element_node_count(&self, element_index: usize) -> usize;

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize);

    /// Returns an adapter that modifies element node indices according to the provided function.
    fn map_element_nodes<F>(self, f: F) -> MapElementNodes<Self, F>
    where
        Self: Sized,
    {
        MapElementNodes {
            mapped: self,
            function: f,
        }
    }
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
    fn assemble_element_matrix_into(&self, element_index: usize, output: DMatrixSliceMut<T>) -> eyre::Result<()>;
}

pub trait ElementVectorAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_vector_into(&self, element_index: usize, output: DVectorSliceMut<T>) -> eyre::Result<()>;
}

pub trait ElementScalarAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T>;
}

pub struct MapElementNodes<Mapped, F> {
    mapped: Mapped,
    function: F,
}

impl<Assembler, F> ElementConnectivityAssembler for MapElementNodes<Assembler, F>
where
    Assembler: ElementConnectivityAssembler,
    F: Fn(usize) -> usize,
{
    fn solution_dim(&self) -> usize {
        self.mapped.solution_dim()
    }

    fn num_elements(&self) -> usize {
        self.mapped.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.mapped.num_elements()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.mapped.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.mapped.populate_element_nodes(output, element_index);
        for idx in output {
            *idx = (self.function)(*idx);
        }
    }
}

impl<T, Assembler, F> ElementScalarAssembler<T> for MapElementNodes<Assembler, F>
where
    T: Scalar,
    Assembler: ElementScalarAssembler<T>,
    F: Fn(usize) -> usize,
{
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T> {
        self.mapped.assemble_element_scalar(element_index)
    }
}

impl<T, Assembler, F> ElementVectorAssembler<T> for MapElementNodes<Assembler, F>
where
    T: Scalar,
    Assembler: ElementVectorAssembler<T>,
    F: Fn(usize) -> usize,
{
    fn assemble_element_vector_into(&self, element_index: usize, output: DVectorSliceMut<T>) -> eyre::Result<()> {
        self.mapped
            .assemble_element_vector_into(element_index, output)
    }
}

impl<T, Assembler, F> ElementMatrixAssembler<T> for MapElementNodes<Assembler, F>
where
    T: Scalar,
    Assembler: ElementMatrixAssembler<T>,
    F: Fn(usize) -> usize,
{
    fn assemble_element_matrix_into(&self, element_index: usize, output: DMatrixSliceMut<T>) -> eyre::Result<()> {
        self.mapped
            .assemble_element_matrix_into(element_index, output)
    }
}

/// Lookup table mapping elements to quadrature rules.
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
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    );

    fn populate_element_quadrature_and_data(
        &self,
        element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
        data: &mut [Self::Data],
    ) {
        self.populate_element_quadrature(element_index, points, weights);
        self.populate_element_data(element_index, data);
    }
}

/// A quadrature table that keeps a separate quadrature rule per element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneralQuadratureTable<T, GeometryDim, Data = ()>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    #[serde(bound(serialize = "OPoint<T, GeometryDim>: Serialize"))]
    #[serde(bound(deserialize = "OPoint<T, GeometryDim>: Deserialize<'de>"))]
    points: NestedVec<OPoint<T, GeometryDim>>,
    weights: NestedVec<T>,
    data: NestedVec<Data>,
}

impl<T, GeometryDim> GeneralQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_and_weights(points: NestedVec<OPoint<T, GeometryDim>>, weights: NestedVec<T>) -> Self {
        let mut data = NestedVec::new();
        for i in 0..points.len() {
            data.push(&vec![(); points.get(i).unwrap().len()]);
        }
        Self::from_points_weights_and_data(points, weights, data)
    }
}

impl<T, GeometryDim, Data> GeneralQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_weights_and_data(
        points: NestedVec<OPoint<T, GeometryDim>>,
        weights: NestedVec<T>,
        data: NestedVec<Data>,
    ) -> Self {
        assert_eq!(points.len(), weights.len());
        assert_eq!(points.len(), data.len());

        // Ensure that each element has a consistent quadrature rule
        let iter = izip!(points.iter(), weights.iter(), data.iter());
        for (element_index, (element_points, element_weights, element_data)) in iter.enumerate() {
            assert_eq!(
                element_points.len(),
                element_weights.len(),
                "Element {} has mismatched number of points and weights.",
                element_index
            );
            assert_eq!(
                element_points.len(),
                element_data.len(),
                "Element {} has mismatched number of points and data.",
                element_index
            );
        }

        Self { points, weights, data }
    }
}

impl<T, GeometryDim, Data> QuadratureTable<T, GeometryDim> for GeneralQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: SmallDim,
    Data: Clone + Default,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    type Data = Data;

    fn element_quadrature_size(&self, element_index: usize) -> usize {
        // TODO: Should we rather return results from all these methods? It seems that currently
        // we are just panicking if the quadrature table size doesn't match the number of elements
        // in the finite element space. This seems bad.
        self.weights
            .get(element_index)
            .expect("Element index out of bounds")
            .len()
    }

    fn populate_element_data(&self, element_index: usize, data: &mut [Self::Data]) {
        let data_for_element = self
            .data
            .get(element_index)
            .expect("Element index out of bounds");
        assert_eq!(data_for_element.len(), data.len());
        data.clone_from_slice(&data_for_element);
    }

    fn populate_element_quadrature(
        &self,
        element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    ) {
        let points_for_element = self
            .points
            .get(element_index)
            .expect("Element index out of bounds");
        let weights_for_element = self
            .weights
            .get(element_index)
            .expect("Element index out of bounds");
        assert_eq!(points_for_element.len(), points.len());
        assert_eq!(weights_for_element.len(), weights.len());
        points.clone_from_slice(&points_for_element);
        weights.clone_from_slice(&weights_for_element);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniformQuadratureTable<T, GeometryDim, Data = ()>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    #[serde(bound(serialize = "OPoint<T, GeometryDim>: Serialize"))]
    #[serde(bound(deserialize = "OPoint<T, GeometryDim>: Deserialize<'de>"))]
    points: Vec<OPoint<T, GeometryDim>>,
    weights: Vec<T>,
    data: Vec<Data>,
}

impl<T, GeometryDim> UniformQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_and_weights(points: Vec<OPoint<T, GeometryDim>>, weights: Vec<T>) -> Self {
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
    pub fn from_points_weights_and_data(points: Vec<OPoint<T, GeometryDim>>, weights: Vec<T>, data: Vec<Data>) -> Self {
        let msg = "Points, weights and data must have the same length.";
        assert_eq!(points.len(), weights.len(), "{}", msg);
        assert_eq!(points.len(), data.len(), "{}", msg);
        Self { points, weights, data }
    }

    pub fn from_quadrature_and_uniform_data(quadrature: QuadraturePair<T, GeometryDim>, data: Data) -> Self
    where
        Data: Clone,
    {
        let (weights, points) = quadrature;
        let data = vec![data; weights.len()];
        Self::from_points_weights_and_data(points, weights, data)
    }
}

impl<T, GeometryDim, Data> QuadratureTable<T, GeometryDim> for UniformQuadratureTable<T, GeometryDim, Data>
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
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    ) {
        assert_eq!(points.len(), self.points.len());
        assert_eq!(weights.len(), self.weights.len());
        points.clone_from_slice(&self.points);
        weights.clone_from_slice(&self.weights);
    }
}

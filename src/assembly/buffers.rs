use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::assembly::local::QuadratureTable;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{
    DMatrix, DefaultAllocator, DimName, Dynamic, MatrixSlice, MatrixSliceMut, OPoint, RealField, Scalar,
};
use crate::space::FiniteElementSpace;
use crate::SmallDim;
use itertools::izip;

#[derive(Debug)]
pub struct BasisFunctionBuffer<T: Scalar> {
    element_nodes: Vec<usize>,
    element_basis_values: Vec<T>,
    element_basis_gradients: DMatrix<T>,
}

impl<T: RealField> Default for BasisFunctionBuffer<T> {
    fn default() -> Self {
        Self {
            element_nodes: Vec::new(),
            element_basis_values: Vec::new(),
            element_basis_gradients: DMatrix::zeros(0, 0),
        }
    }
}

impl<T: RealField> BasisFunctionBuffer<T> {
    pub fn resize(&mut self, node_count: usize, reference_dim: usize) {
        self.element_nodes.resize(node_count, usize::MAX);
        self.element_basis_values.resize(node_count, T::zero());
        self.element_basis_gradients
            .resize_mut(reference_dim, node_count, T::zero());
    }

    pub fn populate_element_nodes_from_space<Space>(&mut self, element_index: usize, space: &Space)
    where
        Space: FiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        space.populate_element_nodes(&mut self.element_nodes, element_index);
    }

    /// TODO: Document that populate_element_nodes should be called first
    pub fn populate_element_basis_values_from_space<Space>(
        &mut self,
        element_index: usize,
        space: &Space,
        reference_coords: &OPoint<T, Space::ReferenceDim>,
    ) where
        Space: FiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        space.populate_element_basis(element_index, &mut self.element_basis_values, reference_coords);
    }

    pub fn populate_element_basis_gradients_from_space<Space>(
        &mut self,
        element_index: usize,
        space: &Space,
        reference_coords: &OPoint<T, Space::ReferenceDim>,
    ) where
        Space: FiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        space.populate_element_gradients(
            element_index,
            MatrixSliceMut::from(&mut self.element_basis_gradients),
            reference_coords,
        );
    }

    pub fn element_nodes(&self) -> &[usize] {
        &self.element_nodes
    }

    pub fn element_basis_values(&self) -> &[T] {
        &self.element_basis_values
    }

    pub fn element_basis_values_mut(&mut self) -> &mut [T] {
        &mut self.element_basis_values
    }

    pub fn element_gradients<D: DimName>(&self) -> MatrixSlice<T, D, Dynamic> {
        MatrixSlice::from(&self.element_basis_gradients)
    }

    pub fn element_gradients_mut<D: DimName>(&mut self) -> MatrixSliceMut<T, D, Dynamic> {
        MatrixSliceMut::from(&mut self.element_basis_gradients)
    }
}

/// A buffer for storing intermediate quadrature data.
#[derive(Debug)]
pub struct QuadratureBuffer<T, D, Data = ()>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    quad_weights: Vec<T>,
    quad_points: Vec<OPoint<T, D>>,
    quad_data: Vec<Data>,
}

impl<T, D, Data> Default for QuadratureBuffer<T, D, Data>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn default() -> Self {
        Self {
            quad_weights: Vec::new(),
            quad_points: Vec::new(),
            quad_data: Vec::new(),
        }
    }
}

impl<T, GeometryDim, Data> QuadratureBuffer<T, GeometryDim, Data>
where
    T: RealField,
    GeometryDim: SmallDim,
    Data: Default + Clone,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    /// Resizes the internal buffer storages to the given size.
    pub fn resize(&mut self, quadrature_size: usize) {
        self.quad_points.resize(quadrature_size, OPoint::origin());
        self.quad_weights.resize(quadrature_size, T::zero());
        self.quad_data.resize(quadrature_size, Data::default());
    }

    /// Populates the buffer by querying a quadrature table with the given element index.
    pub fn populate_element_quadrature_from_table(
        &mut self,
        element_index: usize,
        table: &(impl ?Sized + QuadratureTable<T, GeometryDim, Data = Data>),
    ) {
        let quadrature_size = table.element_quadrature_size(element_index);
        self.resize(quadrature_size);
        table.populate_element_quadrature_and_data(
            element_index,
            &mut self.quad_points,
            &mut self.quad_weights,
            &mut self.quad_data,
        );
    }

    pub fn populate_element_weights_and_points_from_table(
        &mut self,
        element_index: usize,
        table: &(impl ?Sized + QuadratureTable<T, GeometryDim>),
    ) {
        let quadrature_size = table.element_quadrature_size(element_index);
        self.resize(quadrature_size);
        table.populate_element_quadrature(element_index, &mut self.quad_points, &mut self.quad_weights);
    }

    pub fn weights(&self) -> &[T] {
        &self.quad_weights
    }

    pub fn points(&self) -> &[OPoint<T, GeometryDim>] {
        &self.quad_points
    }

    pub fn data(&self) -> &[Data] {
        &self.quad_data
    }

    pub fn weights_and_points(&self) -> (&[T], &[OPoint<T, GeometryDim>]) {
        (self.weights(), self.points())
    }

    /// Calls a closure for each quadrature point currently in the workspace.
    pub fn for_each_quadrature_point<F>(&self, mut f: F) -> eyre::Result<()>
    where
        F: FnMut(T, &OPoint<T, GeometryDim>, &Data) -> eyre::Result<()>,
    {
        assert_eq!(self.quad_weights.len(), self.quad_points.len());
        assert_eq!(self.quad_weights.len(), self.quad_data.len());
        let iter = izip!(&self.quad_weights, &self.quad_points, &self.quad_data);
        for (&w, xi, data) in iter {
            f(w, xi, data)?;
        }
        Ok(())
    }
}

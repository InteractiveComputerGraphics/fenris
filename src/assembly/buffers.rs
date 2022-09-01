use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::assembly::global::gather_global_to_local;
use crate::assembly::local::QuadratureTable;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{DMatrix, DefaultAllocator, DimName, Dynamic, MatrixSlice, MatrixSliceMut, OPoint, Scalar};
use crate::quadrature::Quadrature;
use crate::space::FiniteElementSpace;
use crate::util::compute_interpolation;
use crate::{Real, SmallDim};
use itertools::izip;
use nalgebra::{DVector, DVectorSlice, OMatrix, OVector};

#[derive(Debug)]
pub struct BasisFunctionBuffer<T: Scalar> {
    element_nodes: Vec<usize>,
    element_basis_values: Vec<T>,
    element_basis_gradients: DMatrix<T>,
}

impl<T: Real> Default for BasisFunctionBuffer<T> {
    fn default() -> Self {
        Self {
            element_nodes: Vec::new(),
            element_basis_values: Vec::new(),
            element_basis_gradients: DMatrix::zeros(0, 0),
        }
    }
}

impl<T: Real> BasisFunctionBuffer<T> {
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

impl<T, D, Data> Quadrature<T, D> for QuadratureBuffer<T, D, Data>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Data = Data;

    fn weights(&self) -> &[T] {
        &self.quad_weights
    }

    fn points(&self) -> &[OPoint<T, D>] {
        &self.quad_points
    }

    fn data(&self) -> &[Self::Data] {
        &self.quad_data
    }
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
    T: Real,
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

/// Helper to manage buffers when interpolating FE quantities.
///
/// TODO: Docs
#[derive(Debug)]
pub struct InterpolationBuffer<T: Scalar> {
    basis_buffer: BasisFunctionBuffer<T>,
    u_local: DVector<T>,
}

impl<T: Real> Default for InterpolationBuffer<T> {
    fn default() -> Self {
        Self {
            basis_buffer: Default::default(),
            u_local: DVector::zeros(0),
        }
    }
}

pub struct InterpolationElementBuffer<'a, T: Scalar, Space>
where
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
{
    basis_buffer: &'a mut BasisFunctionBuffer<T>,
    u_local: DVectorSlice<'a, T>,
    space: &'a Space,
    reference_point: OPoint<T, Space::ReferenceDim>,
    element_index: usize,
}

impl<T: Real> InterpolationBuffer<T> {
    pub fn prepare_element_in_space<'a, Space>(
        &'a mut self,
        element_index: usize,
        space: &'a Space,
        u_global: impl Into<DVectorSlice<'a, T>>,
        solution_dim: usize,
    ) -> InterpolationElementBuffer<'a, T, Space>
    where
        Space: FiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        let node_count = space.element_node_count(element_index);
        self.basis_buffer
            .resize(node_count, Space::ReferenceDim::dim());
        self.basis_buffer
            .populate_element_nodes_from_space(element_index, space);
        self.u_local
            .resize_vertically_mut(solution_dim * node_count, T::zero());
        gather_global_to_local(
            DVectorSlice::from(u_global.into()),
            &mut self.u_local,
            self.basis_buffer.element_nodes(),
            solution_dim,
        );

        InterpolationElementBuffer {
            basis_buffer: &mut self.basis_buffer,
            u_local: DVectorSlice::from(&self.u_local),
            space,
            reference_point: OPoint::origin(),
            element_index,
        }
    }
}

impl<'a, T, Space> InterpolationElementBuffer<'a, T, Space>
where
    T: Real,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
{
    pub fn update_reference_point(&mut self, reference_point: &OPoint<T, Space::ReferenceDim>) {
        self.basis_buffer
            .populate_element_basis_values_from_space(self.element_index, self.space, reference_point);
        self.reference_point = reference_point.clone();
    }

    pub fn map_reference_coords(&self) -> OPoint<T, Space::GeometryDim> {
        self.space
            .map_element_reference_coords(self.element_index, &self.reference_point)
    }

    pub fn element_reference_jacobian(&self) -> OMatrix<T, Space::GeometryDim, Space::ReferenceDim> {
        self.space
            .element_reference_jacobian(self.element_index, &self.reference_point)
    }

    pub fn interpolate<S>(&self) -> OVector<T, S>
    where
        S: SmallDim,
        DefaultAllocator: DimAllocator<T, S>,
    {
        compute_interpolation(self.u_local, self.basis_buffer.element_basis_values())
    }

    pub fn basis_values(&self) -> &[T] {
        self.basis_buffer.element_basis_values()
    }
}

//! New functionality that will eventually supercede various methods in the `assembly` module.

use crate::allocators::{BiDimAllocator, SmallDimAllocator, TriDimAllocator};
use crate::assembly::local::{
    compute_volume_u_grad, ElementConnectivityAssembler, ElementMatrixAssembler,
};
use crate::element::{MatrixSlice, MatrixSliceMut};
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{
    DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimName, Dynamic,
    MatrixMN, MatrixSliceMN, Point, RealField, Scalar, VectorN, U1,
};
use crate::space::{FiniteElementSpace2, VolumetricFiniteElementSpace};
use crate::workspace::Workspace;
use crate::SmallDim;
use itertools::izip;
use nalgebra::{DMatrix, MatrixSliceMutMN};
use std::cell::{RefCell, RefMut};
use std::error::Error;
use std::marker::PhantomData;
use std::ops::AddAssign;

pub trait Operator<T> {
    type SolutionDim: SmallDim;

    /// The data associated with the operator.
    ///
    /// Typically this encodes material information, such as density, stiffness and other physical
    /// quantities. This is intended to be paired with data associated with individual
    /// quadrature points during numerical integration.
    type Data: Default + Clone + 'static;
}

pub trait EllipticOperator<T, GeometryDim>: Operator<T>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim, Self::SolutionDim>,
{
    /// TODO: Find better name
    fn compute_elliptic_term(
        &self,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        data: &Self::Data,
    ) -> MatrixMN<T, GeometryDim, Self::SolutionDim>;
}

pub trait EllipticContraction<T, GeometryDim>: EllipticOperator<T, GeometryDim>
where
    T: RealField,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    fn contract(
        &self,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        data: &Self::Data,
        a: &VectorN<T, GeometryDim>,
        b: &VectorN<T, GeometryDim>,
    ) -> MatrixMN<T, Self::SolutionDim, Self::SolutionDim>;

    /// Compute multiple contractions and store the result in the provided matrix.
    ///
    /// The matrix `a` is a `GeometryDim x NodalDim` sized matrix, in which each column
    /// corresponds to a vector of dimension `GeometryDim`. The output matrix is a square matrix
    /// with row and col dimensions `SolutionDim * NodalDim`, consisting of `NodalDim x NodalDim`
    /// block matrices, each with dimension `SolutionDim x SolutionDim`.
    ///
    /// Let c(gradient, a, b) denote the contraction of vectors a and b.
    /// Then the result of c(gradient, a_I, a_J) for each I, J in the range `(0 .. NodalDim)`
    /// must be *added* to `output_IJ`, where `output_IJ` is the `SolutionDim x SolutionDim`
    /// block matrix corresponding to nodes `I` and `J`.
    fn contract_multiple_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        data: &Self::Data,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        a: &MatrixSliceMN<T, GeometryDim, Dynamic>,
    ) {
        let num_nodes = a.ncols();
        let output_dim = num_nodes * Self::SolutionDim::dim();
        assert_eq!(output_dim, output.nrows());
        assert_eq!(output_dim, output.ncols());

        let sdim = Self::SolutionDim::dim();
        for i in 0..num_nodes {
            for j in i..num_nodes {
                let a_i = a.fixed_slice::<GeometryDim, U1>(0, i).clone_owned();
                let a_j = a.fixed_slice::<GeometryDim, U1>(0, j).clone_owned();
                let contraction = self.contract(gradient, data, &a_i, &a_j);
                output
                    .fixed_slice_mut::<Self::SolutionDim, Self::SolutionDim>(i * sdim, j * sdim)
                    .add_assign(&contraction);

                // TODO: We currently assume symmetry. Should maybe have a method that
                // says whether it is symmetric or not?
                if i != j {
                    output
                        .fixed_slice_mut::<Self::SolutionDim, Self::SolutionDim>(j * sdim, i * sdim)
                        .add_assign(&contraction.transpose());
                }
            }
        }
    }
}

pub trait ElementVectorAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        output: DVectorSliceMut<T>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}

#[derive(Debug)]
pub struct ElementEllipticAssembler<'a, T: Scalar, Space, Op, QTable> {
    // TODO: Create builder?
    pub space: &'a Space,
    pub op: &'a Op,
    pub qtable: &'a QTable,
    pub u: DVectorSlice<'a, T>,
}

impl<'a, T: Scalar> ElementEllipticAssembler<'static, T, (), (), ()> {
    pub fn new() -> Self {
        Self {
            space: &(),
            op: &(),
            qtable: &(),
            u: DVectorSlice::from_slice(&[], 0),
        }
    }
}

impl<'a, T, Space, Op, QTable> ElementEllipticAssembler<'a, T, Space, Op, QTable> where T: Scalar {}

impl<'a, T, Space, Op, QTable> ElementConnectivityAssembler
    for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: Scalar,
    Space: VolumetricFiniteElementSpace<T>,
    Op: Operator<T>,
    DefaultAllocator: SmallDimAllocator<T, Space::GeometryDim>,
{
    fn solution_dim(&self) -> usize {
        Op::SolutionDim::dim()
    }

    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(output, element_index)
    }
}

#[derive(Debug)]
struct EllipticAssemblerWorkspace<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    phi_grad_ref: DMatrix<T>,
    u_element: DVector<T>,
    element_nodes: Vec<usize>,
    quad_weights: Vec<T>,
    quad_points: Vec<Point<T, GeometryDim>>,
    quad_data: Vec<Data>,
}

impl<T, GeometryDim, Data> EllipticAssemblerWorkspace<T, GeometryDim, Data>
where
    T: RealField,
    GeometryDim: DimName,
    Data: Default + Clone,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn resize_workspace(&mut self, quadrature_size: usize, solution_dim: usize, node_count: usize) {
        self.element_nodes.resize(node_count, usize::MAX);
        self.u_element
            .resize_vertically_mut(solution_dim * node_count, T::zero());
        self.phi_grad_ref
            .resize_mut(GeometryDim::dim(), node_count, T::zero());
        self.quad_points.resize(quadrature_size, Point::origin());
        self.quad_weights.resize(quadrature_size, T::zero());
        self.quad_data.resize(quadrature_size, Data::default());
    }
}

impl<T, GeometryDim, Data> Default for EllipticAssemblerWorkspace<T, GeometryDim, Data>
where
    T: RealField,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn default() -> Self {
        Self {
            phi_grad_ref: DMatrix::zeros(0, 0),
            u_element: DVector::zeros(0),
            element_nodes: Vec::default(),
            quad_weights: Vec::default(),
            quad_points: Vec::default(),
            quad_data: Vec::default(),
        }
    }
}

impl<'a, T: Scalar, S, O, Q> ElementEllipticAssembler<'a, T, S, O, Q> {
    thread_local! { static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default());  }
}

struct ForEachQuadraturePoint<'a, T, GeometryDim, SolutionDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    SolutionDim: DimName,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    u_grad: &'a MatrixMN<T, GeometryDim, SolutionDim>,
    phi_grad_ref: MatrixSliceMut<'a, T, GeometryDim, Dynamic>,
    jacobian_inv_t: &'a MatrixMN<T, GeometryDim, GeometryDim>,
    jacobian_det: T,
    weight: T,
    data: &'a Data,
}

impl<'a, T: Scalar, Space, Op, QTable> ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Op: Operator<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Data>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Op::SolutionDim>,
{
    /// Calls the given function with a mutable reference to a thread-local workspace.
    ///
    /// Helper method that abstracts away the boilerplate of obtaining the workspace.
    fn with_workspace<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut EllipticAssemblerWorkspace<T, Space::GeometryDim, Op::Data>) -> R,
    {
        Self::WORKSPACE.with(|workspace| {
            // First get through the RefCell
            let mut ws = workspace.borrow_mut();
            // Then get the concrete type from the type-erased workspace
            let ws =
                ws.get_or_default::<EllipticAssemblerWorkspace<T, Space::GeometryDim, Op::Data>>();
            f(ws)
        })
    }

    /// Populates element quantities into workspace buffers and calls the provided function
    /// per quadrature point.
    #[allow(non_snake_case)]
    fn for_each_quadrature_point<F>(
        &self,
        element_index: usize,
        mut f: F,
    ) -> Result<(), Box<dyn Error + Send + Sync>>
    where
        F: FnMut(
            ForEachQuadraturePoint<T, Space::GeometryDim, Op::SolutionDim, Op::Data>,
        ) -> Result<(), Box<dyn Error + Send + Sync>>,
    {
        self.with_workspace(|ws| {
            let quadrature_size = self.qtable.element_quadrature_size(element_index);
            let s = Op::SolutionDim::dim();
            let n = self.element_node_count(element_index);
            ws.resize_workspace(quadrature_size, s, n);

            self.space
                .populate_element_nodes(&mut ws.element_nodes, element_index);
            gather_global_to_local(&self.u, &mut ws.u_element, &ws.element_nodes, s);

            // Reshape u and the output vector into matrices of dimensions s x n,
            // where s is the solution dim and n is the number of nodes
            // (each column corresponds to the vector term associated with the corresponding node)
            let s = Op::SolutionDim::name();
            let n = Dynamic::new(n);
            let u_element = MatrixSliceMN::from_slice_generic(ws.u_element.as_slice(), s, n);

            self.qtable.populate_element_quadrature_and_data(
                element_index,
                &mut ws.quad_points,
                &mut ws.quad_weights,
                &mut ws.quad_data,
            );

            let iter = izip!(&ws.quad_weights, &ws.quad_points, &ws.quad_data);
            for (&w, xi, data) in iter {
                // Compute gradients with respect to reference coords
                let phi_grad_ref = &mut ws.phi_grad_ref;
                self.space.populate_element_gradients(
                    element_index,
                    MatrixSliceMut::from(&mut *phi_grad_ref),
                    &xi,
                );

                // Jacobian
                let J = self.space.element_reference_jacobian(element_index, &xi);
                let J_det = J.determinant();
                let J_inv = J.try_inverse().ok_or_else(|| {
                    Box::<dyn Error + Send + Sync>::from("Singular element Jacobian encountered")
                })?;
                let J_inv_t = J_inv.transpose();

                f(ForEachQuadraturePoint {
                    u_grad: &compute_volume_u_grad(&J_inv_t, &*phi_grad_ref, u_element),
                    phi_grad_ref: MatrixSliceMut::from(&mut *phi_grad_ref),
                    jacobian_inv_t: &J_inv_t,
                    jacobian_det: J_det,
                    weight: w,
                    data,
                })?;
            }

            Ok(())
        })
    }
}

impl<'a, T, Space, Op, QTable> ElementVectorAssembler<T>
    for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Op: EllipticOperator<T, Space::GeometryDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Data>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Op::SolutionDim>,
{
    #[allow(non_snake_case)]
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        mut output: DVectorSliceMut<T>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let s = self.solution_dim();
        let n = self.element_node_count(element_index);
        assert_eq!(output.len(), s * n, "Output vector dimension mismatch");
        self.for_each_quadrature_point(element_index, |for_each_point_data| {
            let ForEachQuadraturePoint {
                u_grad,
                phi_grad_ref,
                jacobian_inv_t,
                jacobian_det,
                weight,
                data,
            } = for_each_point_data;

            // TODO: Document what's going on here
            let g = self.op.compute_elliptic_term(&u_grad, data);
            let g_J_inv_t = g.transpose() * jacobian_inv_t;
            output.gemm(
                weight * jacobian_det.abs(),
                &g_J_inv_t,
                &phi_grad_ref,
                T::one(),
            );
            Ok(())
        })
    }
}

impl<'a, T, Space, Op, QTable> ElementMatrixAssembler<T>
    for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Op: EllipticContraction<T, Space::GeometryDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Data>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Op::SolutionDim>,
{
    // TODO: Reorder method parameters (element_index first)
    #[allow(non_snake_case)]
    fn assemble_element_matrix_into(
        &self,
        mut output: DMatrixSliceMut<T, U1, Dynamic>,
        element_index: usize,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let s = self.solution_dim();
        let n = self.element_node_count(element_index);
        assert_eq!(output.nrows(), s * n, "Output matrix dimension mismatch");
        assert_eq!(output.ncols(), s * n, "Output matrix dimension mismatch");

        self.for_each_quadrature_point(element_index, |for_each_point_data| {
            let ForEachQuadraturePoint {
                u_grad,
                mut phi_grad_ref,
                jacobian_inv_t,
                jacobian_det,
                weight,
                data,
            } = for_each_point_data;

            // TODO: Clean this up a bit
            // Compute gradients with respect to physical coords instead of reference coords
            for mut phi_grad in phi_grad_ref.column_iter_mut() {
                let new_phi_grad = jacobian_inv_t * &phi_grad;
                phi_grad.copy_from(&new_phi_grad);
            }
            let phi_grad = phi_grad_ref;

            // TODO: Scale during the loop up above
            // TODO: Or maybe we should extend the contraction trait to allow a scaling factor
            let scale = weight * jacobian_det.abs();
            // We need to multiply the contraction result by the scale factor.
            // We do this implicitly by multiplying the basis gradients by its square root.
            // This way we don't have to allocate an additional matrix or complicate
            // the trait.
            let mut G = phi_grad;
            G *= scale.sqrt();

            self.op
                .contract_multiple_into(&mut output, data, &u_grad, &MatrixSlice::from(&G));
            Ok(())
        })
    }
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

// TODO: Maybe move to some other module?
pub fn gather_global_to_local<'a, T: Scalar>(
    global: impl Into<DVectorSlice<'a, T>>,
    local: impl Into<DVectorSliceMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    gather_global_to_local_(global.into(), local.into(), indices, solution_dim)
}

fn gather_global_to_local_<T: Scalar>(
    global: DVectorSlice<T>,
    mut local: DVectorSliceMut<T>,
    indices: &[usize],
    solution_dim: usize,
) {
    assert_eq!(
        local.len(),
        indices.len() * solution_dim,
        "Size of local vector must be compatible with solutio mdim and index count"
    );
    let s = solution_dim;
    for (i_local, i_global) in indices.iter().enumerate() {
        local
            .index_mut((s * i_local..s * i_local + s, 0))
            .copy_from(&global.index((s * i_global..s * i_global + s, 0)));
    }
}

pub fn add_local_to_global<'a, T: RealField>(
    local: impl Into<DVectorSlice<'a, T>>,
    global: impl Into<DVectorSliceMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    add_local_to_global_(local.into(), global.into(), indices, solution_dim)
}

pub fn add_local_to_global_<'a, T: RealField>(
    local: DVectorSlice<'a, T>,
    mut global: DVectorSliceMut<'a, T>,
    indices: &[usize],
    solution_dim: usize,
) {
    assert_eq!(
        local.len(),
        indices.len() * solution_dim,
        "Size of local vector must be compatible with solution dim and index count"
    );
    let s = solution_dim;
    for (i_local, i_global) in indices.iter().enumerate() {
        global
            .index_mut((s * i_global..s * i_global + s, 0))
            .add_assign(&local.index((s * i_local..s * i_local + s, 0)));
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

/// A buffer for storing intermediate quadrature data.
#[derive(Debug)]
pub struct QuadratureBuffer<T, D, Data>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    quad_weights: Vec<T>,
    quad_points: Vec<Point<T, D>>,
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
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    /// Resizes the internal buffer storages to the given size.
    pub fn resize(&mut self, quadrature_size: usize) {
        self.quad_points.resize(quadrature_size, Point::origin());
        self.quad_weights.resize(quadrature_size, T::zero());
        self.quad_data.resize(quadrature_size, Data::default());
    }

    /// Populates the buffer by querying a quadrature table with the given element index.
    pub fn populate_element_quadrature_from_table(
        &mut self,
        element_index: usize,
        table: &dyn QuadratureTable<T, GeometryDim, Data = Data>,
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

    /// Calls a closure for each quadrature point currently in the workspace.
    pub fn for_each_quadrature_point<F>(&self, mut f: F) -> Result<(), Box<dyn Error + Sync + Send>>
    where
        F: FnMut(T, &Point<T, GeometryDim>, &Data) -> Result<(), Box<dyn Error + Sync + Send>>,
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

#[derive(Debug)]
pub struct BasisFunctionBuffer<T: Scalar> {
    element_nodes: Vec<usize>,
    element_basis_values: DMatrix<T>,
    element_basis_gradients: DMatrix<T>,
}

impl<T: RealField> Default for BasisFunctionBuffer<T> {
    fn default() -> Self {
        Self {
            element_nodes: Vec::new(),
            element_basis_values: DMatrix::zeros(0, 0),
            element_basis_gradients: DMatrix::zeros(0, 0),
        }
    }
}

impl<T: RealField> BasisFunctionBuffer<T> {
    pub fn resize(&mut self, node_count: usize, reference_dim: usize) {
        self.element_nodes.resize(node_count, usize::MAX);
        self.element_basis_values
            .resize_mut(1, node_count, T::zero());
        self.element_basis_gradients
            .resize_mut(reference_dim, node_count, T::zero());
    }

    pub fn populate_element_nodes_from_space<Space>(&mut self, element_index: usize, space: &Space)
    where
        Space: FiniteElementSpace2<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        let node_count = space.element_node_count(element_index);
        self.resize(node_count, Space::ReferenceDim::dim());
        space.populate_element_nodes(&mut self.element_nodes, element_index);
    }

    /// TODO: Document that populate_element_nodes should be called first
    pub fn populate_element_basis_values_from_space<Space>(
        &mut self,
        element_index: usize,
        space: &Space,
        reference_coords: &Point<T, Space::ReferenceDim>,
    ) where
        Space: FiniteElementSpace2<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        space.populate_element_basis(
            element_index,
            MatrixSliceMut::from(&mut self.element_basis_values),
            reference_coords,
        );
    }

    pub fn populate_element_basis_gradients_from_space<Space>(
        &mut self,
        element_index: usize,
        space: &Space,
        reference_coords: &Point<T, Space::ReferenceDim>,
    ) where
        Space: FiniteElementSpace2<T>,
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

    pub fn element_basis_values<D: DimName>(&self) -> MatrixSlice<T, U1, Dynamic> {
        MatrixSlice::from(&self.element_basis_values)
    }

    pub fn element_basis_values_mut<D: DimName>(&mut self) -> MatrixSliceMut<T, U1, Dynamic> {
        MatrixSliceMut::from(&mut self.element_basis_values)
    }

    pub fn element_gradients_mut<D: DimName>(&mut self) -> MatrixSliceMut<T, D, Dynamic> {
        MatrixSliceMut::from(&mut self.element_basis_gradients)
    }
}

pub trait SourceFunction<T, GeometryDim>: Operator<T>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    fn evaluate(
        &self,
        coords: &Point<T, GeometryDim>,
        data: &Self::Data,
    ) -> VectorN<T, Self::SolutionDim>;
}

pub struct ElementSourceAssembler<'a, T, Space, Source, QTable> {
    // TODO: Create builder API instead of having pub fields
    pub space: &'a Space,
    pub qtable: &'a QTable,
    pub source: &'a Source,
    pub marker: PhantomData<T>,
}

impl<'a, T, Space, Source, QTable> ElementConnectivityAssembler
    for ElementSourceAssembler<'a, T, Space, Source, QTable>
where
    T: Scalar,
    Space: FiniteElementSpace2<T>,
    Source: SourceFunction<T, Space::GeometryDim>,
    DefaultAllocator:
        TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, Source::SolutionDim>,
{
    fn solution_dim(&self) -> usize {
        Source::SolutionDim::dim()
    }

    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(output, element_index)
    }
}

thread_local! { static SOURCE_WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default()) }

struct SourceTermWorkspace<T, D, Data>
where
    T: Scalar,
    D: SmallDim,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    quadrature_buffer: QuadratureBuffer<T, D, Data>,
    basis_buffer: BasisFunctionBuffer<T>,
}

impl<T, D, Data> Default for SourceTermWorkspace<T, D, Data>
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    fn default() -> Self {
        Self {
            quadrature_buffer: QuadratureBuffer::default(),
            basis_buffer: BasisFunctionBuffer::default(),
        }
    }
}

impl<'a, T, Space, Source, QTable> ElementVectorAssembler<T>
    for ElementSourceAssembler<'a, T, Space, Source, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Source: SourceFunction<T, Space::GeometryDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Source::Data>,
    DefaultAllocator:
        TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, Source::SolutionDim>,
{
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        mut output: DVectorSliceMut<T>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        SOURCE_WORKSPACE.with(|ws| {
            // TODO: Is it possible to simplify retrieving a mutable reference to the workspace?
            let mut ws: RefMut<SourceTermWorkspace<T, Space::ReferenceDim, Source::Data>> =
                RefMut::map(ws.borrow_mut(), |ws| ws.get_or_default());
            let ws: &mut SourceTermWorkspace<_, _, _> = &mut *ws;
            let basis_buffer = &mut ws.basis_buffer;
            let quad_buffer = &mut ws.quadrature_buffer;

            // TODO: Should output be cleared or not?
            basis_buffer.populate_element_nodes_from_space(element_index, self.space);

            // TODO: Use QuadratureBuffer in the elliptic assembler trait impls too
            quad_buffer.populate_element_quadrature_from_table(element_index, self.qtable);

            // Reshape output into an `s x n` matrix, so that each column corresponds to the
            // output associated with a node
            let n = basis_buffer.element_nodes().len();
            assert_eq!(
                output.len(),
                n * Source::SolutionDim::dim(),
                "Length of output vector must be consistent with number of nodes and solution dim"
            );
            let mut output = MatrixSliceMutMN::from_slice_generic(
                output.as_mut_slice(),
                Source::SolutionDim::name(),
                Dynamic::new(n),
            );

            quad_buffer.for_each_quadrature_point(|w, xi, data| {
                basis_buffer.populate_element_basis_values_from_space(
                    element_index,
                    self.space,
                    xi,
                );

                let x = self.space.map_element_reference_coords(element_index, xi);
                let j = self.space.element_reference_jacobian(element_index, xi);
                let f = self.source.evaluate(&x, data);

                // The output contribution for quadrature point q is
                //  w * |det J| * [ f_1 f_2 f_3, ... ]
                // where f_I = f * phi_I is the output associated with node I, and phi_I is the
                // basis values of node I.
                // Then the contribution is given by
                //  w * |det J| * [ f * phi_1, f * phi_2, ... ] = w * |det J| * f * phi,
                // where phi is a row vector of basis values
                let phi = basis_buffer.element_basis_values::<Space::ReferenceDim>();
                output.gemm(w * j.determinant().abs(), &f, &phi, T::one());

                Ok(())
            })?;

            Ok(())
        })
    }
}

#[derive(Debug)]
struct SerialVectorAssemblerWorkspace<T: Scalar> {
    vector: DVector<T>,
    nodes: Vec<usize>,
}

impl<T: RealField> Default for SerialVectorAssemblerWorkspace<T> {
    fn default() -> Self {
        Self {
            vector: DVector::zeros(0),
            nodes: vec![],
        }
    }
}

#[derive(Debug)]
pub struct SerialVectorAssembler<T: Scalar> {
    workspace: RefCell<SerialVectorAssemblerWorkspace<T>>,
}

impl<T: RealField> Default for SerialVectorAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: RefCell::new(SerialVectorAssemblerWorkspace::default()),
        }
    }
}

impl<T: RealField> SerialVectorAssembler<T> {
    pub fn assemble_vector_into<'a>(
        &self,
        output: impl Into<DVectorSliceMut<'a, T>>,
        element_assembler: &dyn ElementVectorAssembler<T>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // TODO: Move impl into _ method to remove the impl Into<> compilation overhead
        let mut output = output.into();
        let num_elements = element_assembler.num_elements();
        let n = element_assembler.num_nodes();
        let s = element_assembler.solution_dim();
        assert_eq!(output.len(), s * n, "Output dimensions mismatch");

        let mut workspace = self.workspace.borrow_mut();

        for i in 0..num_elements {
            let element_node_count = element_assembler.element_node_count(i);
            workspace.nodes.resize(element_node_count, usize::MAX);
            workspace
                .vector
                .resize_vertically_mut(s * element_node_count, T::zero());
            element_assembler.populate_element_nodes(&mut workspace.nodes, i);
            element_assembler.assemble_element_vector_into(i, (&mut workspace.vector).into())?;
            add_local_to_global(&workspace.vector, &mut output, &workspace.nodes, s);
        }

        Ok(())
    }

    pub fn assemble_vector(
        &self,
        element_assembler: &dyn ElementVectorAssembler<T>,
    ) -> Result<DVector<T>, Box<dyn Error + Send + Sync>> {
        let n = element_assembler.num_nodes();
        let mut result = DVector::zeros(element_assembler.solution_dim() * n);
        self.assemble_vector_into(&mut result, element_assembler)?;
        Ok(result)
    }
}

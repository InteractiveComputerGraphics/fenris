use std::cell::{RefCell, RefMut};
use std::ops::AddAssign;

use eyre::eyre;
use itertools::izip;
use nalgebra::base::allocator::Allocator;
use nalgebra::{
    DMatrix, DMatrixSliceMut, DefaultAllocator, DimMin, DimName, Dynamic, MatrixMN, MatrixSliceMN,
    RealField, Scalar, VectorN, U1,
};

use crate::allocators::{
    BiDimAllocator, FiniteElementMatrixAllocator, SmallDimAllocator, TriDimAllocator,
    VolumeFiniteElementAllocator,
};
use crate::assembly::global;
use crate::assembly::global::{BasisFunctionBuffer, QuadratureBuffer};
use crate::connectivity::Connectivity;
use crate::element::{MatrixSlice, MatrixSliceMut, VolumetricFiniteElement};
use crate::mesh::Mesh;
use crate::nalgebra::{DVector, DVectorSlice, DVectorSliceMut, MatrixSliceMutMN, Point};
use crate::quadrature::Quadrature;
use crate::space::{FiniteElementConnectivity, VolumetricFiniteElementSpace};
use crate::workspace::Workspace;
use crate::SmallDim;

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

/// Computes the integral of a scalar function f(X, u, grad u) over an element.
#[allow(non_snake_case)]
pub fn compute_element_integral<T, SolutionDim, Element, F>(
    element: &Element,
    u_element: &MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: &impl Quadrature<T, Element::GeometryDim>,
    function: F,
) -> T
where
    T: RealField,
    Element: VolumetricFiniteElement<T>,
    Element::GeometryDim: DimName + DimMin<Element::GeometryDim, Output = Element::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: VolumeFiniteElementAllocator<T, Element::GeometryDim>
        + Allocator<T, Element::GeometryDim, SolutionDim>
        + Allocator<T, SolutionDim, Element::GeometryDim>
        + Allocator<T, SolutionDim, U1>,
    F: Fn(
        &VectorN<T, Element::GeometryDim>,
        &VectorN<T, SolutionDim>,
        &MatrixMN<T, Element::GeometryDim, SolutionDim>,
    ) -> T,
{
    let mut f_e = T::zero();

    let weights = quadrature.weights();
    let points = quadrature.points();

    // TODO: Avoid allocation!
    let mut basis_values = MatrixMN::<_, U1, Dynamic>::zeros(element.num_nodes());
    let mut gradients = MatrixMN::<_, Element::GeometryDim, Dynamic>::zeros(element.num_nodes());

    for (&w, xi) in weights.iter().zip(points) {
        element.populate_basis(MatrixSliceMut::from(&mut basis_values), xi);
        element.populate_basis_gradients(MatrixSliceMut::from(&mut gradients), xi);

        // Jacobian
        let J = element.reference_jacobian(xi);

        let J_det = J.determinant();
        let J_inv = J.try_inverse().expect("Jacobian must be invertible");

        let X = element.map_reference_coords(xi);
        let u = mat_mul_mat_transpose(u_element, &basis_values);
        let u_grad = compute_volume_u_grad(
            &J_inv.transpose(),
            MatrixSlice::from(&gradients),
            MatrixSlice::from(u_element),
        );
        let f = function(&X.coords, &u, &u_grad);
        f_e += f * w * J_det.abs();
    }

    f_e
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
    let mut basis_values = DMatrix::zeros(1, element.num_nodes());

    let weights = quadrature.weights();
    let points = quadrature.points();

    let num_nodes = element.num_nodes();
    let sol_dim = SolutionDim::dim();

    for (&w, xi) in weights.iter().zip(points) {
        let J = element.reference_jacobian(xi);
        let J_det = J.determinant();

        element.populate_basis(MatrixSliceMut::from(&mut basis_values), xi);
        let phi = MatrixSlice::<_, U1, Dynamic>::from(&basis_values);

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

// TODO: Move this to the right spot and don't make it pub(crate)
#[allow(non_snake_case)]
pub(crate) fn compute_volume_u_grad<'a, T, GeometryDim, SolutionDim>(
    jacobian_inv_t: &MatrixMN<T, GeometryDim, GeometryDim>,
    phi_grad_ref: impl Into<MatrixSlice<'a, T, GeometryDim, Dynamic>>,
    u: impl Into<MatrixSlice<'a, T, SolutionDim, Dynamic>>,
) -> MatrixMN<T, GeometryDim, SolutionDim>
where
    T: RealField,
    SolutionDim: DimName,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<T, GeometryDim, SolutionDim>
        + Allocator<T, SolutionDim, GeometryDim>
        + Allocator<T, GeometryDim, GeometryDim>,
{
    let phi_grad_ref = phi_grad_ref.into();
    let u = u.into();
    // We have that grad u = sum_I grad phi_I u_I^T,
    // which can alternatively be written
    //  grad u = G U^T,
    // where G = [ grad phi_1, grad phi_2, ... ] and U = [ u_1, u_2, ... ]
    // are matrices containing the (physical domain) gradients and per-node solution variables u_I.
    // We note that `grad phi_I = inv(J)^T * grad phi_ref_I`, where `grad phi_ref_I` is the
    // gradient of basis function `phi_I` with respect to the reference coordinates.
    // Unfortunately, we cannot easily directly compute G U^T since there is only
    // A^T B through gemm_tr available in `nalgebra`. Therefore we instead compute
    // the sum of outer products
    let G_ref = phi_grad_ref;
    let mut u_grad = MatrixMN::<_, GeometryDim, SolutionDim>::zeros();
    for (phi_I_grad_ref, u_I) in G_ref.column_iter().zip(u.column_iter()) {
        // Instead of computing each column of G (gradients in physical space),
        // we instead use the reference coordinates and only multiply by J^{-T} for the end result,
        // which involves substantially less work
        // TODO: Rewrite the earlier explanation to clarify this
        // (i.e. that we compute J^{-T} * G_ref * U^T insetad of G U^T)
        u_grad.ger(T::one(), &phi_I_grad_ref, &u_I, T::one());
    }
    jacobian_inv_t * u_grad
}

/// Computes
fn mat_mul_mat_transpose<'a, T, R, C>(
    a: impl Into<MatrixSlice<'a, T, R, Dynamic>>,
    b: impl Into<MatrixSlice<'a, T, C, Dynamic>>,
) -> MatrixMN<T, R, C>
where
    T: RealField,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    mat_mul_mat_transpose_(a.into(), b.into())
}

/// Computes the matrix product `A * B^T` where `A` and `B` have fixed row counts, but
/// dynamic column counts.
fn mat_mul_mat_transpose_<'a, T, R, C>(
    a: MatrixSlice<'a, T, R, Dynamic>,
    b: MatrixSlice<'a, T, C, Dynamic>,
) -> MatrixMN<T, R, C>
where
    T: RealField,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>,
{
    assert_eq!(a.nrows(), b.nrows());
    let mut result = MatrixMN::<T, R, C>::zeros();

    // Compute A B^T = sum_k a_k * b_k^T   (outer product)
    // where a_k and b_k represent column k in A and B, respectively
    for (a_k, b_k) in a.column_iter().zip(b.column_iter()) {
        result.ger(T::one(), &a_k, &b_k, T::one());
    }
    result
}

pub trait Operator {
    type SolutionDim: SmallDim;

    /// The parameters associated with the operator.
    ///
    /// Typically this encodes material information, such as density, stiffness and other physical
    /// quantities. This is intended to be paired with data associated with individual
    /// quadrature points during numerical integration.
    type Parameters: Default + Clone + 'static;
}

pub trait EllipticOperator<T, GeometryDim>: Operator
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim, Self::SolutionDim>,
{
    /// TODO: Find better name
    fn compute_elliptic_term(
        &self,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        data: &Self::Parameters,
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
        data: &Self::Parameters,
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
        data: &Self::Parameters,
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
    ) -> eyre::Result<()>;
}

/// TODO: The builder here is pretty complex. Is it possible to simplify without losing too
/// much type safety?
pub struct ElementEllipticAssemblerBuilder<Space, Op, QTable, U> {
    space: Space,
    op: Op,
    qtable: QTable,
    u: U,
}

impl ElementEllipticAssemblerBuilder<(), (), (), ()> {
    pub fn new() -> Self {
        Self {
            space: (),
            op: (),
            qtable: (),
            u: (),
        }
    }
}

impl<Op, QTable, U> ElementEllipticAssemblerBuilder<(), Op, QTable, U> {
    pub fn with_finite_element_space<Space>(
        self,
        space: &Space,
    ) -> ElementEllipticAssemblerBuilder<&Space, Op, QTable, U> {
        ElementEllipticAssemblerBuilder {
            space,
            op: self.op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

impl<Space, QTable, U> ElementEllipticAssemblerBuilder<Space, (), QTable, U> {
    pub fn with_operator<Op>(
        self,
        op: &Op,
    ) -> ElementEllipticAssemblerBuilder<Space, &Op, QTable, U> {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

impl<Space, Op, U> ElementEllipticAssemblerBuilder<Space, Op, (), U> {
    pub fn with_quadrature_table<QTable>(
        self,
        qtable: &QTable,
    ) -> ElementEllipticAssemblerBuilder<Space, Op, &QTable, U> {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op: self.op,
            qtable,
            u: self.u,
        }
    }
}

impl<Space, Op, QTable> ElementEllipticAssemblerBuilder<Space, Op, QTable, ()> {
    pub fn with_u<'a, T>(
        self,
        u: impl Into<DVectorSlice<'a, T>>,
    ) -> ElementEllipticAssemblerBuilder<Space, Op, QTable, DVectorSlice<'a, T>>
    where
        T: Scalar,
    {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op: self.op,
            qtable: self.qtable,
            u: u.into(),
        }
    }
}

impl<'a, T, Space, Op, QTable>
    ElementEllipticAssemblerBuilder<&'a Space, &'a Op, &'a QTable, DVectorSlice<'a, T>>
where
    T: Scalar,
{
    pub fn build(self) -> ElementEllipticAssembler<'a, T, Space, Op, QTable> {
        ElementEllipticAssembler {
            space: self.space,
            op: self.op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

#[derive(Debug)]
pub struct ElementEllipticAssembler<'a, T: Scalar, Space, Op, QTable> {
    space: &'a Space,
    op: &'a Op,
    qtable: &'a QTable,
    u: DVectorSlice<'a, T>,
}

impl<'a, T, Space, Op, QTable> ElementConnectivityAssembler
    for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: Scalar,
    Space: VolumetricFiniteElementSpace<T>,
    Op: Operator,
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
    Op: Operator,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Parameters>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Op::SolutionDim>,
{
    /// Calls the given function with a mutable reference to a thread-local workspace.
    ///
    /// Helper method that abstracts away the boilerplate of obtaining the workspace.
    fn with_workspace<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut EllipticAssemblerWorkspace<T, Space::GeometryDim, Op::Parameters>) -> R,
    {
        Self::WORKSPACE.with(|workspace| {
            // First get through the RefCell
            let mut ws = workspace.borrow_mut();
            // Then get the concrete type from the type-erased workspace
            let ws =
                ws.get_or_default::<EllipticAssemblerWorkspace<T, Space::GeometryDim, Op::Parameters>>();
            f(ws)
        })
    }

    /// Populates element quantities into workspace buffers and calls the provided function
    /// per quadrature point.
    #[allow(non_snake_case)]
    fn for_each_quadrature_point<F>(&self, element_index: usize, mut f: F) -> eyre::Result<()>
    where
        F: FnMut(
            ForEachQuadraturePoint<T, Space::GeometryDim, Op::SolutionDim, Op::Parameters>,
        ) -> eyre::Result<()>,
    {
        self.with_workspace(|ws| {
            let quadrature_size = self.qtable.element_quadrature_size(element_index);
            let s = Op::SolutionDim::dim();
            let n = self.element_node_count(element_index);
            ws.resize_workspace(quadrature_size, s, n);

            self.space
                .populate_element_nodes(&mut ws.element_nodes, element_index);
            global::gather_global_to_local(&self.u, &mut ws.u_element, &ws.element_nodes, s);

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
                let J_inv = J
                    .try_inverse()
                    .ok_or_else(|| eyre!("Singular element Jacobian encountered"))?;
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
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Parameters>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Op::SolutionDim>,
{
    #[allow(non_snake_case)]
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        mut output: DVectorSliceMut<T>,
    ) -> eyre::Result<()> {
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
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Parameters>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Op::SolutionDim>,
{
    #[allow(non_snake_case)]
    fn assemble_element_matrix_into(
        &self,
        element_index: usize,
        mut output: DMatrixSliceMut<T, U1, Dynamic>,
    ) -> eyre::Result<()> {
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

    fn as_connectivity_assembler(&self) -> &dyn ElementConnectivityAssembler {
        self
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

pub trait SourceFunction<T, GeometryDim>: Operator
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    fn evaluate(
        &self,
        coords: &Point<T, GeometryDim>,
        data: &Self::Parameters,
    ) -> VectorN<T, Self::SolutionDim>;
}

pub struct ElementSourceAssemblerBuilder<SpaceRef, SourceRef, QTableRef> {
    space: SpaceRef,
    source: SourceRef,
    qtable: QTableRef,
}

impl ElementSourceAssemblerBuilder<(), (), ()> {
    pub fn new() -> Self {
        Self {
            space: (),
            source: (),
            qtable: (),
        }
    }
}

impl<SpaceRef, SourceRef, QTableRef> ElementSourceAssemblerBuilder<SpaceRef, SourceRef, QTableRef> {
    pub fn with_finite_element_space<Space>(
        self,
        space: &Space,
    ) -> ElementSourceAssemblerBuilder<&Space, SourceRef, QTableRef> {
        ElementSourceAssemblerBuilder {
            space,
            source: self.source,
            qtable: self.qtable,
        }
    }

    pub fn with_source<Source>(
        self,
        source: &Source,
    ) -> ElementSourceAssemblerBuilder<SpaceRef, &Source, QTableRef> {
        ElementSourceAssemblerBuilder {
            space: self.space,
            source,
            qtable: self.qtable,
        }
    }

    pub fn with_quadrature_table<QTable>(
        self,
        qtable: &QTable,
    ) -> ElementSourceAssemblerBuilder<SpaceRef, SourceRef, &QTable> {
        ElementSourceAssemblerBuilder {
            space: self.space,
            source: self.source,
            qtable,
        }
    }
}

impl<'a, Space, Source, QTable> ElementSourceAssemblerBuilder<&'a Space, &'a Source, &'a QTable> {
    pub fn build(self) -> ElementSourceAssembler<'a, Space, Source, QTable> {
        ElementSourceAssembler {
            space: self.space,
            qtable: self.qtable,
            source: self.source,
        }
    }
}

pub struct ElementSourceAssembler<'a, Space, Source, QTable> {
    space: &'a Space,
    qtable: &'a QTable,
    source: &'a Source,
}

thread_local! { static SOURCE_WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default()) }

impl<'a, Space, Source, QTable> ElementConnectivityAssembler
    for ElementSourceAssembler<'a, Space, Source, QTable>
where
    Space: FiniteElementConnectivity,
    Source: Operator,
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
    for ElementSourceAssembler<'a, Space, Source, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Source: SourceFunction<T, Space::GeometryDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Source::Parameters>,
    DefaultAllocator:
        TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, Source::SolutionDim>,
{
    fn assemble_element_vector_into(
        &self,
        element_index: usize,
        mut output: DVectorSliceMut<T>,
    ) -> eyre::Result<()> {
        SOURCE_WORKSPACE.with(|ws| {
            // TODO: Is it possible to simplify retrieving a mutable reference to the workspace?
            let mut ws: RefMut<SourceTermWorkspace<T, Space::ReferenceDim, Source::Parameters>> =
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

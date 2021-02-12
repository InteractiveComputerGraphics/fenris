//! New functionality that will eventually supercede various methods in the `assembly` module.

use crate::allocators::{BiDimAllocator, SmallDimAllocator};
use crate::assembly::local::{
    compute_volume_u_grad, ElementConnectivityAssembler, ElementMatrixAssembler,
};
use crate::element::{MatrixSlice, MatrixSliceMut};
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{
    DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimName, Dynamic,
    MatrixMN, MatrixSliceMN, Point, RealField, Scalar, VectorN, U1,
};
use crate::space::VolumetricFiniteElementSpace;
use crate::workspace::Workspace;
use crate::SmallDim;
use itertools::izip;
use nalgebra::DMatrix;
use std::cell::RefCell;
use std::error::Error;
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

pub struct ElementEllipticAssembler<'a, T: Scalar, Space, Op, QTable> {
    space: Space,
    op: Op,
    qtable: QTable,
    u: DVectorSlice<'a, T>,
}

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
            // let mut output = MatrixSliceMutMN::from_slice_generic(output.as_mut_slice(), s, n);

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

    fn num_elements(&self) -> usize;

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
    let s = solution_dim;
    for (i_local, i_global) in indices.iter().enumerate() {
        local
            .index_mut((.., i_local))
            .copy_from(&global.index((s * i_global..s * i_global + s, ..)));
    }
}

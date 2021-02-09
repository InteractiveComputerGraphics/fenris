//! New functionality that will eventually supercede various methods in the `assembly` module.

use crate::allocators::{BiDimAllocator, FiniteElementAllocator};
use crate::assembly::local::{compute_volume_u_grad, ElementConnectivityAssembler};
use crate::element::MatrixSliceMut;
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
    ) -> Result<(), Box<dyn Send + Error>>;
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
    Space: FiniteElementSpace2<T>,
    Op: Operator<T>,
    DefaultAllocator: FiniteElementAllocator<T, Space::GeometryDim, Space::ReferenceDim>
        + Allocator<T, Space::GeometryDim, Op::SolutionDim>,
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
struct EllipticVectorWorkspace<T, GeometryDim, Data>
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

impl<T, GeometryDim, Data> EllipticVectorWorkspace<T, GeometryDim, Data>
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

impl<T, GeometryDim, Data> Default for EllipticVectorWorkspace<T, GeometryDim, Data>
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
    ) -> Result<(), Box<dyn Send + Error>> {
        Self::WORKSPACE.with(|workspace| {
            let mut ws = workspace.borrow_mut();
            let ws =
                ws.get_or_default::<EllipticVectorWorkspace<T, Space::GeometryDim, Op::Data>>();

            let quadrature_size = self.qtable.element_quadrature_size(element_index);
            let s = Op::SolutionDim::dim();
            let n = self.element_node_count(element_index);
            ws.resize_workspace(quadrature_size, s, n);
            assert_eq!(output.len(), s * n);

            self.space
                .populate_element_nodes(&mut ws.element_nodes, element_index);
            gather_global_to_local(&self.u, &mut ws.u_element, &ws.element_nodes, s);

            // Reshape u and the output vector into matrices of dimensions s x n,
            // where s is the solution dim and n is the number of nodes
            // (each column corresponds to the vector term associated with the corresponding node)
            let s = Op::SolutionDim::name();
            let n = Dynamic::new(n);
            let u_element = MatrixSliceMN::from_slice_generic(ws.u_element.as_slice(), s, n);
            let mut output = MatrixSliceMutMN::from_slice_generic(output.as_mut_slice(), s, n);

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

                // TODO: Make error instead of panic?
                let J_inv = J.try_inverse().expect("Jacobian must be invertible");
                let J_inv_t = J_inv.transpose();
                let u_grad = compute_volume_u_grad(&J_inv_t, &*phi_grad_ref, u_element);

                let g = self.op.compute_elliptic_term(&u_grad, data);
                let g_J_inv_t = g.transpose() * &J_inv_t;
                output.gemm(w * J_det.abs(), &g_J_inv_t, &phi_grad_ref, T::one());
            }

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

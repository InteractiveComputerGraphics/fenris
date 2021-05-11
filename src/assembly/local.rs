use std::error::Error;
use std::ops::AddAssign;

use alga::general::{ClosedAdd, ClosedMul};
use nalgebra::base::allocator::Allocator;
use nalgebra::{
    DMatrix, DMatrixSliceMut, DefaultAllocator, DimMin, DimName, Dynamic, MatrixMN, MatrixSliceMN,
    RealField, Scalar, VectorN, U1,
};
use num::{One, Zero};

use crate::allocators::{FiniteElementMatrixAllocator, VolumeFiniteElementAllocator};

use crate::connectivity::Connectivity;
use crate::element::{MatrixSlice, MatrixSliceMut, VolumetricFiniteElement};
use crate::mesh::Mesh;
use crate::quadrature::Quadrature;

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
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    fn as_connectivity_assembler(&self) -> &dyn ElementConnectivityAssembler;
}

pub trait GeneralizedEllipticOperator<T, SolutionDim, GeometryDim>
where
    T: Scalar,
    SolutionDim: DimName,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, SolutionDim>,
{
    fn compute_elliptic_term(
        &self,
        gradient: &MatrixMN<T, GeometryDim, SolutionDim>,
    ) -> MatrixMN<T, GeometryDim, SolutionDim>;
}

pub trait GeneralizedEllipticContraction<T, SolutionDim, GeometryDim>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SolutionDim: DimName,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, SolutionDim>
        + Allocator<T, GeometryDim, GeometryDim>
        + Allocator<T, SolutionDim, SolutionDim>
        + Allocator<T, GeometryDim>
        + Allocator<T, U1, GeometryDim>,
{
    fn contract(
        &self,
        gradient: &MatrixMN<T, GeometryDim, SolutionDim>,
        a: &VectorN<T, GeometryDim>,
        b: &VectorN<T, GeometryDim>,
    ) -> MatrixMN<T, SolutionDim, SolutionDim>;

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
    ///
    /// TODO: Consider using a unit-stride matrix slice for performance reasons.
    fn contract_multiple_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        gradient: &MatrixMN<T, GeometryDim, SolutionDim>,
        a: &MatrixSliceMN<T, GeometryDim, Dynamic>,
    ) {
        let num_nodes = a.ncols();
        let output_dim = num_nodes * SolutionDim::dim();
        assert_eq!(output_dim, output.nrows());
        assert_eq!(output_dim, output.ncols());

        let sdim = SolutionDim::dim();
        for i in 0..num_nodes {
            for j in i..num_nodes {
                let a_i = a.fixed_slice::<GeometryDim, U1>(0, i).clone_owned();
                let a_j = a.fixed_slice::<GeometryDim, U1>(0, j).clone_owned();
                let contraction = self.contract(gradient, &a_i, &a_j);
                output
                    .fixed_slice_mut::<SolutionDim, SolutionDim>(i * sdim, j * sdim)
                    .add_assign(&contraction);

                // TODO: We currently assume symmetry. Should maybe have a method that
                // says whether it is symmetric or not?
                if i != j {
                    output
                        .fixed_slice_mut::<SolutionDim, SolutionDim>(j * sdim, i * sdim)
                        .add_assign(&contraction.transpose());
                }
            }
        }
    }
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
        let f = function(&X, &u, &u_grad);
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

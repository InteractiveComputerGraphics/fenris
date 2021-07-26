use crate::allocators::BiDimAllocator;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{
    DMatrixSliceMut, DVectorSlice, DefaultAllocator, DimName, MatrixMN, RealField, Scalar, VectorN,
};
use crate::{SmallDim, Symmetry};

mod laplace;
pub use laplace::*;
use nalgebra::min;

pub trait Operator<T, GeometryDim> {
    type SolutionDim: SmallDim;

    /// The parameters associated with the operator.
    ///
    /// Typically this encodes material information, such as density, stiffness and other physical
    /// quantities. This is intended to be paired with data associated with individual
    /// quadrature points during numerical integration.
    type Parameters: Default + Clone + 'static;
}

pub trait EllipticOperator<T, GeometryDim>: Operator<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim, Self::SolutionDim>,
{
    /// Compute the elliptic operator $g = g(\nabla u)$ with the provided
    /// [operator parameters](Operator::Parameters).
    fn compute_elliptic_operator(
        &self,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        parameters: &Self::Parameters,
    ) -> MatrixMN<T, GeometryDim, Self::SolutionDim>;
}

/// A contraction operator encoding derivative information for an elliptic operator.
///
/// The contraction operator for an elliptic operator $g = g(\nabla u)$ evaluated at $\nabla u$
/// is defined as the $s \times s$ matrix associated with vectors $a, b \in \mathbb{R}^d$ by
///
/// $$ \\mathcal{C}\_{g} (\nabla u, a, b)
///     := a_k \pd{g_{ki}}{G_{mj}} (\nabla u) \\, b_m \enspace e_i \otimes e_j, $$
///
/// where $G = \nabla u$. We have used Einstein summation notation to simplify the notation
/// for the above expression.
///
/// TODO: Maybe return results in impls...?
pub trait EllipticContraction<T, GeometryDim>: Operator<T, GeometryDim>
where
    T: RealField,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    /// Compute $ \mathcal{C}_g(\nabla u, a, b)$ with the given parameters.
    fn contract(
        &self,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        a: &VectorN<T, GeometryDim>,
        b: &VectorN<T, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> MatrixMN<T, Self::SolutionDim, Self::SolutionDim>;

    /// Whether the contraction operator is symmetric.
    ///
    /// The contraction operator is *symmetric* if, for all $\nabla u, a, b$, regardless
    /// of [operator parameters](Operator::Parameters),
    /// $$
    ///     \mathcal{C}_g(\nabla u, a, b) = \mathcal{C}_g(\nabla u, b, a)^T.
    /// $$
    ///
    /// Symmetry can be exploited to only fill about half of the matrix entries in batch operations
    /// (see [accumulate_contractions_into](Self::accumulate_contractions_into)). The default
    /// implementation indicates non-symmetry.
    fn symmetry(&self) -> Symmetry {
        Symmetry::NonSymmetric
    }

    /// Compute the contraction for a number of vectors at the same time, with the given
    /// parameters.
    ///
    /// The vectors $a \in \mathbb{R}^{dM}$ and $b \in \mathbb{R}^{dN}$ are stacked vectors
    /// $$
    /// \begin{align*}
    /// a := \begin{pmatrix}
    /// a_1 \newline
    /// \vdots \newline
    /// a_M
    /// \end{pmatrix},
    /// \qquad
    /// b:= \begin{pmatrix}
    /// b_1 \newline
    /// \vdots \newline
    /// b_N
    /// \end{pmatrix}
    /// \end{align*}
    /// $$
    /// and $a_I \in \mathbb{R}^d$, $b_J \in \mathbb{R}^d$ for $I = 1, \dots, M$, $J = 1, \dots, N$.
    /// Let $C \in \mathbb{R}^{sM \times sN}$ denote the output matrix,
    /// which is a block matrix of the form
    /// $$
    /// \begin{align*}
    /// C := \begin{pmatrix}
    /// C_{11} & \dots  & C_{1N} \newline
    /// \vdots & \ddots & \vdots \newline
    /// C_{M1} & \dots  & C_{MN}
    /// \end{pmatrix}
    /// \end{align*},
    /// $$
    /// where each block $C_{IJ}$ is an $s \times s$ matrix. This method **accumulates** the
    /// block-wise **scaled** contractions in the following manner:
    ///
    /// $$
    /// C_{IJ} \gets C_{IJ} + \alpha \mathcal{C}_g(\nabla u, a_I, b_J).
    /// $$
    ///
    /// The default implementation repeatedly calls [contract](Self::contract). However,
    /// this might often be inefficient: Since $\nabla u$ is constant for all vectors
    /// $a_I, b_J$, it's often possible to compute the operation for all vectors
    /// at once much more efficiently than one at a time. For performance reasons, it is therefore
    /// often advisable to override this method.
    ///
    /// The method can exploit symmetry: If [self.symmetry()](Self::symmetry) indicates symmetry,
    /// then only the **block upper triangle** needs to be filled. More precisely,
    /// only $C_{IJ}$ for $I \leq J$ need to be populated.
    /// Consumers of this method
    /// **must** take this into account by checking for symmetry of the operator.
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()` or `a.len()` is not divisible by $d$ (`GeometryDim`).
    ///
    /// Panics if `output.nrows() != s * M` or `output.ncols() != output.ncols() * N`.
    #[allow(non_snake_case)]
    fn accumulate_contractions_into(
        &self,
        mut output: DMatrixSliceMut<T>,
        alpha: T,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        a: DVectorSlice<T>,
        b: DVectorSlice<T>,
        parameters: &Self::Parameters,
    ) {
        let d = GeometryDim::dim();
        let s = Self::SolutionDim::dim();
        assert!(
            a.len() % d == 0,
            "Dimension of a must be divisible by d (GeometryDim)"
        );
        assert!(
            b.len() % d == 0,
            "Dimension of b must be divisible by d (GeometryDim)"
        );
        let M = a.len() / d;
        let N = b.len() / d;
        assert_eq!(
            output.nrows(),
            s * M,
            "Number of rows in output matrix is not consistent with a"
        );
        assert_eq!(
            output.ncols(),
            s * N,
            "Number of columns in output matrix is not consistent with b"
        );
        let s_times_s = (Self::SolutionDim::name(), Self::SolutionDim::name());
        let symmetry = self.symmetry();

        // Note: We fill the matrix (block) column-by-column since the matrix is stored in
        // column-major format
        for J in 0..N {
            let row_range = match symmetry {
                Symmetry::Symmetric => min(J + 1, M),
                Symmetry::NonSymmetric => M,
            };
            for I in 0..row_range {
                let a_I = a.rows_generic(d * I, GeometryDim::name()).clone_owned();
                let b_J = b.rows_generic(d * J, GeometryDim::name()).clone_owned();
                let mut c_IJ = output.generic_slice_mut((s * I, s * J), s_times_s);
                let contraction = self.contract(gradient, &a_I, &b_J, parameters);
                c_IJ += contraction * alpha;
            }
        }
    }
}

/// An energy function associated with an elliptic operator.
///
/// The elliptic energy is a function $\psi: \mathbb{R}^{d \times s} \rightarrow \mathbb{R}$
/// that represents some energy-like quantity *per unit volume*. Typically the elliptic energy
/// arises in applications as the total potential energy over the domain
///
/// $$ E[u] := \int_{\Omega} \psi (\nabla u) \dx. $$
///
/// The elliptic energy is then related to the elliptic operator
/// $g: \mathbb{R}^{d \times s} \rightarrow \mathbb{R}^{d \times s}$ by the relation
///
/// $$ g = \pd{\psi}{G} $$
///
/// where $G = \nabla u$.
/// This relationship lets us connect the total energy to the weak form associated with $g$
/// by noticing that the functional derivative gives us the functional differential
/// with respect to a test function $v$
///
/// $$ \partial E = \int_{\Omega} \pd{\psi}{G} : \nabla v \dx
///     = \int_{\Omega} g : \nabla v \dx. $$
///
/// The simplest example of an elliptic energy is the
/// [Dirichlet energy](https://en.wikipedia.org/wiki/Dirichlet_energy)
/// $$ E[u] = \int_{\Omega} \frac{1}{2} \| \nabla u \|^2 \dx $$
/// where in our framework, $ \psi (\nabla u) = \frac{1}{2} \| \nabla u \|^2$ and
/// $g = \nabla u$, which gives the weak form associated with Laplace's equation.
///
///
///
/// TODO: Extend elliptic energy to have an additional domain dependence,
/// e.g. $\psi = \psi(x, \nabla u)$.
pub trait EllipticEnergy<T, GeometryDim>: Operator<T, GeometryDim>
where
    T: RealField,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    fn compute_energy(
        &self,
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        parameters: &Self::Parameters,
    ) -> T;
}

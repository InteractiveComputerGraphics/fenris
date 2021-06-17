use std::ops::AddAssign;

use crate::allocators::BiDimAllocator;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{
    DMatrixSliceMut, DefaultAllocator, DimName, Dynamic, MatrixMN, MatrixSliceMN, RealField,
    Scalar, VectorN, U1,
};
use crate::SmallDim;

mod laplace;

pub use laplace::*;

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

pub trait EllipticContraction<T, GeometryDim>: Operator
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
pub trait EllipticEnergy<T, GeometryDim>: Operator
where
    T: RealField,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    fn compute_energy(
        gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
        parameters: &Self::Parameters,
    ) -> T;
}

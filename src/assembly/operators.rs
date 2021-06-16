use crate::allocators::{BiDimAllocator, SmallDimAllocator};
use crate::nalgebra::{DefaultAllocator, DMatrixSliceMut, Dynamic, MatrixMN, MatrixSliceMN, RealField, Scalar, U1, Vector1, VectorN, DimName};
use crate::nalgebra::allocator::Allocator;
use crate::SmallDim;
use std::ops::AddAssign;

/// The Laplace operator $\Delta = \nabla^2$.
///
/// TODO: We need to make this precise, i.e. what *exactly* this operator does
/// (and sign convention)
///
/// TODO: Docs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaplaceOperator;

/// Declare properties of the Laplace operator
impl Operator for LaplaceOperator {
    /// The Laplace operator operates on scalar values, so the dimension of our solution variable
    /// is 1.
    type SolutionDim = U1;
    /// There are no parameters (density, stiffness, temperature etc.) associated with the Laplace
    /// operator.
    type Parameters = ();
}

impl<T, D> EllipticContraction<T, D> for LaplaceOperator
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: SmallDimAllocator<T, D>
{
    // TODO: Document
    fn contract(
        &self,
        _gradient: &MatrixMN<T, D, Self::SolutionDim>,
        _data: &Self::Parameters,
        a: &VectorN<T, D>,
        b: &VectorN<T, D>,
    ) -> MatrixMN<T, Self::SolutionDim, Self::SolutionDim> {
        Vector1::new(a.dot(&b))
    }
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

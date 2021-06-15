use crate::nalgebra::{U1, RealField, DefaultAllocator, MatrixMN, VectorN, Vector1};
use crate::assembly::local::{Operator, EllipticContraction};
use crate::allocators::SmallDimAllocator;
use crate::SmallDim;

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
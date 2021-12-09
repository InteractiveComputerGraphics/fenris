use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::assembly::operators::{EllipticContraction, EllipticEnergy, EllipticOperator, Operator};
use crate::nalgebra::{DefaultAllocator, OMatrix, OVector, RealField, Vector1, U1};
use crate::{SmallDim, Symmetry};
use numeric_literals::replace_float_literals;

/// The Laplace operator $\Delta = \nabla^2$.
///
/// TODO: We need to make this precise, i.e. what *exactly* this operator does
/// (and sign convention)
///
/// TODO: Docs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaplaceOperator;

/// Declare properties of the Laplace operator
impl<T, GeometryDim> Operator<T, GeometryDim> for LaplaceOperator {
    /// The Laplace operator operates on scalar values, so the dimension of our solution variable
    /// is 1.
    type SolutionDim = U1;
    /// There are no parameters (density, stiffness, temperature etc.) associated with the Laplace
    /// operator.
    type Parameters = ();
}

impl<T, D> EllipticEnergy<T, D> for LaplaceOperator
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: BiDimAllocator<T, D, Self::SolutionDim>,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn compute_energy(&self, gradient: &OMatrix<T, D, Self::SolutionDim>, _parameters: &Self::Parameters) -> T {
        0.5 * gradient.dot(&gradient)
    }
}

impl<T, D> EllipticOperator<T, D> for LaplaceOperator
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: BiDimAllocator<T, D, Self::SolutionDim>,
{
    fn compute_elliptic_operator(
        &self,
        gradient: &OMatrix<T, D, Self::SolutionDim>,
        _data: &Self::Parameters,
    ) -> OMatrix<T, D, Self::SolutionDim> {
        gradient.clone()
    }
}

impl<T, D> EllipticContraction<T, D> for LaplaceOperator
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: DimAllocator<T, D>,
{
    // TODO: Document
    fn contract(
        &self,
        _gradient: &OMatrix<T, D, Self::SolutionDim>,
        a: &OVector<T, D>,
        b: &OVector<T, D>,
        _data: &Self::Parameters,
    ) -> OMatrix<T, Self::SolutionDim, Self::SolutionDim> {
        Vector1::new(a.dot(&b))
    }

    fn symmetry(&self) -> Symmetry {
        Symmetry::Symmetric
    }
}

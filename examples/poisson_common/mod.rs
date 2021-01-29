use fenris::assembly::{GeneralizedEllipticContraction, GeneralizedEllipticOperator};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Matrix1, RealField, Scalar, VectorN, U1};

pub struct PoissonEllipticOperator;

impl<T, GeometryDim> GeneralizedEllipticOperator<T, U1, GeometryDim> for PoissonEllipticOperator
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
{
    fn compute_elliptic_term(&self, gradient: &VectorN<T, GeometryDim>) -> VectorN<T, GeometryDim> {
        gradient.clone_owned()
    }
}

impl<T, GeometryDim> GeneralizedEllipticContraction<T, U1, GeometryDim> for PoissonEllipticOperator
where
    T: RealField,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>
        + Allocator<T, GeometryDim, GeometryDim>
        + Allocator<T, U1, GeometryDim>,
{
    fn contract(
        &self,
        _gradient: &VectorN<T, GeometryDim>,
        a: &VectorN<T, GeometryDim>,
        b: &VectorN<T, GeometryDim>,
    ) -> Matrix1<T> {
        Matrix1::new(a.dot(&b))
    }
}

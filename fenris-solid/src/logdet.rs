use crate::PhysicalDim;
use fenris::allocators::DimAllocator;
use fenris::nalgebra::{DefaultAllocator, Matrix1, Matrix2, Matrix3, OMatrix};
use fenris::util::try_transmute_ref;
use fenris::Real;
use numeric_literals::replace_float_literals;

/// Compute $\log(\det \vec F)$ for the deformation gradient $\vec F$ given $\pd{\vec u}{\vec X}$.
///
/// The implementation may be far more accurate than first forming $\vec F$, which discards
/// valuable information for small deformations (small $\pd{\vec u}{\vec X}$).
///
/// The implementation is based on the technique described in the
/// [libCEED documentation](https://libceed.org/en/latest/examples/solids).
#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn log_det_F<T, D>(du_dX: &OMatrix<T, D, D>) -> Option<T>
where
    T: Real,
    D: PhysicalDim,
    DefaultAllocator: DimAllocator<T, D>,
{
    match D::USIZE {
        1 => {
            let du_dX: &Matrix1<T> = try_transmute_ref(du_dX).unwrap();
            let det = du_dX[(0, 0)];
            (det > 0.0).then(|| det.ln())
        }
        2 => log_det_F_2d(try_transmute_ref(du_dX).unwrap()),
        3 => log_det_F_3d(try_transmute_ref(du_dX).unwrap()),
        _ => unreachable!("Physical dimensions do not extend past 3 dimensions"),
    }
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).unwrap())]
fn log_det_F_2d<T: Real>(du_dX: &Matrix2<T>) -> Option<T> {
    // See comments in 3D impl for more elaborate explanation
    // Given a matrix A = [a, b; c, d] the determinant is
    //  det(A) = ad - bc
    // Let U = du_dX be the displacement Jacobian and uij its entries. Then we can write our determinant as
    //   det(F) = det(I + U) = a*d - b*c = (1+u11)*(1+u22) - b*c
    //          = 1 + u11*u22 + u11 + u22 - b*c
    //          = 1 + γ
    // and so
    //   log(det(F)) = log(1 + γ) = log1p(γ)
    let u11 = du_dX[(0, 0)];
    let u22 = du_dX[(1, 1)];
    let b = du_dX[(0, 1)];
    let c = du_dX[(1, 0)];
    let gamma = u11 * u22 + u11 + u22 - b * c;
    (gamma > -1.0).then(|| T::ln_1p(gamma))
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).unwrap())]
fn log_det_F_3d<T: Real>(du_dX: &Matrix3<T>) -> Option<T> {
    // Given a matrix A = [a, b, c; d, e, f; g, h, i] the determinant is
    //  det(A) = aei + bfg + cdh - ceg - bdi - afh
    // The first term is the product of the diagonals. Since F = I + du_dX,
    // the diagonal of F will be close to 1 for small du_dX,
    // and so the determinant is dominated by this first term.
    // Let U = du_dX be the displacement Jacobian and uij its entries. Then we can write our determinant as
    //   det(F) = det(I + U) = (1+u11)*(1+u22)*(1+u33) + ...
    //          = 1 + u11*u22*u33 + u11*u22 + u11*u33 + u22*u33 + u11 + u22 + u33 + ...
    //          = 1 + γ
    // and so
    //   log(det(F)) = log(1 + γ) = log1p(γ)
    let u11 = du_dX[(0, 0)];
    let u22 = du_dX[(1, 1)];
    let u33 = du_dX[(2, 2)];
    let a = 1.0 + u11;
    let e = 1.0 + u22;
    let i = 1.0 + u33;
    let b = du_dX[(0, 1)];
    let c = du_dX[(0, 2)];
    let d = du_dX[(1, 0)];
    let f = du_dX[(1, 2)];
    let g = du_dX[(2, 0)];
    let h = du_dX[(2, 1)];
    let gamma = u11 * u22 * u33 + u11 * u22 + u11 * u33 + u22 * u33 + u11 + u22 + u33 + b * f * g + c * d * h
        - c * e * g
        - b * d * i
        - a * f * h;
    (gamma > -1.0).then(|| T::ln_1p(gamma))
}

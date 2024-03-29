//! Quadrature rules for 1D domains.
use crate::quadrature::{convert_quadrature_rule_from_1d_f64, QuadraturePair1d};
use crate::Real;
use fenris_quadrature::univariate;

pub fn gauss<T: Real>(num_points: usize) -> QuadraturePair1d<T> {
    let (weights, points) = univariate::gauss(num_points);
    convert_quadrature_rule_from_1d_f64((weights, points))
}

pub fn try_gauss_lobatto<T: Real>(num_points: usize) -> Option<QuadraturePair1d<T>> {
    univariate::try_gauss_lobatto(num_points).map(convert_quadrature_rule_from_1d_f64)
}

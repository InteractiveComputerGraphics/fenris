//! Quadrature rules constructed from products of 1D quadrature rules.
use crate::quadrature::{
    convert_quadrature_rule_from_2d_f64, convert_quadrature_rule_from_3d_f64, QuadraturePair2d, QuadraturePair3d,
};
use crate::Real;
use fenris_quadrature::tensor;

pub fn quadrilateral_gauss<T: Real>(num_points_per_dim: usize) -> QuadraturePair2d<T> {
    let (weights, points) = tensor::quadrilateral_gauss(num_points_per_dim);
    convert_quadrature_rule_from_2d_f64((weights, points))
}

pub fn hexahedron_gauss<T: Real>(num_points_per_dim: usize) -> QuadraturePair3d<T> {
    let (weights, points) = tensor::hexahedron_gauss(num_points_per_dim);
    convert_quadrature_rule_from_3d_f64((weights, points))
}

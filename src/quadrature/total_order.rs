//! Quadrature rules parametrized by polynomial total-order accuracy.
//!
//! TODO: Docs
//!
//! TODO: Tests? Can test that we have equivalence with `fenris-quadrature` maybe

use fenris_quadrature::polyquad;

use crate::quadrature;
use crate::quadrature::{QuadratureError, QuadraturePair2d, QuadraturePair3d};
use crate::Real;

pub fn triangle<T: Real>(strength: usize) -> Result<QuadraturePair2d<T>, QuadratureError> {
    let (weights, points) = polyquad::triangle(strength)?;
    Ok(quadrature::convert_quadrature_rule_from_2d_f64((weights, points)))
}

pub fn quadrilateral<T: Real>(strength: usize) -> Result<QuadraturePair2d<T>, QuadratureError> {
    let (weights, points) = polyquad::quadrilateral(strength)?;
    Ok(quadrature::convert_quadrature_rule_from_2d_f64((weights, points)))
}

pub fn tetrahedron<T: Real>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError> {
    let (weights, points) = polyquad::tetrahedron(strength)?;
    Ok(quadrature::convert_quadrature_rule_from_3d_f64((weights, points)))
}

pub fn hexahedron<T: Real>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError> {
    let (weights, points) = polyquad::hexahedron(strength)?;
    Ok(quadrature::convert_quadrature_rule_from_3d_f64((weights, points)))
}

pub fn prism<T: Real>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError> {
    let (weights, points) = polyquad::prism(strength)?;
    Ok(quadrature::convert_quadrature_rule_from_3d_f64((weights, points)))
}

pub fn pyramid<T: Real>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError> {
    let (weights, points) = polyquad::pyramid(strength)?;
    Ok(quadrature::convert_quadrature_rule_from_3d_f64((weights, points)))
}

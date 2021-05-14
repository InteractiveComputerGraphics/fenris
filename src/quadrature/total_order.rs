//! Quadrature rules parametrized by polynomial total-order accuracy.
//!
//! TODO: Docs
//!
//! TODO: Tests? Can test that we have equivalence with `fenris-quadrature` maybe

use crate::quadrature::{QuadraturePair2d, QuadratureError, QuadraturePair3d};
use crate::nalgebra::{Point2, Point3, RealField, convert};
use fenris_quadrature::{polyquad};

fn convert_quadrature_rule_from_2d_f64<T>(quadrature: fenris_quadrature::Rule2d)
    -> QuadraturePair2d<T>
where
    T: RealField,
{
    let (weights, points) = quadrature;
    let weights = weights.into_iter()
        .map(convert)
        .collect();
    let points = points.into_iter()
        .map(Point2::from)
        .map(convert)
        .collect();
    (weights, points)
}

fn convert_quadrature_rule_from_3d_f64<T>(quadrature: fenris_quadrature::Rule3d)
                                          -> QuadraturePair3d<T>
    where
        T: RealField,
{
    let (weights, points) = quadrature;
    let weights = weights.into_iter()
        .map(convert)
        .collect();
    let points = points.into_iter()
        .map(Point3::from)
        .map(convert)
        .collect();
    (weights, points)
}

pub fn triangle<T: RealField>(strength: usize) -> Result<QuadraturePair2d<T>, QuadratureError>
{
    let (weights, points) = polyquad::triangle(strength)?;
    Ok(convert_quadrature_rule_from_2d_f64((weights, points)))
}

pub fn quadrilateral<T: RealField>(strength: usize) -> Result<QuadraturePair2d<T>, QuadratureError>
{
    let (weights, points) = polyquad::quadrilateral(strength)?;
    Ok(convert_quadrature_rule_from_2d_f64((weights, points)))
}

pub fn tetrahedron<T: RealField>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError>
{
    let (weights, points) = polyquad::tetrahedron(strength)?;
    Ok(convert_quadrature_rule_from_3d_f64((weights, points)))
}

pub fn hexahedron<T: RealField>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError>
{
    let (weights, points) = polyquad::hexahedron(strength)?;
    Ok(convert_quadrature_rule_from_3d_f64((weights, points)))
}

pub fn prism<T: RealField>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError>
{
    let (weights, points) = polyquad::prism(strength)?;
    Ok(convert_quadrature_rule_from_3d_f64((weights, points)))
}

pub fn pyramid<T: RealField>(strength: usize) -> Result<QuadraturePair3d<T>, QuadratureError>
{
    let (weights, points) = polyquad::pyramid(strength)?;
    Ok(convert_quadrature_rule_from_3d_f64((weights, points)))
}
//! Quadrature rules for finite element reference domains.
//!
#![cfg_attr(feature = "doc-images",
cfg_attr(all(),
doc = ::embed_doc_image::embed_image!("ref_hex", "assets/reference_hex.svg")))]
#![cfg_attr(
not(feature = "doc-images"),
doc = "**Doc images not enabled**. Compile with feature `doc-images` and Rust version >= 1.54 \
           to enable."
)]
//!
//! The main purpose of this crate is to support the `fenris` FEM library. However, it has been
//! designed so that the quadrature rules available here may be used completely independently
//! of `fenris`.
//!
//! # Reference domains
//!
//! ## Segment
//!
//! ## Triangle
//!
//! ## Quadrilateral
//!
//! ## Hexahedron
//!
//! ![Reference hexahedron][ref_hex]
//!
//! ## Tetrahedron
//!
//! ## Pyramid
//!
//! ## Prism
//!
//! ![Reference hexahedron][ref_hex]
//!
//! TODO: Document conventions for reference domains
//!
//! TODO: Document how quadratures work, e.g. the concept of a reference domain and that
//! quadrature rules are specific to a reference domain

use std::fmt;
use std::fmt::{Display, Formatter};

pub mod polyquad;
pub mod univariate;

/// Library-wide error type.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Indicates that a rule satisfying the given requirements is not available.
    NoRuleAvailable,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRuleAvailable => {
                write!(
                    f,
                    "There is no quadrature rule satisfying the requirements available"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// A D-dimensional point.
pub type Point<const D: usize> = [f64; D];

/// A D-dimensional quadrature rule.
///
/// A quadrature rule consists of weights and points.
pub type Rule<const D: usize> = (Vec<f64>, Vec<Point<D>>);

/// Integrates the given function with the given quadrature rule.
///
/// # Examples
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use matrixcompare::assert_scalar_eq;
/// use fenris_quadrature::integrate;
///
/// // Integrate f over the reference triangle
/// let f = |x, y| x * x * y;
///
/// // The total order of f is 3, so we obtain a quadrature rule with strength 3
/// let rule = fenris_quadrature::polyquad::triangle(3)?;
/// let integral = integrate(&rule, |&[x, y]| f(x, y));
/// let expected_integral = -2.0 / 15.0;
/// assert_scalar_eq!(integral, expected_integral, comp=abs, tol=1e-14);
/// # Ok(()) }
/// ```
pub fn integrate<const D: usize>(rule: &Rule<D>, f: impl Fn(&Point<D>) -> f64) -> f64 {
    let (weights, points) = rule;
    weights.iter().zip(points).map(|(w, p)| w * f(p)).sum()
}

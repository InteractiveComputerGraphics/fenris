//! Quadrature rules for finite element reference domains.
//!
//! The main purpose of this crate is to support the `fenris` FEM library. However, it has been
//! designed so that the quadrature rules available here may be used completely independently
//! of `fenris`.
//!
//! TODO: Document conventions for reference domains
//!
//! TODO: Tests!

use std::fmt;
use std::fmt::{Display, Formatter};

pub mod polyquad;

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

/// A two-dimensional point.
pub type Point2 = Point<2>;

/// A three-dimensional point.
pub type Point3 = Point<3>;

/// A D-dimensional rule.
pub type Rule<const D: usize> = (Vec<f64>, Vec<Point<D>>);

/// A two-dimensional quadrature rule.
pub type Rule2d = Rule<2>;

/// A three-dimensional rule.
pub type Rule3d = Rule<3>;

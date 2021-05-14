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

/// A two-dimensional point
pub type Point2 = [f64; 2];

/// A three-dimensional point.
pub type Point3 = [f64; 3];

/// A two-dimensional quadrature rule.
pub type Rule2d = (Vec<f64>, Vec<Point2>);

/// A three-dimensional rule.
pub type Rule3d = (Vec<f64>, Vec<Point3>);

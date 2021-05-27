//! Quadrature rules for various 2D and 3D domains generated by polyquad.
//!
//! This module contains quadrature rules published in the [paper][paper]
//!
//! ```text
//! Witherden, Freddie D., and Peter E. Vincent.
//! "On the identification of symmetric quadrature rules for finite element methods."
//! Computers & Mathematics with Applications 69, no. 10 (2015): 1232-1241.
//! ```
//!
//! The quadrature rules are symmetric and have positive weights.
//!
//! TODO: Document maximum strengths for various domains
//!
//! [paper]: https://www.sciencedirect.com/science/article/pii/S0898122115001224#f000035

use crate::{Error, Rule};

/// Attempt to create a quadrature rule for the reference triangle with the provided strength.
///
/// The returned quadrature rule is the smallest rule which provides sufficient accuracy.
/// Note that it's possible for the strength of the returned rule to be higher than required, but
/// it will never be smaller.
///
/// # Errors
///
/// Returns an error if there is no quadrature rule available with sufficient strength.
pub fn triangle(strength: usize) -> Result<Rule<2>, Error> {
    tri_select_minimum(strength)
}

/// Attempt to create a quadrature rule for the reference quadrilateral with the provided strength.
///
/// The returned quadrature rule is the smallest rule which provides sufficient accuracy.
/// Note that it's possible for the strength of the returned rule to be higher than required, but
/// it will never be smaller.
///
/// # Errors
///
/// Returns an error if there is no quadrature rule available with sufficient strength.
pub fn quadrilateral(strength: usize) -> Result<Rule<2>, Error> {
    quad_select_minimum(strength)
}

/// Attempt to create a quadrature rule for the reference tetrahedron with the provided strength.
///
/// The returned quadrature rule is the smallest rule which provides sufficient accuracy.
/// Note that it's possible for the strength of the returned rule to be higher than required, but
/// it will never be smaller.
///
/// # Errors
///
/// Returns an error if there is no quadrature rule available with sufficient strength.
pub fn tetrahedron(strength: usize) -> Result<Rule<3>, Error> {
    tet_select_minimum(strength)
}

/// Attempt to create a quadrature rule for the reference hexahedron with the provided strength.
///
/// The returned quadrature rule is the smallest rule which provides sufficient accuracy.
/// Note that it's possible for the strength of the returned rule to be higher than required, but
/// it will never be smaller.
///
/// # Errors
///
/// Returns an error if there is no quadrature rule available with sufficient strength.
pub fn hexahedron(strength: usize) -> Result<Rule<3>, Error> {
    hex_select_minimum(strength)
}

/// Attempt to create a quadrature rule for the reference prism with the provided strength.
///
/// The returned quadrature rule is the smallest rule which provides sufficient accuracy.
/// Note that it's possible for the strength of the returned rule to be higher than required, but
/// it will never be smaller.
///
/// # Errors
///
/// Returns an error if there is no quadrature rule available with sufficient strength.
pub fn prism(strength: usize) -> Result<Rule<3>, Error> {
    pri_select_minimum(strength)
}

/// Attempt to create a quadrature rule for the reference pyramid with the provided strength.
///
/// The returned quadrature rule is the smallest rule which provides sufficient accuracy.
/// Note that it's possible for the strength of the returned rule to be higher than required, but
/// it will never be smaller.
///
/// # Errors
///
/// Returns an error if there is no quadrature rule available with sufficient strength.
pub fn pyramid(strength: usize) -> Result<Rule<3>, Error> {
    pyr_select_minimum(strength)
}

// Load generated code containing quadrature rules generated by build.rs
include!(concat!(env!("OUT_DIR"), "/polyquad/tri.rs"));
include!(concat!(env!("OUT_DIR"), "/polyquad/quad.rs"));
include!(concat!(env!("OUT_DIR"), "/polyquad/tet.rs"));
include!(concat!(env!("OUT_DIR"), "/polyquad/hex.rs"));
include!(concat!(env!("OUT_DIR"), "/polyquad/pri.rs"));
include!(concat!(env!("OUT_DIR"), "/polyquad/pyr.rs"));
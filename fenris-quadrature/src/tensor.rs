//! 2D and 3D quadrature rules formed by tensor product formulations.
//!
//! For quadrilaterals and hexahedra, quadrature rules can be constructed as tensor products
//! of 1D rules. This module provides rules constructed in this fashion.

use crate::Rule;
use crate::univariate::gauss;

/// A Gauss quadrature rule for the reference quadrilateral.
///
/// The rule is constructed as a tensor product from 1D rules, with the provided number of
/// points per dimension.
pub fn quadrilateral_gauss(num_points_per_dim: usize) -> Rule<2> {
    let n = num_points_per_dim;
    let (weights1d, points1d) = gauss(n);
    let mut weights2d = Vec::with_capacity(n * n);
    let mut points2d = Vec::with_capacity(n * n);

    let rule1d_iter = || weights1d.iter().zip(&points1d);

    for (&wx, &[x]) in rule1d_iter() {
        for (&wy, &[y]) in rule1d_iter() {
            let w = wx * wy;
            weights2d.push(w);
            points2d.push([x, y]);
        }
    }

    (weights2d, points2d)
}

/// A Gauss quadrature rule for the reference hexahedron.
///
/// The rule is constructed as a tensor product from 1D rules, with the provided number of
/// points per dimension.
pub fn hexahedron_gauss(num_points_per_dim: usize) -> Rule<3> {
    let n = num_points_per_dim;
    let (weights1d, points1d) = gauss(n);
    let mut weights3d = Vec::with_capacity(n * n * n);
    let mut points3d = Vec::with_capacity(n * n * n);

    let rule1d_iter = || weights1d.iter().zip(&points1d);

    for (&wx, &[x]) in rule1d_iter() {
        for (&wy, &[y]) in rule1d_iter() {
            for (&wz, &[z]) in rule1d_iter() {
                let w = wx * wy * wz;
                weights3d.push(w);
                points3d.push([x, y, z]);
            }
        }
    }

    (weights3d, points3d)
}
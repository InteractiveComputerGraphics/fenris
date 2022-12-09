use matrixcompare::assert_scalar_eq;
use nalgebra::point;
use proptest::prelude::*;
use fenris_geometry::{Triangle, Triangle2d};
use fenris_geometry::predicates::orient2d_inexact;

#[test]
fn test_orient2d_inexact_simple_example() {
    let a = point![1.0, 0.0];
    let b = point![2.0, 1.0];
    let c = point![-1.0, 2.0];
    assert_scalar_eq!(orient2d_inexact(&a, &b, &c), 2.0 * Triangle([a, b, c]).signed_area(),
        comp = abs, tol = 1e-9);
}

proptest! {
    #[test]
    fn orient2d_inexact_matches_twice_triangle_signed_area(triangle: Triangle2d<f64>) {
        let Triangle([a, b, c]) = &triangle;
        assert_scalar_eq!(orient2d_inexact(&a, &b, &c), 2.0 * triangle.signed_area(),
            comp = abs, tol = 1e-9 * triangle.area());
    }
}
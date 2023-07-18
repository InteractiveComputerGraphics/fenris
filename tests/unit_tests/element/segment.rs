use fenris::element::{project_physical_coordinates, FiniteElement, Segment2d2Element};

use fenris::geometry::LineSegment2d;

use fenris::util::proptest::point2_f64_strategy;

use nalgebra::{Point1, Point2, Vector1};

use proptest::prelude::*;
use util::assert_approx_matrix_eq;

#[test]
// TODO: Rename to segment2d
fn map_reference_coords_edge2d() {
    let a = Point2::new(5.0, 3.0);
    let b = Point2::new(10.0, 4.0);
    let edge = Segment2d2Element::from(LineSegment2d::from_end_points(a, b));

    let x0 = edge.map_reference_coords(&Point1::new(-1.0));
    assert!(x0.coords.relative_eq(&a.coords, 1e-10, 1e-10));

    let x1 = edge.map_reference_coords(&Point1::new(1.0));
    assert!(x1.coords.relative_eq(&b.coords, 1e-10, 1e-10));

    let x2 = edge.map_reference_coords(&Point1::new(0.0));
    assert!(x2
        .coords
        .relative_eq(&((a.coords + b.coords) / 2.0), 1e-10, 1e-10));

    let x3 = edge.map_reference_coords(&Point1::new(0.5));
    assert!(x3
        .coords
        .relative_eq(&(0.25 * a.coords + 0.75 * b.coords), 1e-10, 1e-10));
}

proptest! {
    #[test]
    fn edge2d_element_jacobian_is_derivative_of_transform(
        (a, b, xi) in (point2_f64_strategy(), point2_f64_strategy(), -1.0..=1.0)
    ) {
        let segment = LineSegment2d::from_end_points(a, b);
        let element = Segment2d2Element::from(segment);
        let xi = Point1::new(xi);

        // Finite difference parameter
        let h = 1e-6;
        let hvec = Vector1::new(h);

        // TODO: Extend VectorFunction and approximate_jacobian to allow
        // maps between domains of different dimension
        let j = element.reference_jacobian(&xi);

        // Approximate Jacobian by finite differences
        let x_plus = element.map_reference_coords(&(xi + hvec));
        let x_minus = element.map_reference_coords(&(xi - hvec));
        let j_approx = (x_plus - x_minus) / (2.0 * h);

        let tol = (a.coords +  b.coords).norm() * h;
        assert_approx_matrix_eq!(j, j_approx, abstol=tol);

    }

    #[test]
    fn edge2d_element_project_physical_coords_with_perturbation(
        (a, b, xi, eps) in (point2_f64_strategy(), point2_f64_strategy(),
                            -1.0..=1.0, prop_oneof!(Just(0.0), -1.0 .. 1.0))
    ) {
        let segment = LineSegment2d::from_end_points(a, b);

        prop_assume!(segment.length() > 0.0);

        let element = Segment2d2Element::from(&segment);
        let xi = Point1::new(xi);
        let x = element.map_reference_coords(&xi);
        // Perturb the surface point in the direction normal to the surface. This checks
        // that the projection actually still manages to correctly reproduce the original point.
        let x_perturbed = &x + eps * segment.normal_dir();
        let xi_proj = project_physical_coordinates(&element, &Point2::from(x_perturbed)).unwrap();
        let x_reconstructed = element.map_reference_coords(&xi_proj);

        prop_assert!((x_reconstructed - x).norm() <= 1e-12);
    }
}

use fenris_geometry::{GeneralPolygon, LineSegment2d, Orientation, Polygon};
use matrixcompare::assert_scalar_eq;
use nalgebra::{Point2, Vector2};

use std::f64;
use util::assert_approx_matrix_eq;

#[test]
fn polygon_area_signed_unsigned() {
    // This is a simple, but fairly non-convex polygon oriented counter-clockwise
    let vertices = vec![
        Point2::new(-5.0, -2.0),
        Point2::new(-3.0, -3.0),
        Point2::new(-1.0, 0.0),
        Point2::new(-3.0, -1.0),
        Point2::new(-5.0, 1.0),
        Point2::new(-3.0, 1.0),
        Point2::new(-6.0, 3.0),
    ];

    let polygon = GeneralPolygon::from_vertices(vertices.clone());

    let expected_area = 10.5;
    assert_scalar_eq!(
        polygon.signed_area(),
        expected_area,
        comp = abs,
        tol = 1e-12
    );
    assert_scalar_eq!(polygon.area(), expected_area, comp = abs, tol = 1e-12);

    let vertices_reversed = {
        let mut v = vertices.clone();
        v.reverse();
        v
    };

    let reversed_polygon = GeneralPolygon::from_vertices(vertices_reversed);

    assert_scalar_eq!(
        reversed_polygon.signed_area(),
        -expected_area,
        comp = abs,
        tol = 1e-12
    );
    assert_scalar_eq!(
        reversed_polygon.area(),
        expected_area,
        comp = abs,
        tol = 1e-12
    );
}

#[test]
fn polygon_intersects_segment() {
    // This is a simple, but fairly non-convex polygon oriented counter-clockwise
    let vertices = vec![
        Point2::new(-5.0, -2.0),
        Point2::new(-3.0, -3.0),
        Point2::new(-1.0, 0.0),
        Point2::new(-3.0, -1.0),
        Point2::new(-5.0, 1.0),
        Point2::new(-3.0, 1.0),
        Point2::new(-6.0, 3.0),
    ];

    let polygon = GeneralPolygon::from_vertices(vertices.clone());

    {
        // Segment is outside (and also outside its convex hull)
        let segment = LineSegment2d::new(Point2::new(-8.0, -1.0), Point2::new(-7.0, 3.0));
        assert_eq!(polygon.intersects_segment(&segment), false);
    }

    {
        // Segment is outside (but inside the convex hull of the polygon)
        let segment = LineSegment2d::new(Point2::new(-3.0, 0.0), Point2::new(-2.0, 1.0));
        assert_eq!(polygon.intersects_segment(&segment), false);
    }

    {
        // Segment is completely inside the polygon
        let segment = LineSegment2d::new(Point2::new(-3.0, -2.0), Point2::new(-5.0, 0.0));
        assert_eq!(polygon.intersects_segment(&segment), true);
    }

    {
        // Segment is partially inside, with one of its endpoints inside
        let segment = LineSegment2d::new(Point2::new(-3.0, -2.0), Point2::new(-4.0, 0.5));
        assert_eq!(polygon.intersects_segment(&segment), true);
    }

    {
        // Segment is partially inside, with none of its endpoints inside
        let segment = LineSegment2d::new(Point2::new(0.0, -1.0), Point2::new(-6.0, 0.0));
        assert_eq!(polygon.intersects_segment(&segment), true);
    }
}

#[test]
fn polygon_closest_edge() {
    // This is a simple, but fairly non-convex polygon oriented counter-clockwise
    let vertices = vec![
        Point2::new(-5.0, -2.0),
        Point2::new(-3.0, -3.0),
        Point2::new(-1.0, 0.0),
        Point2::new(-3.0, -1.0),
        Point2::new(-5.0, 1.0),
        Point2::new(-3.0, 1.0),
        Point2::new(-6.0, 3.0),
    ];

    let polygon = GeneralPolygon::from_vertices(vertices.clone());

    {
        // Point is outside, but inside the convex hull of the polygon
        let point = Point2::new(-3.0, 0.0);
        let closest_edge = polygon.closest_edge(&point).unwrap();
        let expected_t = f64::sqrt(0.5) / 2.82842712474619;

        assert_eq!(closest_edge.edge_index, 3);
        assert_scalar_eq!(
            closest_edge.signed_distance,
            f64::sqrt(0.5),
            comp = abs,
            tol = 1e-12
        );
        assert_scalar_eq!(
            closest_edge.edge_parameter,
            expected_t,
            comp = abs,
            tol = 1e-12
        );
        assert_approx_matrix_eq!(
            closest_edge.edge_point.coords,
            Vector2::new(-3.5, -0.5),
            abstol = 1e-12
        );
    }

    {
        // Point is exactly on the boundary
        let point = Point2::new(-4.5, 1.0);
        let closest_edge = polygon.closest_edge(&point).unwrap();
        let expected_t = 0.25;

        assert_eq!(closest_edge.edge_index, 4);
        assert_scalar_eq!(closest_edge.signed_distance, 0.0, comp = abs, tol = 1e-12);
        assert_scalar_eq!(
            closest_edge.edge_parameter,
            expected_t,
            comp = abs,
            tol = 1e-12
        );
        assert_approx_matrix_eq!(closest_edge.edge_point.coords, point.coords, abstol = 1e-12);
    }

    {
        // Point is inside, closest to a vertex
        let point = Point2::new(-3.1, -1.4);
        let closest_edge = polygon.closest_edge(&point).unwrap();

        // Whether edge 2 or 3 is reported is not well-defined. It can be either.
        assert!([2, 3].contains(&closest_edge.edge_index));
        let expected_t = if closest_edge.edge_index == 2 {
            1.0
        } else {
            0.0
        };
        assert_scalar_eq!(
            closest_edge.signed_distance,
            -0.412310562561766,
            comp = abs,
            tol = 1e-12
        );
        assert_scalar_eq!(
            closest_edge.edge_parameter,
            expected_t,
            comp = abs,
            tol = 1e-12
        );
        assert_approx_matrix_eq!(
            closest_edge.edge_point.coords,
            vertices[3].coords,
            abstol = 1e-12
        );
    }

    {
        // Point is inside, closest to an edge
        let point = Point2::new(-5.0, 0.0);
        let closest_edge = polygon.closest_edge(&point).unwrap();
        let expected_t = 0.61538461538;

        assert_eq!(closest_edge.edge_index, 6);
        assert_scalar_eq!(
            closest_edge.signed_distance,
            -0.392232270276368,
            comp = abs,
            tol = 1e-12
        );
        assert_scalar_eq!(
            closest_edge.edge_parameter,
            expected_t,
            comp = abs,
            tol = 1e-10
        );
        assert_approx_matrix_eq!(
            closest_edge.edge_point.coords,
            Vector2::new(-5.384615384615385, -0.076923076923077),
            abstol = 1e-12
        );
    }
}

#[test]
fn polygon_orient() {
    // This is a simple, but fairly non-convex polygon oriented counter-clockwise
    let vertices = vec![
        Point2::new(-5.0, -2.0),
        Point2::new(-3.0, -3.0),
        Point2::new(-1.0, 0.0),
        Point2::new(-3.0, -1.0),
        Point2::new(-5.0, 1.0),
        Point2::new(-3.0, 1.0),
        Point2::new(-6.0, 3.0),
    ];
    let mut polygon = GeneralPolygon::from_vertices(vertices.clone());

    // Check that area is still the same as a sanity check that we don't accidentally
    // completely break the polygon somehow
    let area = polygon.area();

    assert_eq!(polygon.orientation(), Orientation::Counterclockwise);

    polygon.orient(Orientation::Counterclockwise);
    assert_eq!(polygon.orientation(), Orientation::Counterclockwise);

    polygon.orient(Orientation::Clockwise);
    assert_eq!(polygon.orientation(), Orientation::Clockwise);
    assert_scalar_eq!(polygon.area(), area, comp = abs, tol = 1e-14);
    // We guarantee that the first vertex is the same
    assert_eq!(polygon.vertices()[0], vertices[0]);

    // Changing the orientation back again should completely restore the original polygon
    polygon.orient(Orientation::Counterclockwise);
    assert_eq!(polygon.orientation(), Orientation::Counterclockwise);
    assert_eq!(polygon.vertices(), vertices.as_slice());
}

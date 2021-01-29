use fenris_geometry::{ConvexPolygon, HalfPlane, Line2d, LineSegment2d, Triangle};

use nalgebra::{Point2, Unit, Vector2};

use matrixcompare::assert_scalar_eq;

use util::assert_approx_matrix_eq;

#[test]
fn half_plane_surface_distance_and_contains_point() {
    let x0 = Point2::new(1.0, -1.0);
    let n = Unit::new_normalize(Vector2::new(-1.0, 1.0));
    let half_plane = HalfPlane::from_point_and_normal(x0, n);

    {
        let x = Point2::new(-1.0, 1.0);
        let dist = half_plane.surface_distance_to_point(&x);
        let expected = 2.828427124746;
        let diff: f64 = dist - expected;

        assert!(diff.abs() < 1e-6);
        assert!(half_plane.contains_point(&x));
    }

    {
        let x = Point2::new(2.0, 1.0);
        let dist = half_plane.surface_distance_to_point(&x);
        let expected = 0.7071067811865;
        let diff: f64 = dist - expected;

        assert!(diff.abs() < 1e-6);
    }
}

#[test]
fn empty_polygon_intersect_halfplane() {
    let x0 = Point2::new(0.5, -1.0);
    let n = Unit::new_normalize(Vector2::new(0.3, -2.0));
    let empty = ConvexPolygon::<f64>::from_vertices(vec![]);

    let intersection = empty.intersect_halfplane(&HalfPlane::from_point_and_normal(x0, n));

    assert_eq!(empty, intersection);
}

#[test]
fn point_polygon_intersect_halfplane() {
    let x0 = Point2::new(1.0, -1.0);
    let n = Unit::new_normalize(Vector2::new(-1.0, 1.0));
    let half_plane = HalfPlane::from_point_and_normal(x0, n);

    // Point inside of half plane
    {
        let x = Point2::new(-1.0, 1.0);
        let poly = ConvexPolygon::from_vertices(vec![x]);
        let intersection = poly.intersect_halfplane(&half_plane);
        assert_eq!(intersection, poly);
    }

    // Point outside of half plane
    {
        let x = Point2::new(2.0, -1.0);
        let poly = ConvexPolygon::from_vertices(vec![x]);
        let intersection = poly.intersect_halfplane(&half_plane);
        assert_eq!(intersection, ConvexPolygon::from_vertices(vec![]));
    }
}

#[test]
fn line_polygon_intersect_halfplane() {
    let x0 = Point2::new(1.0, -1.0);
    let n = Unit::new_normalize(Vector2::new(-1.0, 1.0));
    let half_plane = HalfPlane::from_point_and_normal(x0, n);

    // Line represented as polygon intersecting the surface of the halfplane
    {
        let x1 = Point2::new(-1.0, 1.0);
        let x2 = Point2::new(2.0, -1.0);
        let poly = ConvexPolygon::from_vertices(vec![x1, x2]);
        let intersection = poly.intersect_halfplane(&half_plane);
        let expected = ConvexPolygon::from_vertices(vec![x1, Point2::new(1.4, -0.6)]);

        assert_approx_matrix_eq!(
            intersection.vertices()[0],
            expected.vertices()[0],
            abstol = 1e-6
        );
        assert_approx_matrix_eq!(
            intersection.vertices()[1],
            expected.vertices()[1],
            abstol = 1e-6
        );
    }
}

#[test]
fn line_line_intersection() {
    let line1 =
        Line2d::from_point_and_dir(Point2::new(0.0, -1.0), Vector2::new(1.0, 1.0).normalize());
    let line2 =
        Line2d::from_point_and_dir(Point2::new(-2.0, 2.0), Vector2::new(4.0, -2.0).normalize());

    let intersection = line1.intersect(&line2).expect("Intersection exists");

    assert_approx_matrix_eq!(
        intersection,
        Point2::new(4.0 / 3.0, 1.0 / 3.0),
        abstol = 1e-6
    );
}

#[test]
fn triangle_polygon_intersect_halfplane() {
    let triangle = ConvexPolygon::from_vertices(vec![
        Point2::new(0.0, 3.0),
        Point2::new(-2.0, 0.0),
        Point2::new(1.0, -1.0),
    ]);

    let halfplane = HalfPlane::from_point_and_normal(
        Point2::new(2.0, 2.0),
        Unit::new_normalize(Vector2::new(-4.0, 3.0)),
    );

    let intersection = triangle.intersect_halfplane(&halfplane);

    assert_eq!(intersection.vertices().len(), 4);
    assert_approx_matrix_eq!(
        intersection.vertices()[0],
        Point2::new(0.0, 3.0),
        abstol = 1e-12
    );
    assert_approx_matrix_eq!(
        intersection.vertices()[1],
        Point2::new(-2.0, 0.0),
        abstol = 1e-12
    );
    assert_approx_matrix_eq!(
        intersection.vertices()[2],
        Point2::new(0.0, -2.0 / 3.0),
        abstol = 1e-12
    );
    assert_approx_matrix_eq!(
        intersection.vertices()[3],
        Point2::new(0.6875, 0.25),
        abstol = 1e-12
    );
}

#[test]
#[ignore]
/// TODO: Make this test pass!
fn triangle_intersect_box_vertex_intersection() {
    // The vertices of the triangle exactly lie on the edges of the box
    let a = Point2::new(1.0, 1.0);
    let c = Point2::new(3.0, 0.0);
    let d = Point2::new(2.0, -2.0);
    let triangle_poly = ConvexPolygon::from_vertices(vec![d, c, a]);
    let box_poly = ConvexPolygon::from_vertices(vec![
        Point2::new(0.0, -2.0),
        Point2::new(3.0, -2.0),
        Point2::new(3.0, 1.0),
        Point2::new(0.0, 1.0),
    ]);

    let intersection = box_poly.intersect_polygon(&triangle_poly);
    dbg!(&intersection);
    assert_eq!(intersection.vertices().len(), 3);
}

#[test]
fn triangle_triangle_intersection() {
    let triangle1 = ConvexPolygon::from_vertices(vec![
        Point2::new(0.0, 3.0),
        Point2::new(-2.0, 0.0),
        Point2::new(1.0, -1.0),
    ]);

    let triangle2 = ConvexPolygon::from_vertices(vec![
        Point2::new(-2.0, 1.0),
        Point2::new(-1.0, -1.0),
        Point2::new(2.0, 2.0),
    ]);

    let intersection = triangle1.intersect_polygon(&triangle2);

    assert_eq!(intersection.vertices().len(), 6);
    let v = intersection.vertices();
    assert_approx_matrix_eq!(v[0], Point2::new(-1.2, 1.2), abstol = 1e-12);
    assert_approx_matrix_eq!(
        v[1],
        Point2::new(-1.714285714285714, 0.428571428571429),
        abstol = 1e-12
    );
    assert_approx_matrix_eq!(v[2], Point2::new(-1.4, -0.2), abstol = 1e-12);
    assert_approx_matrix_eq!(v[3], Point2::new(-0.5, -0.5), abstol = 1e-12);
    assert_approx_matrix_eq!(v[4], Point2::new(0.6, 0.6), abstol = 1e-12);
    assert_approx_matrix_eq!(
        v[5],
        Point2::new(0.352941176470588, 1.588235294117647),
        abstol = 1e-12
    );
}

#[test]
fn triangulate() {
    let a = Point2::new(2.0, 0.0);
    let b = Point2::new(6.0, 4.0);
    let c = Point2::new(4.0, 6.0);
    let d = Point2::new(1.0, 5.0);
    let e = Point2::new(1.0, 2.0);

    {
        // Empty
        let poly = ConvexPolygon::<f64>::from_vertices(Vec::new());
        assert!(poly.triangulate_into_vec().is_empty());
    }

    {
        // Point
        let poly = ConvexPolygon::from_vertices(vec![a]);
        assert!(poly.triangulate_into_vec().is_empty());
    }

    {
        // Line segment
        let poly = ConvexPolygon::from_vertices(vec![a, b]);
        assert!(poly.triangulate_into_vec().is_empty());
    }

    {
        // Triangle
        let poly = ConvexPolygon::from_vertices(vec![a, b, c]);
        assert_eq!(poly.triangulate_into_vec(), vec![Triangle([a, b, c])]);
    }

    {
        // Quad
        let poly = ConvexPolygon::from_vertices(vec![a, b, c, d]);
        assert_eq!(
            poly.triangulate_into_vec(),
            vec![Triangle([a, b, c]), Triangle([a, c, d])]
        );
    }

    {
        // Pentagon
        let poly = ConvexPolygon::from_vertices(vec![a, b, c, d, e]);
        assert_eq!(
            poly.triangulate_into_vec(),
            vec![
                Triangle([a, b, c]),
                Triangle([a, c, d]),
                Triangle([a, d, e])
            ]
        )
    }
}

#[test]
fn line_segment_intersect_segment_parametric() {
    let segment1 = LineSegment2d::new(Point2::new(2.0, 3.0), Point2::new(3.0, 0.0));
    let segment2 = LineSegment2d::new(Point2::new(3.0, 1.0), Point2::new(3.0, 4.0));
    assert_eq!(segment1.intersect_segment_parametric(&segment2), None);
}

#[test]
fn line_segment_intersect_polygon() {
    let a = Point2::new(2.0, 3.0);
    let b = Point2::new(3.0, 0.0);
    let segment = LineSegment2d::new(a, b);

    let polygon = ConvexPolygon::from_vertices(vec![
        Point2::new(0.0, 1.0),
        Point2::new(3.0, 1.0),
        Point2::new(3.0, 4.0),
        Point2::new(0.0, 4.0),
    ]);

    let result = segment
        .intersect_polygon(&polygon)
        .expect("Intersection is not empty");
    let expected_intersection =
        LineSegment2d::new(Point2::new(2.0, 3.0), Point2::new(8.0 / 3.0, 1.0));

    // The line segment may be defined in two ways, but its midpoint and length uniquely
    // defines its shape
    assert_approx_matrix_eq!(
        result.midpoint(),
        expected_intersection.midpoint(),
        abstol = 1e-12
    );
    assert_scalar_eq!(
        result.length(),
        expected_intersection.length(),
        comp = abs,
        tol = 1e-12
    );
}

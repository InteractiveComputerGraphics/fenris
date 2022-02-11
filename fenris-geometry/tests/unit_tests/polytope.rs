use fenris_geometry::proptest::half_plane;
use fenris_geometry::{assert_line_segments_approx_equal, prop_assert_line_segments_approx_equal};
use fenris_geometry::{ConvexPolygon, HalfPlane, Line2d, LineSegment2d, Triangle};
use nalgebra::{point, vector, Point2, Unit, Vector2};
use proptest::prelude::*;
use util::assert_approx_matrix_eq;

#[test]
fn half_plane_surface_distance_and_contains_point() {
    let x0 = Point2::new(1.0, -1.0);
    let n = Unit::new_normalize(Vector2::new(1.0, -1.0));
    let half_plane = HalfPlane::from_point_and_normal(x0, n);

    {
        let x = Point2::new(-1.0, 1.0);
        let dist = half_plane.signed_distance_to_point(&x);
        let expected = -2.828427124746;
        let diff: f64 = dist - expected;

        assert!(diff.abs() < 1e-6);
        assert!(half_plane.contains_point(&x));
    }

    {
        let x = Point2::new(2.0, 1.0);
        let dist = half_plane.signed_distance_to_point(&x);
        let expected = -0.7071067811865;
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
    let n = Unit::new_normalize(Vector2::new(1.0, -1.0));
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
    let n = Unit::new_normalize(Vector2::new(1.0, -1.0));
    let half_plane = HalfPlane::from_point_and_normal(x0, n);

    // Line represented as polygon intersecting the surface of the halfplane
    {
        let x1 = Point2::new(-1.0, 1.0);
        let x2 = Point2::new(2.0, -1.0);
        let poly = ConvexPolygon::from_vertices(vec![x1, x2]);
        let intersection = poly.intersect_halfplane(&half_plane);
        let expected = ConvexPolygon::from_vertices(vec![x1, Point2::new(1.4, -0.6)]);

        assert_approx_matrix_eq!(intersection.vertices()[0], expected.vertices()[0], abstol = 1e-6);
        assert_approx_matrix_eq!(intersection.vertices()[1], expected.vertices()[1], abstol = 1e-6);
    }
}

#[test]
fn line_line_intersection() {
    let line1 = Line2d::from_point_and_dir(Point2::new(0.0, -1.0), Vector2::new(1.0, 1.0).normalize());
    let line2 = Line2d::from_point_and_dir(Point2::new(-2.0, 2.0), Vector2::new(4.0, -2.0).normalize());

    let intersection = line1.intersect(&line2).expect("Intersection exists");

    assert_approx_matrix_eq!(intersection, Point2::new(4.0 / 3.0, 1.0 / 3.0), abstol = 1e-6);
}

#[test]
fn triangle_polygon_intersect_halfplane() {
    let triangle = ConvexPolygon::from_vertices(vec![
        Point2::new(0.0, 3.0),
        Point2::new(-2.0, 0.0),
        Point2::new(1.0, -1.0),
    ]);

    let halfplane =
        HalfPlane::from_point_and_normal(Point2::new(2.0, 2.0), Unit::new_normalize(Vector2::new(4.0, -3.0)));

    let intersection = triangle.intersect_halfplane(&halfplane);

    assert_eq!(intersection.vertices().len(), 4);
    assert_approx_matrix_eq!(intersection.vertices()[0], Point2::new(0.0, 3.0), abstol = 1e-12);
    assert_approx_matrix_eq!(intersection.vertices()[1], Point2::new(-2.0, 0.0), abstol = 1e-12);
    assert_approx_matrix_eq!(intersection.vertices()[2], Point2::new(0.0, -2.0 / 3.0), abstol = 1e-12);
    assert_approx_matrix_eq!(intersection.vertices()[3], Point2::new(0.6875, 0.25), abstol = 1e-12);
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
    assert_approx_matrix_eq!(v[1], Point2::new(-1.714285714285714, 0.428571428571429), abstol = 1e-12);
    assert_approx_matrix_eq!(v[2], Point2::new(-1.4, -0.2), abstol = 1e-12);
    assert_approx_matrix_eq!(v[3], Point2::new(-0.5, -0.5), abstol = 1e-12);
    assert_approx_matrix_eq!(v[4], Point2::new(0.6, 0.6), abstol = 1e-12);
    assert_approx_matrix_eq!(v[5], Point2::new(0.352941176470588, 1.588235294117647), abstol = 1e-12);
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
            vec![Triangle([a, b, c]), Triangle([a, c, d]), Triangle([a, d, e])]
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
fn line_segment_intersect_half_plane() {
    let a = point![1.0, 2.0];
    let b = point![2.0, 1.0];
    let intersection_point = point![1.6, 1.4];
    let segment = LineSegment2d::new(a, b);
    let half_plane = {
        let half_plane_point = point![1.0, 1.0];
        let normal = Unit::new_normalize(vector![-0.8, 1.2]);
        HalfPlane::from_point_and_normal(half_plane_point, normal)
    };

    let intersection = segment.intersect_half_plane(&half_plane).unwrap();
    let expected = LineSegment2d::new(b, intersection_point);
    assert_line_segments_approx_equal!(intersection, expected, abstol = 1e-14);
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
    let expected_intersection = LineSegment2d::new(Point2::new(2.0, 3.0), Point2::new(8.0 / 3.0, 1.0));
    assert_line_segments_approx_equal!(result, expected_intersection, abstol = 1e-12);
}

#[derive(Debug, Clone)]
struct LineSegment2dHalfPlaneIntersection {
    // The intersection point may not be needed for a particular test, but it's still useful information when debugging
    #[allow(dead_code)]
    pub intersection_point: Point2<f64>,
    pub half_plane: HalfPlane<f64>,
    pub input_segment: LineSegment2d<f64>,
    pub output_segment: LineSegment2d<f64>,
}

proptest! {
    #[test]
    fn line_segment_2d_half_plane_intersection(problem in intersecting_line_segment_2d_and_half_plane()) {
        let intersection = problem.input_segment.intersect_half_plane(&problem.half_plane)
            .expect("The intersection is by design non-empty");
        prop_assert_line_segments_approx_equal!(intersection, problem.output_segment, abstol=1e-9);
    }

    #[test]
    fn line_segment_2d_contained_in_half_plane_intersection(
        (segment, half_plane) in line_segment_2d_contained_in_half_plane())
    {
        let intersection = segment.intersect_half_plane(&half_plane).unwrap();
        prop_assert_line_segments_approx_equal!(intersection, segment, abstol=1e-9);
    }

    #[test]
    fn disjoint_line_segment_2d_and_half_plane_intersection(
        (segment, half_plane) in disjoint_line_segment_2d_and_half_plane())
    {
        prop_assert!(segment.intersect_half_plane(&half_plane).is_none());
    }
}

/// A strategy for generating half planes and line segments whose intersection is a subset of the input line segment.
fn intersecting_line_segment_2d_and_half_plane() -> impl Strategy<Value = LineSegment2dHalfPlaneIntersection> {
    // Given a half plane represented by a point x0 on its surface and an outward-facing normal n,
    // we let t be a vector tangent to the plane, and define an intersection point as
    //  x_i = x_0 + t_i * t
    // for any scalar t_i.
    //
    // Next, we pick a point *inside* the half-plane as
    //  x1 = x_0 + t1 * t + n1 * n
    // where t1 is a scalar and n1 < 0 ensures that the point is on the "inside" of the half-plane.
    //
    // Finally, we pick a scalar alpha >= 0 and define
    //  x2 = x_i + alpha * (x_i - x1)
    // so that x2 is a point *outside* the halfplane
    //
    // Let now L be the line segment pointing from x1 to x2. Then the intersection of L with the half plane is given
    // by the line segment pointing from x1 to x_i.
    let scalar = -10.0..10.0;
    let negative_scalar = -1.0..-1e-3;
    let non_negative_scalar = 0.0..10.0;
    (
        half_plane(),
        scalar.clone(),
        scalar.clone(),
        negative_scalar,
        non_negative_scalar,
        proptest::bool::ANY,
    )
        .prop_map(|(half_plane, t_i, t1, n1, alpha, should_flip)| {
            let x0 = half_plane.point();
            let n = half_plane.normal();
            let ref t = half_plane.surface().tangent();
            let ref x_i = half_plane.point() + t_i * t;
            let x1 = x0 + t1 * t + n1 * n;
            let x2 = x_i + alpha * (x_i - x1);
            // Randomly flip the order of vertices in the output in order to avoid bias in representation
            let output_segment = if should_flip {
                LineSegment2d::new(*x_i, x1)
            } else {
                LineSegment2d::new(x1, *x_i)
            };

            LineSegment2dHalfPlaneIntersection {
                intersection_point: half_plane.point() + t_i * t,
                input_segment: LineSegment2d::new(x1, x2),
                output_segment,
                half_plane,
            }
        })
}

/// A strategy for generating line segments and half planes whose intersection is empty.
fn disjoint_line_segment_2d_and_half_plane() -> impl Strategy<Value = (LineSegment2d<f64>, HalfPlane<f64>)> {
    line_segment_2d_and_half_plane(-10.0..10.0, 1e-3..10.0)
}

/// A strategy for generating line segments and half planes where the line segment is entirely contained
/// in the half plane.
fn line_segment_2d_contained_in_half_plane() -> impl Strategy<Value = (LineSegment2d<f64>, HalfPlane<f64>)> {
    line_segment_2d_and_half_plane(-10.0..10.0, -10.0..-1e-3)
}

fn line_segment_2d_and_half_plane(
    tangent_coord: impl Strategy<Value = f64> + Clone,
    normal_coord: impl Strategy<Value = f64> + Clone,
) -> impl Strategy<Value = (LineSegment2d<f64>, HalfPlane<f64>)> {
    // Given a half plane defined by a point x0 and normal n, we let t be a unit vector tangent to the plane.
    // Then we can express points in space by the relation
    //  x = x0 + alpha * t + beta * n
    // Hence, by choosing the sign of beta, we can decide on whether points should be inside/outside the half-plane.
    let alpha = tangent_coord;
    let beta = normal_coord;
    (half_plane(), alpha.clone(), beta.clone(), alpha.clone(), beta.clone()).prop_map(
        |(half_plane, alpha1, beta1, alpha2, beta2)| {
            let x0 = half_plane.point();
            let n = half_plane.normal();
            let t = half_plane.surface().tangent();
            let x1 = x0 + alpha1 * t + beta1 * n;
            let x2 = x0 + alpha2 * t + beta2 * n;
            let segment = LineSegment2d::new(x1, x2);
            (segment, half_plane)
        },
    )
}

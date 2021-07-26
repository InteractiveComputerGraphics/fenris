use fenris_geometry::{ConvexPolyhedron, Distance, Hexahedron, SignedDistance, Tetrahedron, Triangle};
use matrixcompare::assert_scalar_eq;
use nalgebra::{Point2, Point3, Vector2};
use util::assert_approx_matrix_eq;

#[test]
fn triangle_signed_distance_and_distance() {
    let triangle = Triangle([Point2::new(1.0, 2.0), Point2::new(4.0, 0.0), Point2::new(3.0, 3.0)]);

    // Outside, closest to edge 0
    {
        let p = Point2::new(1.0, 0.0);
        let sdf = triangle.query_signed_distance(&p).unwrap();
        assert_eq!(sdf.feature_id, 0);
        assert_approx_matrix_eq!(
            sdf.point.coords,
            Vector2::new(1.9230769230769, 1.3846153846154),
            abstol = 1e-10
        );
        assert_scalar_eq!(sdf.signed_distance, 1.6641005886757, comp = abs, tol = 1e-10);
        assert_scalar_eq!(triangle.distance(&p), 1.6641005886757, comp = abs, tol = 1e-10);
    }

    // Outside, closest to vertex 1
    {
        let p = Point2::new(5.0, 0.0);
        let sdf = triangle.query_signed_distance(&p).unwrap();
        // Closest edge is ambiguous, can be either 0 or 1
        assert!([0, 1].contains(&sdf.feature_id));
        assert_approx_matrix_eq!(sdf.point.coords, Vector2::new(4.0, 0.0), abstol = 1e-10);
        assert_scalar_eq!(sdf.signed_distance, 1.0, comp = abs, tol = 1e-10);
        assert_scalar_eq!(triangle.distance(&p), 1.0, comp = abs, tol = 1e-10);
    }

    // Outside, closest to edge 1
    {
        let p = Point2::new(4.0, 3.0);
        let sdf = triangle.query_signed_distance(&p).unwrap();
        // Closest edge is ambiguous, can be either 0 or 1
        assert_eq!(sdf.feature_id, 1);
        assert_approx_matrix_eq!(sdf.point.coords, Vector2::new(3.1, 2.7), abstol = 1e-10);
        assert_scalar_eq!(sdf.signed_distance, 0.9486832980505, comp = abs, tol = 1e-10);
        assert_scalar_eq!(triangle.distance(&p), 0.9486832980505, comp = abs, tol = 1e-10);
    }

    // Outside, closest to edge 2
    {
        let p = Point2::new(2.0, 3.0);
        let sdf = triangle.query_signed_distance(&p).unwrap();
        // Closest edge is ambiguous, can be either 0 or 1
        assert_eq!(sdf.feature_id, 2);
        assert_approx_matrix_eq!(sdf.point.coords, Vector2::new(2.2, 2.6), abstol = 1e-10);
        assert_scalar_eq!(sdf.signed_distance, 0.4472135955, comp = abs, tol = 1e-10);
        assert_scalar_eq!(triangle.distance(&p), 0.4472135955, comp = abs, tol = 1e-10);
    }

    // Inside, closest to edge 0
    {
        let p = Point2::new(3.0, 1.0);
        let sdf = triangle.query_signed_distance(&p).unwrap();
        // Closest edge is ambiguous, can be either 0 or 1
        assert_eq!(sdf.feature_id, 0);
        assert_approx_matrix_eq!(
            sdf.point.coords,
            Vector2::new(2.8461538461538, 0.7692307692308),
            abstol = 1e-10
        );
        assert_scalar_eq!(sdf.signed_distance, -0.2773500981126, comp = abs, tol = 1e-10);
        assert_eq!(triangle.distance(&p), 0.0);
    }

    // Inside, closest to edge 1
    {
        let p = Point2::new(3.0, 2.0);
        let sdf = triangle.query_signed_distance(&p).unwrap();
        // Closest edge is ambiguous, can be either 0 or 1
        assert_eq!(sdf.feature_id, 1);
        assert_approx_matrix_eq!(sdf.point.coords, Vector2::new(3.3, 2.1), abstol = 1e-10);
        assert_scalar_eq!(sdf.signed_distance, -0.3162277660168, comp = abs, tol = 1e-10);
        assert_eq!(triangle.distance(&p), 0.0);
    }

    // Inside, closest to edge 2
    {
        let p = Point2::new(2.0, 2.0);
        let sdf = triangle.query_signed_distance(&p).unwrap();
        // Closest edge is ambiguous, can be either 0 or 1
        assert_eq!(sdf.feature_id, 2);
        assert_approx_matrix_eq!(sdf.point.coords, Vector2::new(1.8, 2.4), abstol = 1e-10);
        assert_scalar_eq!(sdf.signed_distance, -0.4472135955, comp = abs, tol = 1e-10);
        assert_eq!(triangle.distance(&p), 0.0);
    }
}

#[test]
fn cube_polyhedron_signed_distance() {
    let cube = Hexahedron::from_vertices([
        Point3::new(-1.0, -1.0, -1.0),
        Point3::new(1.0, -1.0, -1.0),
        Point3::new(1.0, 1.0, -1.0),
        Point3::new(-1.0, 1.0, -1.0),
        Point3::new(-1.0, -1.0, 1.0),
        Point3::new(1.0, -1.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(-1.0, 1.0, 1.0),
    ]);

    // First test points on the outside, one for each face
    {
        // Face 0
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-0.5, -0.5, -1.6))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-0.5, -0.5, -1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, 0.6, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 0);
        }

        // Face 1
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-0.5, -1.3, 0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-0.5, -1.0, 0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, 0.3, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 1);
        }

        // Face 2
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(1.5, 0.5, -0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(1.0, 0.5, -0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, 0.5, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 2);
        }

        // Face 3
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-0.5, 1.4, 0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-0.5, 1.0, 0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, 0.4, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 3);
        }

        // Face 4
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-1.5, -0.5, -0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-1.0, -0.5, -0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, 0.5, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 4);
        }

        // Face 5
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(0.5, -0.5, 1.2))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(0.5, -0.5, 1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, 0.2, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 5);
        }
    }

    // Then test points outside, whose closest point is a vertex
    // (i.e. in the Voronoi regions of the vertices)
    {
        // Note: The face given as "feature_id" is somewhat arbitrary, but it must be
        // one of the faces that contain the vertex

        // Vertex 0
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-2.0, -2.0, -2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-1.0, -1.0, -1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([0, 1, 4].contains(&sdf_result.feature_id));
        }

        // Vertex 1
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(2.0, -2.0, -2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(1.0, -1.0, -1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([0, 1, 2].contains(&sdf_result.feature_id));
        }

        // Vertex 2
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(2.0, 2.0, -2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(1.0, 1.0, -1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([0, 2, 3].contains(&sdf_result.feature_id));
        }

        // Vertex 3
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-2.0, 2.0, -2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-1.0, 1.0, -1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([0, 3, 4].contains(&sdf_result.feature_id));
        }

        // Vertex 4
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-2.0, -2.0, 2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-1.0, -1.0, 1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([1, 4, 5].contains(&sdf_result.feature_id));
        }

        // Vertex 5
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(2.0, -2.0, 2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(1.0, -1.0, 1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([1, 2, 5].contains(&sdf_result.feature_id));
        }

        // Vertex 6
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(2.0, 2.0, 2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(1.0, 1.0, 1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([2, 3, 5].contains(&sdf_result.feature_id));
        }

        // Vertex 7
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-2.0, 2.0, 2.0))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-1.0, 1.0, 1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, f64::sqrt(3.0), comp = abs, tol = 1e-12);
            assert!([3, 4, 5].contains(&sdf_result.feature_id));
        }
    }

    // Test faces on the inside
    {
        // Face 0
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-0.5, -0.5, -0.9))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-0.5, -0.5, -1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, -0.1, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 0);
        }

        // Face 1
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-0.5, -0.8, 0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-0.5, -1.0, 0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, -0.2, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 1);
        }

        // Face 2
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(0.8, 0.5, -0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(1.0, 0.5, -0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, -0.2, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 2);
        }

        // Face 3
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-0.5, 0.9, 0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-0.5, 1.0, 0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, -0.1, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 3);
        }

        // Face 4
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(-0.8, -0.5, -0.5))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(-1.0, -0.5, -0.5), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, -0.2, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 4);
        }

        // Face 5
        {
            let sdf_result = cube
                .query_signed_distance(&Point3::new(0.5, -0.5, 0.7))
                .unwrap();
            assert_approx_matrix_eq!(sdf_result.point, Point3::new(0.5, -0.5, 1.0), abstol = 1e-12);
            assert_scalar_eq!(sdf_result.signed_distance, -0.3, comp = abs, tol = 1e-12);
            assert_eq!(sdf_result.feature_id, 5);
        }
    }
}

#[test]
fn convex_polygon_3d_compute_volume() {
    let tetrahedron = Tetrahedron::<f64>::reference();
    assert_scalar_eq!(tetrahedron.compute_volume(), 4.0 / 3.0, comp = abs, tol = 1e-12);

    let hexahedron = Hexahedron::<f64>::reference();
    assert_scalar_eq!(hexahedron.compute_volume(), 8.0, comp = abs, tol = 1e-12);
}

use fenris_geometry::{SimplePolygon2d, LineSegment2d, Polygon2d, SimplePolygon3d, HalfSpace};
use matrixcompare::{assert_matrix_eq, assert_scalar_eq};
use nalgebra::{clamp, Isometry3, point, Point2, Point3, vector, Vector2, Vector3};

use std::f64;
use std::f64::consts::PI;
use std::fs::create_dir_all;
use std::path::Path;
use itertools::izip;
use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand::rngs::{StdRng};
use rand_distr::{Normal};
use fenris::eyre;
use fenris::vtkio::model::{ByteOrder, DataSet, PolyDataPiece, Version, VertexNumbers};
use fenris::vtkio::Vtk;
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

    let polygon = SimplePolygon2d::from_vertices(vertices.clone());

    let expected_area = 10.5;
    assert_scalar_eq!(polygon.signed_area(), expected_area, comp = abs, tol = 1e-12);
    assert_scalar_eq!(polygon.area(), expected_area, comp = abs, tol = 1e-12);

    let vertices_reversed = {
        let mut v = vertices.clone();
        v.reverse();
        v
    };

    let reversed_polygon = SimplePolygon2d::from_vertices(vertices_reversed);

    assert_scalar_eq!(reversed_polygon.signed_area(), -expected_area, comp = abs, tol = 1e-12);
    assert_scalar_eq!(reversed_polygon.area(), expected_area, comp = abs, tol = 1e-12);
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

    let polygon = SimplePolygon2d::from_vertices(vertices.clone());

    {
        // Segment is outside (and also outside its convex hull)
        let segment = LineSegment2d::from_end_points(Point2::new(-8.0, -1.0), Point2::new(-7.0, 3.0));
        assert_eq!(polygon.intersects_segment(&segment), false);
    }

    {
        // Segment is outside (but inside the convex hull of the polygon)
        let segment = LineSegment2d::from_end_points(Point2::new(-3.0, 0.0), Point2::new(-2.0, 1.0));
        assert_eq!(polygon.intersects_segment(&segment), false);
    }

    {
        // Segment is completely inside the polygon
        let segment = LineSegment2d::from_end_points(Point2::new(-3.0, -2.0), Point2::new(-5.0, 0.0));
        assert_eq!(polygon.intersects_segment(&segment), true);
    }

    {
        // Segment is partially inside, with one of its endpoints inside
        let segment = LineSegment2d::from_end_points(Point2::new(-3.0, -2.0), Point2::new(-4.0, 0.5));
        assert_eq!(polygon.intersects_segment(&segment), true);
    }

    {
        // Segment is partially inside, with none of its endpoints inside
        let segment = LineSegment2d::from_end_points(Point2::new(0.0, -1.0), Point2::new(-6.0, 0.0));
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

    let polygon = SimplePolygon2d::from_vertices(vertices.clone());

    {
        // Point is outside, but inside the convex hull of the polygon
        let point = Point2::new(-3.0, 0.0);
        let closest_edge = polygon.closest_edge(&point).unwrap();
        let expected_t = f64::sqrt(0.5) / 2.82842712474619;

        assert_eq!(closest_edge.edge_index, 3);
        assert_scalar_eq!(closest_edge.signed_distance, f64::sqrt(0.5), comp = abs, tol = 1e-12);
        assert_scalar_eq!(closest_edge.edge_parameter, expected_t, comp = abs, tol = 1e-12);
        assert_approx_matrix_eq!(closest_edge.edge_point.coords, Vector2::new(-3.5, -0.5), abstol = 1e-12);
    }

    {
        // Point is exactly on the boundary
        let point = Point2::new(-4.5, 1.0);
        let closest_edge = polygon.closest_edge(&point).unwrap();
        let expected_t = 0.25;

        assert_eq!(closest_edge.edge_index, 4);
        assert_scalar_eq!(closest_edge.signed_distance, 0.0, comp = abs, tol = 1e-12);
        assert_scalar_eq!(closest_edge.edge_parameter, expected_t, comp = abs, tol = 1e-12);
        assert_approx_matrix_eq!(closest_edge.edge_point.coords, point.coords, abstol = 1e-12);
    }

    {
        // Point is inside, closest to a vertex
        let point = Point2::new(-3.1, -1.4);
        let closest_edge = polygon.closest_edge(&point).unwrap();

        // Whether edge 2 or 3 is reported is not well-defined. It can be either.
        assert!([2, 3].contains(&closest_edge.edge_index));
        let expected_t = if closest_edge.edge_index == 2 { 1.0 } else { 0.0 };
        assert_scalar_eq!(
            closest_edge.signed_distance,
            -0.412310562561766,
            comp = abs,
            tol = 1e-12
        );
        assert_scalar_eq!(closest_edge.edge_parameter, expected_t, comp = abs, tol = 1e-12);
        assert_approx_matrix_eq!(closest_edge.edge_point.coords, vertices[3].coords, abstol = 1e-12);
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
        assert_scalar_eq!(closest_edge.edge_parameter, expected_t, comp = abs, tol = 1e-10);
        assert_approx_matrix_eq!(
            closest_edge.edge_point.coords,
            Vector2::new(-5.384615384615385, -0.076923076923077),
            abstol = 1e-12
        );
    }
}

#[test]
fn simple_polygon_3d_area_simple_example() {
    // polygon in xy-plane (so 2D polygon)
    {
        let poly = SimplePolygon3d::from_vertices(vec![
            point![0.0, 0.0, 0.0],
            point![1.0, 0.0, 0.0],
            point![0.0, 1.0, 0.0]
        ]);
        assert_matrix_eq!(poly.area_vector(), 0.5 * Vector3::z());
    }

    // polygon not in xy-plane (or any of the other standard axis-aligned planes)
    {
        let poly = SimplePolygon3d::from_vertices(vec![
            point![0.0, 0.0, 0.0],
            point![0.0, 1.0, 0.0],
            point![1.0, 1.0, 1.0],
        ]);

        let expected_area = 1.0 / f64::sqrt(2.0);
        let expected_normal = vector![1.0, 0.0, -1.0].normalize();
        assert_scalar_eq!(poly.area(), expected_area, comp = abs, tol = 1e-14);
        assert_matrix_eq!(poly.area_vector(), expected_area * expected_normal, comp = abs, tol = 1e-14);
    }
}

#[test]
fn simple_polygon_3d_area_vector_random_examples() {
    let dir = Path::new("output/tests/geometry/polygon/area_vector/");
    // TODO: Use reliable rng
    let mut rng = StdRng::seed_from_u64(2094583429058094235);
    let num_polygons = 200;

    for i in 0 .. num_polygons {
        let poly2d = generate_random_simple_polygon_2d(&mut rng);
        let expected_area = poly2d.area();

        // Due to the numbers being generated, the area of the polygon should be within a
        // few orders of magnitude of 1, unless it happens to be degenerate
        // (in which case a relative comparison would fail)
        let tol = 1e-14;

        // Only export the first 200 samples, in order to avoid polluting the
        // output directory
        if i < 200 {
            let filename = format!("polygon_2d_{i}.vtk");
            let poly2d_as_3d = poly2d.apply_isometry(&Isometry3::identity());
            export_simple_polygon_3d_vtk(dir.join(filename), &poly2d_as_3d).unwrap();
        }

        let isometry = Standard.sample(&mut rng);
        let poly3d = poly2d.apply_isometry(&isometry);
        assert_scalar_eq!(poly3d.area(), expected_area, comp = abs, tol = tol);

        let normal = isometry * Vector3::z_axis();
        let expected_area_vector = normal.into_inner() * expected_area;
        assert_matrix_eq!(poly3d.area_vector(), expected_area_vector, comp = abs, tol = tol);
    }
}

#[test]
fn simple_polygon_3d_intersect_half_space() {
    // TODO: Need a way to automatically test this!
    let mut rng = StdRng::seed_from_u64(2094583429058094235);
    let num_polygons = 200;
    for i in 0 .. num_polygons {
        let mut polygon = generate_random_simple_polygon_3d(&mut rng);
        let halfspace = HalfSpace::from_point_and_normal(Point3::origin(), Vector3::x_axis());
        polygon.intersect_half_space(&halfspace);

        // Only export the first 200 samples, in order to avoid filling up our hard drive
        // for larger sample numbers
        if i < 200 {
            let dir = Path::new("output/tests/geometry/polygon/temp/");
            let filename = format!("polygon_{i}.vtk");
            let polygon_path = dir.join(filename);
            let halfspace_path = dir.join(format!("halfspace_{i}.vtk"));
            let intersection_path = dir.join(format!("polygon_intersection_{i}.vtk"));
            export_simple_polygon_3d_vtk(polygon_path, &polygon).unwrap();
            export_half_space_vtk(halfspace_path, &halfspace).unwrap();
            export_simple_polygon_3d_vtk(intersection_path, &polygon).unwrap();
        }
    }
}

fn generate_random_simple_polygon_2d(rng: &mut impl Rng) -> SimplePolygon2d<f64> {
    // Only capable of generating star-shaped polygons
    let n_max = 20;
    let n = rng.gen_range(3 .. n_max);

    let mut r_values = vec![0.0];
    let mut theta_values = vec![0.0];

    let delta_mean = 2.0 * PI / (n as f64);
    let std_dev = delta_mean / 2.0;
    let delta_dist = Normal::new(delta_mean, std_dev).unwrap();

    for i in 1 .. n {
        let prev_theta = theta_values[i - 1];
        let delta = delta_dist.sample(rng);
        let delta = clamp(delta, 0.0, PI);
        let theta = f64::min(prev_theta + delta, 2.0 * PI);
        theta_values.push(theta);
        r_values.push(rng.gen_range(0.0 .. 1.0));
    }

    let vertices = izip!(r_values, theta_values)
        .map(|(r, theta)| point![ r * theta.cos(), r * theta.sin() ])
        .collect();

    SimplePolygon2d::from_vertices(vertices)
}

fn generate_random_simple_polygon_3d(rng: &mut impl Rng) -> SimplePolygon3d<f64> {
    let isometry: Isometry3<f64> = Standard.sample(rng);
    generate_random_simple_polygon_2d(rng)
        .apply_isometry(&isometry)
}

fn simple_polygon_3d_vtk_data_set(polygon: &SimplePolygon3d<f64>) -> DataSet {
    let num_verts = polygon.vertices().len();
    let mut vertex_buffer = Vec::with_capacity(num_verts * 3);

    // Note: We export as lines instead of polys to avoid cases where ParaView
    // (even after triangulation) does not correctly render the Polygon
    let mut connectivity = Vec::with_capacity(num_verts);
    let mut lines_offsets = Vec::with_capacity(num_verts);
    let mut poly_offsets = Vec::with_capacity(1);

    for (idx_current, v) in polygon.vertices().iter().enumerate() {
        let idx_next = (idx_current + 1) % num_verts;
        vertex_buffer.extend_from_slice(v.coords.as_slice());
        connectivity.extend_from_slice(&[idx_current as u64, idx_next as u64]);
        lines_offsets.push(connectivity.len() as u64);
    }
    poly_offsets.push(connectivity.len() as u64);

    let piece = PolyDataPiece {
        points: vertex_buffer.into(),
        verts: None,
        lines: Some(VertexNumbers::XML {
            connectivity: connectivity.clone(),
            offsets: lines_offsets
        }),
        polys: Some(VertexNumbers::XML {
            connectivity,
            offsets: poly_offsets
        }),
        strips: None,
        data: Default::default()
    };

    DataSet::from(piece)
}

fn export_simple_polygon_3d_vtk(path: impl AsRef<Path>, polygon: &SimplePolygon3d<f64>)
    -> eyre::Result<()> {
    let path = path.as_ref();
    let data_set = simple_polygon_3d_vtk_data_set(polygon);

    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }

    Vtk {
        version: Version { major: 1, minor: 0 },
        title: "Polygon".to_string(),
        byte_order: ByteOrder::BigEndian,
        data: data_set,
        file_path: None
    }.export_ascii(path)?;

    Ok(())
}

fn export_half_space_vtk(path: impl AsRef<Path>, half_space: &HalfSpace<f64>) -> eyre::Result<()> {
    let [t1, t2] = half_space.plane().compute_tangent_vectors().map(|t| t.into_inner());
    let p = half_space.point();
    let e = 5.0;
    let corners = vec![
        p + e * t1 - e * t2,
        p + e * t1 + e * t2,
        p - e * t1 + e * t2,
        p - e * t1 - e * t2,
    ];
    let polygon = SimplePolygon3d::from_vertices(corners);
    export_simple_polygon_3d_vtk(path, &polygon)
}
use fenris::mesh::procedural::create_unit_box_uniform_tet_mesh_3d;
use fenris_geometry::{compute_winding_number_for_triangles_3d, Triangle};
use matrixcompare::assert_scalar_eq;
use nalgebra::point;

// TODO: Move other triangle tests over here

#[test]
fn test_solid_angle() {
    let a = point![3.0, 4.0, 5.0];
    let b = point![2.0, 2.0, 1.0];
    let c = point![-1.0, -2.0, 4.0];
    let triangle = Triangle([a, b, c]);
    let p = point![5.0, 3.0, 1.0];
    let solid_angle = triangle.compute_solid_angle(&p);

    assert_scalar_eq!(solid_angle, 0.3079885663640642, comp = float);
}

#[test]
fn winding_number_closed_mesh() {
    let triangle_mesh = create_unit_box_uniform_tet_mesh_3d::<f64>(4).extract_surface_mesh();

    let inside_points = [
        point![0.1, 0.1, 0.1],
        point![0.9, 0.2, 0.8],
        point![0.999, 0.999, 0.00001],
    ];

    for point in inside_points {
        let w = compute_winding_number_for_triangles_3d(triangle_mesh.cell_iter(), &point);
        assert_scalar_eq!(w, 1.0, comp = float, ulp = 1024);
    }

    let outside_points = [
        point![0.001, 0.5, 1.1],
        point![2.0, 3.0, 5.0],
        point![-0.1, -0.2, 0.3],
        point![0.999, 0.999, 1.001],
    ];

    for point in outside_points {
        let w = compute_winding_number_for_triangles_3d(triangle_mesh.cell_iter(), &point);
        assert_scalar_eq!(w, 0.0, comp = float, ulp = 1024);
    }
}

use fenris::element::{ElementConnectivity, FiniteElement};
use fenris::mesh::procedural::create_unit_square_uniform_tri_mesh_2d;
use fenris::mesh::TriangleMesh2d;
use fenris::space::{FindClosestElement, FiniteElementSpace, SpatiallyIndexed};
use matrixcompare::assert_matrix_eq;
use nalgebra::Point2;

#[test]
fn spatially_indexed_closest_element_at_interfaces() {
    // Map points from points on the boundary of the reference element to physical space
    // for each element, then find the closest element and point
    let mesh: TriangleMesh2d<f64> = create_unit_square_uniform_tri_mesh_2d(10);
    let space = SpatiallyIndexed::from_space(mesh.clone());
    let boundary_points = [
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, 0.5],
        [0.5, -1.0],
        [0.0, 0.0],
    ]
    .map(Point2::from);

    for conn in mesh.connectivity() {
        let element = conn.element(mesh.vertices()).unwrap();
        for xi in &boundary_points {
            let x = element.map_reference_coords(xi);
            let (element_idx, ref_coords) = space.find_closest_element_and_reference_coords(&x).unwrap();
            let x_closest = space.map_element_reference_coords(element_idx, &ref_coords);
            assert_matrix_eq!(x.coords, x_closest.coords, comp = abs, tol = 1e-12);
        }
    }
}

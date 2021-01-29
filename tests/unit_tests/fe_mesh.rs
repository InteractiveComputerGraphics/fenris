use fenris::mesh::{Hex27Mesh, Mesh2d, Mesh3d};
use nalgebra::{Point2, Point3, Vector2, Vector3};

use fenris::connectivity::{
    Hex8Connectivity, Quad4d2Connectivity, Quad9d2Connectivity, Tri3d2Connectivity,
    Tri6d2Connectivity,
};
use util::assert_approx_matrix_eq;

#[test]
fn quad4_to_quad9_single_element_mesh() {
    let vertices = vec![
        Point2::new(2.0, 3.0),
        Point2::new(3.0, 3.0),
        Point2::new(3.0, 4.0),
        Point2::new(2.0, 4.0),
    ];
    let quad4 = Quad4d2Connectivity([0, 1, 2, 3]);
    let quad4_mesh = Mesh2d::from_vertices_and_connectivity(vertices.clone(), vec![quad4]);

    let quad9_mesh = Mesh2d::<f64, Quad9d2Connectivity>::from(quad4_mesh);
    let quad9_connectivity = quad9_mesh.connectivity().first().unwrap();

    assert_eq!(quad9_connectivity[0..4], [0, 1, 2, 3]);
    assert_eq!(quad9_connectivity[4..8], [4, 5, 6, 7]);
    assert_eq!(quad9_connectivity[8], 8);

    const EPS: f64 = 1e-12;
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[0].coords,
        vertices[0].coords,
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[1].coords,
        vertices[1].coords,
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[2].coords,
        vertices[2].coords,
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[3].coords,
        vertices[3].coords,
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[4].coords,
        Vector2::new(2.5, 3.0),
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[5].coords,
        Vector2::new(3.0, 3.5),
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[6].coords,
        Vector2::new(2.5, 4.0),
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[7].coords,
        Vector2::new(2.0, 3.5),
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        quad9_mesh.vertices()[8].coords,
        Vector2::new(2.5, 3.5),
        abstol = EPS
    );
}

#[test]
fn tri3_to_tri6_single_element_mesh() {
    let vertices = vec![
        Point2::new(2.0, 3.0),
        Point2::new(3.0, 3.0),
        Point2::new(3.0, 4.0),
    ];
    let tri3 = Tri3d2Connectivity([0, 1, 2]);
    let tri3_mesh = Mesh2d::from_vertices_and_connectivity(vertices.clone(), vec![tri3]);

    let tri6_mesh = Mesh2d::<f64, Tri6d2Connectivity>::from(tri3_mesh);
    let tri6_connectivity = tri6_mesh.connectivity().first().unwrap();

    assert_eq!(tri6_connectivity[0..3], [0, 1, 2]);
    assert_eq!(tri6_connectivity[3..6], [3, 4, 5]);

    const EPS: f64 = 1e-12;
    assert_approx_matrix_eq!(
        tri6_mesh.vertices()[0].coords,
        vertices[0].coords,
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        tri6_mesh.vertices()[1].coords,
        vertices[1].coords,
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        tri6_mesh.vertices()[2].coords,
        vertices[2].coords,
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        tri6_mesh.vertices()[3].coords,
        Vector2::new(2.5, 3.0),
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        tri6_mesh.vertices()[4].coords,
        Vector2::new(3.0, 3.5),
        abstol = EPS
    );
    assert_approx_matrix_eq!(
        tri6_mesh.vertices()[5].coords,
        Vector2::new(2.5, 3.5),
        abstol = EPS
    );
}

#[test]
fn hex8_to_hex27_single_element_mesh() {
    let vertices = vec![
        Point3::new(2.0, 3.0, 1.0),
        Point3::new(4.0, 3.0, 1.0),
        Point3::new(4.0, 4.0, 1.0),
        Point3::new(2.0, 4.0, 1.0),
        Point3::new(2.0, 3.0, 5.0),
        Point3::new(4.0, 3.0, 5.0),
        Point3::new(4.0, 4.0, 5.0),
        Point3::new(2.0, 4.0, 5.0),
    ];
    let hex8 = Hex8Connectivity([0, 1, 2, 3, 4, 5, 6, 7]);
    let hex8_mesh = Mesh3d::from_vertices_and_connectivity(vertices.clone(), vec![hex8]);

    let hex27_mesh = Hex27Mesh::from(&hex8_mesh);
    let hex27_connectivity = hex27_mesh.connectivity().first().unwrap();

    assert_eq!(hex27_connectivity.0[0..8], [0, 1, 2, 3, 4, 5, 6, 7]);
    // assert_eq!(tri6_connectivity[3..6], [3, 4, 5]);

    const EPS: f64 = 1e-12;

    let v = hex27_mesh.vertices();

    // Vertex nodes
    assert_approx_matrix_eq!(v[0].coords, vertices[0].coords, abstol = EPS);
    assert_approx_matrix_eq!(v[1].coords, vertices[1].coords, abstol = EPS);
    assert_approx_matrix_eq!(v[2].coords, vertices[2].coords, abstol = EPS);
    assert_approx_matrix_eq!(v[3].coords, vertices[3].coords, abstol = EPS);
    assert_approx_matrix_eq!(v[4].coords, vertices[4].coords, abstol = EPS);
    assert_approx_matrix_eq!(v[5].coords, vertices[5].coords, abstol = EPS);
    assert_approx_matrix_eq!(v[6].coords, vertices[6].coords, abstol = EPS);
    assert_approx_matrix_eq!(v[7].coords, vertices[7].coords, abstol = EPS);

    let edge_midpoint = |a: usize, b: usize| (vertices[a].coords + vertices[b].coords) / 2.0;
    let midpoint = |indices: &[usize]| {
        indices
            .iter()
            .copied()
            .map(|i| vertices[i])
            .fold(Vector3::zeros(), |acc, v| acc + v.coords)
            / (indices.len() as f64)
    };

    // Edge nodes
    assert_approx_matrix_eq!(v[8].coords, edge_midpoint(0, 1), abstol = EPS);
    assert_approx_matrix_eq!(v[9].coords, edge_midpoint(0, 3), abstol = EPS);
    assert_approx_matrix_eq!(v[10].coords, edge_midpoint(0, 4), abstol = EPS);
    assert_approx_matrix_eq!(v[11].coords, edge_midpoint(1, 2), abstol = EPS);
    assert_approx_matrix_eq!(v[12].coords, edge_midpoint(1, 5), abstol = EPS);
    assert_approx_matrix_eq!(v[13].coords, edge_midpoint(2, 3), abstol = EPS);
    assert_approx_matrix_eq!(v[14].coords, edge_midpoint(2, 6), abstol = EPS);
    assert_approx_matrix_eq!(v[15].coords, edge_midpoint(3, 7), abstol = EPS);
    assert_approx_matrix_eq!(v[16].coords, edge_midpoint(4, 5), abstol = EPS);
    assert_approx_matrix_eq!(v[17].coords, edge_midpoint(4, 7), abstol = EPS);
    assert_approx_matrix_eq!(v[18].coords, edge_midpoint(5, 6), abstol = EPS);
    assert_approx_matrix_eq!(v[19].coords, edge_midpoint(6, 7), abstol = EPS);

    // Face nodes
    assert_approx_matrix_eq!(v[20].coords, midpoint(&[0, 1, 2, 3]), abstol = EPS);
    assert_approx_matrix_eq!(v[21].coords, midpoint(&[0, 1, 5, 4]), abstol = EPS);
    assert_approx_matrix_eq!(v[22].coords, midpoint(&[0, 3, 7, 4]), abstol = EPS);
    assert_approx_matrix_eq!(v[23].coords, midpoint(&[1, 2, 6, 5]), abstol = EPS);
    assert_approx_matrix_eq!(v[24].coords, midpoint(&[2, 3, 6, 7]), abstol = EPS);
    assert_approx_matrix_eq!(v[25].coords, midpoint(&[4, 5, 6, 7]), abstol = EPS);

    // Center node
    assert_approx_matrix_eq!(
        v[26].coords,
        midpoint(&[0, 1, 2, 3, 4, 5, 6, 7]),
        abstol = EPS
    );
}

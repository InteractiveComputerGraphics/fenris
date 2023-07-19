use fenris::space::FixedInterpolator;
use nalgebra::{DVector, Vector2};

use proptest::prelude::*;

use matrixcompare::assert_scalar_eq;

#[test]
fn interpolate_into() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let supported_nodes = vec![
        0, 1, // Interpolation point 1
        0, 3, 4, // Interpolation point 2
        1, 2, 4, // Interpolation point 3
    ]; // Interpolation point 4
    let supported_node_offsets = vec![0, 2, 5, 8, 8];

    let interpolator = FixedInterpolator::from_compressed_values(
        values.into_iter().zip(supported_nodes).collect(),
        supported_node_offsets,
    );

    let u = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let mut interpolated_values = vec![Vector2::zeros(); 4];

    interpolator.interpolate_into(&mut interpolated_values, &u);

    let v = interpolated_values;
    assert_scalar_eq!(v[0].x, 7.0);
    assert_scalar_eq!(v[0].y, 10.0);
    assert_scalar_eq!(v[1].x, 76.0);
    assert_scalar_eq!(v[1].y, 88.0);
    assert_scalar_eq!(v[2].x, 125.0);
    assert_scalar_eq!(v[2].y, 146.0);
    assert_scalar_eq!(v[3].x, 0.0);
    assert_scalar_eq!(v[3].y, 0.0);
}

// TODO: Use this in rewritten tests?
// fn find_interior_point<T>(polygon: &ConvexPolygon<T>) -> Point2<T>
// where
//     T: Real,
// {
//     // Find an interior point by averaging all vertices (note: this is not in general the centroid)
//     let num_vertices = polygon.vertices().len();
//     let vertex_sum = polygon
//         .vertices()
//         .iter()
//         .map(|p| p.coords)
//         .fold(Vector2::zeros(), |a, b| a + b);
//     Point2::from(vertex_sum / T::from_usize(num_vertices).unwrap())
// }

proptest! {

    // TODO: Rewrite this test without NodalModel/Quad4Model
    // #[test]
    // fn finite_element_interpolator_is_identity_for_node_vertices(
    //     mesh in rectangular_uniform_mesh_strategy(1.0, 8))
    // {
    //     prop_assume!(mesh.connectivity().len() > 0);
    //
    //     let quadrature = quad_quadrature_strength_5_f64();
    //     let model = Quad4Model::from_mesh_and_quadrature(mesh.clone(), quadrature);
    //     let interpolation_vertices = mesh.vertices();
    //     let interpolator = model.make_interpolator(interpolation_vertices).unwrap();
    //
    //     // Assume that the solution variable is x, with x corresponding to the position
    //     // of vertices. So x_h(X) = sum_i N_i(X) x_i, where x_h(X) is the deformed
    //     // position at reference position X. Then, clearly, if x_i == X_i, with X_i
    //     // being a node in the Lagrange finite element basis, then
    //     // x_h(X_i) == X_i.
    //     let vertices: Vec<_> = mesh.vertices()
    //         .iter()
    //         .map(|x| x.coords)
    //         .collect();
    //
    //     let solution_variables = flatten_vertically(&vertices).unwrap();
    //     let mut result = vec![Vector2::zeros(); interpolation_vertices.len()];
    //
    //     interpolator.interpolate_into(&mut result, &solution_variables);
    //
    //     assert_eq!(result.len(), interpolation_vertices.len());
    //     for (v_result, v_expected) in result.iter().zip(interpolation_vertices) {
    //         // TODO: Use matrixcompare
    //         let diff = v_result - v_expected.coords;
    //         prop_assert!(diff.norm() <= 1e-6);
    //     }
    // }

    // TODO: Rewrite this test without NodalModel/Quad4Model
    // #[test]
    // fn finite_element_interpolation_at_interior_points_is_bounded_by_nodal_values(
    //     (mesh, u) in rectangular_uniform_mesh_strategy(1.0, 8).prop_flat_map(|mesh| {
    //         let ndof = mesh.vertices().len();
    //         let u_strategy = vec(-10.0..10.0, ndof).prop_map(move |v| DVector::from_iterator(ndof, v));
    //         (Just(mesh), u_strategy)
    //     }))
    // {
    //     assert_eq!(mesh.vertices().len(), u.len());
    //     prop_assume!(mesh.connectivity().len() > 0);
    //
    //     // It is easy to show that |u_h(x)| = |sum_i N_i(x) u_i| <= max_i |u_i|,
    //     // where u_i are the nodal values of the nodes for which x is in the support of the
    //     // nodal basis function.
    //
    //     let quadrature = quad_quadrature_strength_5_f64();
    //     let model = Quad4Model::from_mesh_and_quadrature(mesh.clone(), quadrature);
    //
    //     let interpolation_vertices: Vec<_> = mesh.connectivity().iter()
    //         .map(|connectivity| connectivity.cell(mesh.vertices()).unwrap())
    //         .map(|cell| ConvexPolygon::try_from(cell).expect("Meshes should only have convex cells"))
    //         .map(|polygon| find_interior_point(&polygon))
    //         .collect();
    //
    //     let interpolator = model.make_interpolator(&interpolation_vertices).unwrap();
    //     let mut result = vec![OVector::<f64, U1>::zeros(); interpolation_vertices.len()];
    //
    //     interpolator.interpolate_into(&mut result, &u);
    //
    //     prop_assert_eq!(result.len(), interpolation_vertices.len());
    //
    //     for (i, u_result) in result.iter().map(|p| p[0]).enumerate() {
    //         let cell_connectivity = mesh.connectivity()[i];
    //         let smaller_than_neighbor_nodes = cell_connectivity.vertex_indices()
    //             .iter()
    //             .map(|idx| u[*idx])
    //             .any(|u_j| u_result.abs() <= u_j.abs());
    //         prop_assert!(smaller_than_neighbor_nodes);
    //     }
    // }
}

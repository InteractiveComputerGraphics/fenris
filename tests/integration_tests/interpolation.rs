use itertools::{izip, Itertools};
use matrixcompare::assert_matrix_eq;
use nalgebra::{DVectorSlice, Point2, Vector1, Vector2};
use fenris::assembly::buffers::{BufferUpdate, InterpolationBuffer};
use fenris::mesh::procedural::create_unit_square_uniform_tri_mesh_2d;
use fenris::mesh::TriangleMesh2d;
use fenris::space::{FiniteElementConnectivity, FiniteElementSpace, InterpolateGradientInSpace, InterpolateInSpace, SpatiallyIndexed};
use fenris::util::global_vector_from_point_fn;
use fenris::quadrature;
use fenris::quadrature::Quadrature;

#[test]
fn spatially_indexed_interpolation_trimesh() {
    // We interpolate at internal (quadrature) points of a finite element space
    // in two ways:
    //  - by computing the values in reference coordinate space of each element
    //    (this forms the "expected" values)
    //  - by interpolating the quantity at the *physical* coordinates
    // This way we verify that the latter approach produces expected results.
    let mesh: TriangleMesh2d<f64> = create_unit_square_uniform_tri_mesh_2d(15);

    // Arbitrary scalar function u(p), where p is a 2-dimensional point
    let u = |p: &Point2<f64>| {
        let (x, y) = (p.x, p.y);
        Vector1::new((x.cos() + y.sin()) * x.powi(2))
    };
    let u_vec = global_vector_from_point_fn(&mesh.vertices(), u);
    let space = SpatiallyIndexed::from_space(mesh);

    let quadrature = quadrature::total_order::triangle::<f64>(4).unwrap();
    let mut interpolation_buffer = InterpolationBuffer::default();

    // For each element, compute interpolated value of quadrature points plus
    // the map to physical space. Then later we'll interpolate at these same points (in physical
    // space), so that we already know the correct answer.
    let (x_expected, u_expected, grad_u_expected): (Vec<_>, Vec<_>, Vec<_>) = (0 .. space.num_elements())
        .flat_map(|i| {
            quadrature.points()
                .iter()
                .map(|xi_j| {
                    let mut buffer = interpolation_buffer.prepare_element_in_space(i, &space, &u_vec, 1);
                    buffer.update_reference_point(xi_j, BufferUpdate::Both);
                    let u_j: Vector1<_> = buffer.interpolate();
                    let grad_u_j_ref: Vector2<_> = buffer.interpolate_ref_gradient();
                    let j_inv_t = buffer.element_reference_jacobian()
                        .try_inverse()
                        .unwrap()
                        .transpose();
                    let grad_u_j = j_inv_t * grad_u_j_ref;
                    let x_j = space.map_element_reference_coords(i, xi_j);
                    (x_j, u_j, grad_u_j)
                }).collect::<Vec<_>>()
        })
        .multiunzip();

    let mut u_result = vec![Vector1::zeros(); x_expected.len()];
    let mut grad_u_result = vec![Vector2::zeros(); x_expected.len()];
    space.interpolate_at_points(&x_expected, DVectorSlice::from(&u_vec), &mut u_result);
    space.interpolate_gradient_at_points(&x_expected, DVectorSlice::from(&u_vec), &mut grad_u_result);

    let iter = izip!(u_result, grad_u_result, u_expected, grad_u_expected);
    for (u, grad_u, u_expected, grad_u_expected) in iter {
        assert_matrix_eq!(u, u_expected, comp = abs, tol = 1e-12);
        assert_matrix_eq!(grad_u, grad_u_expected, comp = abs, tol = 1e-12);
    }
}
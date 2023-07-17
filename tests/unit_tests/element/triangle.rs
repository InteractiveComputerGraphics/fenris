use fenris::element::{
    ClosestPoint, ClosestPointInElement, ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement,
    Tri3d2Element, Tri3d3Element, Tri6d2Element,
};
use fenris::error::estimate_element_L2_error;
use fenris::geometry::proptest::clockwise_triangle2d_strategy_f64;
use fenris::geometry::{Triangle, Triangle2d};
use fenris::integrate::IntegrationWorkspace;
use fenris::mesh::procedural::create_unit_square_uniform_tri_mesh_2d;
use fenris::mesh::TriangleMesh2d;
use fenris::nalgebra::DVector;
use fenris::quadrature;

use fenris_geometry::LineSegment3d;
use fenris_optimize::calculus::{approximate_jacobian, VectorFunctionBuilder};

use matrixcompare::{assert_matrix_eq, assert_scalar_eq, prop_assert_matrix_eq};
use nalgebra::{distance, point, DVectorView, DimName, Dyn, OMatrix, OPoint, Point2, Vector1, U1, U2, U3, U6};

use crate::unit_tests::element::{is_likely_in_tri_ref_interior, point_in_tri_ref_domain};
use proptest::prelude::*;
use util::assert_approx_matrix_eq;

#[test]
fn tri3d2_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Tri3d2Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U3>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn tri6d2_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Tri6d2Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U6>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn tri3d2_closest_point_is_a_vertex() {
    // We test the case where the closest point is a vertex because the proptests don't cover it.
    let ref_vertices = Tri3d2Element::reference().vertices().clone();
    let vertices = [[1.0, 0.0], [2.0, 1.0], [-1.0, 2.0]].map(Point2::from);
    let element = Tri3d2Element::from_vertices(vertices);

    macro_rules! assert_exterior_closest_point {
        ($point:expr, ref_coords = $ref_coords:expr) => {{
            let result = element.closest_point(&$point);
            assert!(matches!(result, ClosestPoint::ClosestPoint(_)));
            assert_matrix_eq!(
                result.point().coords,
                $ref_coords.coords,
                comp = abs,
                tol = 1e-9 * element.diameter()
            );
        }};
    }

    assert_exterior_closest_point!(point![5.0, 2.0], ref_coords = ref_vertices[1]);
    assert_exterior_closest_point!(point![2.0, -1.0], ref_coords = ref_vertices[0]);
    assert_exterior_closest_point!(point![-3.0, 2.0], ref_coords = ref_vertices[2]);
}

#[test]
fn tri3d2_closest_point_interior_point() {
    let vertices = [[1.0, 0.0], [2.0, 1.0], [-1.0, 2.0]].map(Point2::from);
    let element = Tri3d2Element::from_vertices(vertices);

    let xi = point![-0.5, -0.5];
    let x = element.map_reference_coords(&xi);
    let result = element.closest_point(&x);
    assert!(matches!(result, ClosestPoint::InElement(_)));
    assert_matrix_eq!(
        result.point().coords,
        xi.coords,
        comp = abs,
        tol = 1e-9 * element.diameter()
    );
}

#[test]
fn tri3d2_closest_point_degenerate_elements() {
    // TODO: These tests are not at all comprehensive. I've tried to generate degenerate
    // inputs, but it's difficult to find inputs that actually cause serious problems.
    // I think ultimately such problematic inputs have to come from problematic real problems...

    // Degenerate to a single point
    {
        let v = point![3.0, 3.0];
        let element = Tri3d2Element::from_vertices([v, v, v]);

        let x = point![2.0, 2.0];
        let xi = element.closest_point(&x).point().clone();
        let x_element = element.map_reference_coords(&xi);
        assert_matrix_eq!(x_element.coords, v.coords, comp = abs, tol = 1e-12);
    }

    // Degenerate to a line
    {
        let vertices = [[1.0, 1.0], [2.0, 1.0], [0.5, 1.0]].map(Point2::from);
        let element = Tri3d2Element::from_vertices(vertices);

        let x = point![1.3, 1.5];
        let xi = element.closest_point(&x).point().clone();
        let x_element = element.map_reference_coords(&xi);

        assert_matrix_eq!(x_element.coords, point![1.3, 1.0].coords, comp = abs, tol = 1e-12);
    }

    // *Almost* degenerate to a line
    {
        let eps = 1e-15;
        let vertices = [[1.0, 1.0], [3.0, 2.0], [2.0, 1.5 + eps]].map(Point2::from);
        let element = Tri3d2Element::from_vertices(vertices);

        let x = point![2.0, 1.5 + eps / 2.0];
        let xi = element.closest_point(&x).point().clone();
        let x_element = element.map_reference_coords(&xi);

        assert_matrix_eq!(x_element.coords, x.coords, comp = abs, tol = 1e-12);
    }
}

#[test]
fn tri3d2_closest_point_boundary_points() {
    // Map points from the boundary of the reference element
    // to physical space, then check that the closest point returned by the element
    // has the same reference coordinates
    let mesh: TriangleMesh2d<f64> = create_unit_square_uniform_tri_mesh_2d(10);
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
            let xi_closest = element.closest_point(&x).point().clone();
            assert_matrix_eq!(xi_closest.coords, xi.coords, comp = abs, tol = 1e-12);
        }
    }
}

proptest! {
    #[test]
    fn tri3_affine_function_error_is_zero(tri in clockwise_triangle2d_strategy_f64()) {
        let element = Tri3d2Element::from(tri);
        // If u_exact is an affine function, then we can exactly represent it
        // with a Tri3 element. Then the basis weights of u_h are given by
        // u_exact(x_i), where x_i are the coordinates of each node of the element.
        let u_exact = |x: &Point2<f64>| 2.0 * x[0] - 3.0 * x[1] + 1.5;
        let u_weights = DVector::from_vec(tri.0.iter().map(|x| u_exact(x)).collect::<Vec<_>>());

        // TODO: Use lower strength quadrature
        let (weights, points) = quadrature::total_order::triangle(5).unwrap();
        let error = estimate_element_L2_error(
            &element,
            &|x: &Point2<_>| Vector1::new(u_exact(x)),
            DVectorView::from(&u_weights),
            &weights,
            &points,
            &mut IntegrationWorkspace::default());

        assert_scalar_eq!(error, 0.0, comp=abs, tol=element.diameter() * 1e-12);
    }

    #[test]
    fn tri6_affine_function_error_is_zero(tri in clockwise_triangle2d_strategy_f64()) {
        let element = Tri6d2Element::from(Tri3d2Element::from(tri));
        // If u_exact is an affine function, then we can exactly represent it
        // with a Tri3 element. Then the basis weights of u_h are given by
        // u_exact(x_i), where x_i are the coordinates of each node of the element.
        let u_exact = |x: &Point2<f64>| 2.0 * x[0] - 3.0 * x[1] + 1.5;
        let u_weights = DVector::from_vec(element.vertices().iter().map(|x| u_exact(x)).collect::<Vec<_>>());

        // TODO: Use lower strength quadrature
        let (weights, points) = quadrature::total_order::triangle(5).unwrap();
        let error = estimate_element_L2_error(
            &element,
            &|x: &Point2<_>| Vector1::new(u_exact(x)),
            DVectorView::from(&u_weights),
            &weights,
            &points,
            &mut IntegrationWorkspace::default());

        assert_scalar_eq!(error, 0.0, comp=abs, tol=element.diameter() * 1e-12);
    }

    #[test]
    fn tri6_quadratic_function_error_is_zero(tri in clockwise_triangle2d_strategy_f64()) {
        let element = Tri6d2Element::from(Tri3d2Element::from(tri));
        // If u_exact is an affine function, then we can exactly represent it
        // with a Tri3 element. Then the basis weights of u_h are given by
        // u_exact(x_i), where x_i are the coordinates of each node of the element.
        let u_exact = |x: &Point2<f64>| 2.0 * x[0] * x[0] - 3.0 * x[1] * x[0] + 0.5 * x[1] + 1.5;
        let u_weights = DVector::from_vec(element.vertices().iter().map(|x| u_exact(x)).collect::<Vec<_>>());

        // TODO: Use lower strength quadrature
        let (weights, points) = quadrature::total_order::triangle(5).unwrap();
        let error = estimate_element_L2_error(
            &element,
            &|x: &Point2<_>| Vector1::new(u_exact(x)),
            DVectorView::from(&u_weights),
            &weights,
            &points,
            &mut IntegrationWorkspace::default());

        // TODO: Check tolerance
        assert_scalar_eq!(error, 0.0, comp=abs, tol=element.diameter() * element.diameter() * 1e-12);
    }

        #[test]
    fn tri3d2_element_gradient_is_derivative_of_transform(
        (tri, xi) in (any::<Triangle2d<f64>>(), point_in_tri_ref_domain())
    ) {

        let elem = Tri3d2Element::from(tri);

        // Finite difference parameter
        let h = 1e-6;
        // Note: Function values are given as row vectors, so we transpose to get the result,
        // and we must also transpose the end result
        let f = VectorFunctionBuilder::with_dimension(3).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U2::name(), U1::name())).clone_owned());
            x.copy_from(&elem.evaluate_basis(&xi).transpose());
        });

        let grad = elem.gradients(&xi);
        let grad_approx = approximate_jacobian(f, &DVectorView::<_, Dyn>::from(&xi.coords).clone_owned(), &h).transpose();

        assert_approx_matrix_eq!(grad, &grad_approx, abstol=1e-5);
    }

    #[test]
    fn tri6d2_element_gradient_is_derivative_of_transform(
        (tri, xi) in (any::<Triangle2d<f64>>(), point_in_tri_ref_domain())
    ) {

        let elem = Tri6d2Element::from(Tri3d2Element::from(tri));

        // Finite difference parameter
        let h = 1e-6;
        // Note: Function values are given as row vectors, so we transpose to get the result,
        // and we must also transpose the end result
        let f = VectorFunctionBuilder::with_dimension(6).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U2::name(), U1::name())).clone_owned());
            x.copy_from(&elem.evaluate_basis(&xi).transpose());
        });

        let grad = elem.gradients(&xi);
        let grad_approx = approximate_jacobian(f, &DVectorView::<_, Dyn>::from(&xi.coords).clone_owned(), &h).transpose();

        assert_approx_matrix_eq!(grad, &grad_approx, abstol=1e-5);
    }

    #[test]
    fn tri3_closest_point_in_interior_is_identity(element: Tri3d2Element<f64>, xi in point_in_tri_ref_domain()) {
        // We cannot compare the reference coordinates directly, because the element
        // may be degenerate. So instead we check that the closest point in reference coords
        // maps back to the same point in physical space
        let x = element.map_reference_coords(&xi);
        let xi2 = element.closest_point(&x).point().clone();
        let x2 = element.map_reference_coords(&xi2);
        prop_assert_matrix_eq!(x2.coords, x.coords, comp = abs, tol = element.diameter() * 1e-9);
    }

    #[test]
    fn tri3_exterior_closest_point(
        element: Tri3d2Element<f64>,
        t in 0.0 ..= 1.0,
        edge in 0 .. 3usize,
        extrusion in 0.0 ..= 5.0)
    {
        let reference = Tri3d2Element::reference();
        let ref_edge = Triangle(*reference.vertices()).edge(edge);
        let element_edge = Triangle(*element.vertices()).edge(edge);
        let xi0 = ref_edge.point_from_parameter(t);
        let x0 = element.map_reference_coords(&xi0);

        // Produce a point by extruding along the normal direction of the edge.
        // The closest point to this point will by definition be the starting point.
        let x = x0 + extrusion * element_edge.normal_dir();
        let result = element.closest_point(&x);
        prop_assert!(matches!(result, ClosestPoint::ClosestPoint(_)));
        let mapped_result = element.map_reference_coords(result.point());
        prop_assert_matrix_eq!(mapped_result.coords, x0.coords,
                               comp = abs, tol = element.diameter() * 1e-9);
    }

    #[test]
    fn tri3d3_closest_point_on_triangle_is_identity(element: Tri3d3Element<f64>, xi in point_in_tri_ref_domain()) {
        // We cannot compare the reference coordinates directly, because the element
        // may be degenerate. So instead we check that the closest point in reference coords
        // maps back to the same point in physical space
        let x = element.map_reference_coords(&xi);
        let xi2 = element.closest_point(&x).point().clone();
        let x2 = element.map_reference_coords(&xi2);
        prop_assert_matrix_eq!(x2.coords, x.coords, comp = abs, tol = element.diameter() * 1e-6);
    }

    #[test]
    fn tri3d3_edge_voronoi_region_closest_point(
        element: Tri3d3Element<f64>,
        t in 0.0 ..= 1.0,
        edge_idx in 0 .. 3usize,
        outward_factor in 0.0 ..= 5.0,
        orthogonal_factor in -5.0 ..= 5.0)
    {
        // xi is the closest point in reference coordinates, x0 in physical coords
        let reference_element = Tri3d2Element::reference();
        let reference_edge = Triangle(*reference_element.vertices()).edge(edge_idx);
        let xi = reference_edge.point_from_parameter(t);
        let x0 = element.map_reference_coords(&xi);

        // Construct a point x exterior to the triangle for which xi is the closest point
        // We pick a point in the Voronoi region belonging to the given edge
        let triangle = Triangle(*element.vertices());
        let edge = triangle.edge(edge_idx);
        let outward_vec = edge.tangent_dir().cross(&triangle.normal_dir());
        let x = x0 + outward_factor * outward_vec + orthogonal_factor * triangle.normal_dir();

        let closest = element.closest_point(&x);
        let xi_closest = closest.point();
        prop_assert!(is_likely_in_tri_ref_interior(xi_closest));

        let x_closest = element.map_reference_coords(xi_closest);
        let tol = f64::max(element.diameter(), distance(&x, &x0)) * 1e-6;

        prop_assert_matrix_eq!(x_closest.coords, x0.coords, comp = abs, tol = tol);
        prop_assert!(matches!(closest, ClosestPoint::ClosestPoint(_)));
    }

    #[test]
    fn tri3d3_interior_voronoi_region_closest_point(
        element: Tri3d3Element<f64>,
        xi in point_in_tri_ref_domain(),
        orthogonal_factor in -5.0 ..= 5.0,
    ) {
        // xi is the closest point in reference coordinates, x in physical coords
        let x0 = element.map_reference_coords(&xi);

        // Construct a point x interior to the triangle for which xi is the closest point
        // We pick a point in the Voronoi region belonging to the triangle's interior
        let triangle = Triangle(*element.vertices());
        let x = x0 + orthogonal_factor * triangle.normal_dir();

        let closest = element.closest_point(&x);
        let xi_closest = closest.point();
        prop_assert!(is_likely_in_tri_ref_interior(xi_closest));

        let x_closest = element.map_reference_coords(xi_closest);
        let tol = f64::max(element.diameter(), distance(&x, &x0)) * 1e-6;
        prop_assert_matrix_eq!(x_closest.coords, x0.coords, comp = abs, tol = tol);
        prop_assert!(matches!(closest, ClosestPoint::ClosestPoint(_)));
    }

    #[test]
    fn tri3d3_vertex_voronoi_region_closest_point(
        element: Tri3d3Element<f64>,
        vertex_idx in 0 .. 3usize,
        outward_factor1 in 0.0 ..= 5.0,
        outward_factor2 in 0.0 ..= 5.0,
        orthogonal_factor in -5.0 ..= 5.0)
    {
        // xi is the closest point in reference coordinates, x0 in physical coords
        let reference_element = Tri3d2Element::reference();
        let xi = reference_element.vertices()[vertex_idx];
        let x0 = element.map_reference_coords(&xi);

        let triangle = Triangle(*element.vertices());
        let [v1, v_mid, v2] = [(vertex_idx + 2) % 3, vertex_idx, (vertex_idx + 1) % 3]
            .map(|idx| element.vertices()[idx].clone());
        let edge1 = LineSegment3d::from_end_points(v1, v_mid);
        let edge2 = LineSegment3d::from_end_points(v_mid, v2);

        let outward1 = edge1.tangent_dir().cross(&triangle.normal_dir());
        let outward2 = edge2.tangent_dir().cross(&triangle.normal_dir());

        let x = x0
            + outward_factor1 * outward1
            + outward_factor2 * outward2
            + orthogonal_factor * triangle.normal_dir();

        let closest = element.closest_point(&x);
        let xi_closest = closest.point().clone();
        prop_assert!(is_likely_in_tri_ref_interior(&xi_closest));

        let x_closest = element.map_reference_coords(&xi_closest);
        let tol = f64::max(element.diameter(), distance(&x, &x0)) * 1e-6;
        prop_assert_matrix_eq!(x_closest.coords, x0.coords, comp = abs, tol = tol);
        prop_assert!(matches!(closest, ClosestPoint::ClosestPoint(_)));
    }
}

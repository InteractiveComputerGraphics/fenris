use crate::unit_tests::element::{is_definitely_in_tet_ref_interior, is_likely_in_tet_ref_interior, point_in_tet_ref_domain, point_in_tri_ref_domain};
use fenris::connectivity::{Connectivity, Tet4Connectivity};
use fenris::element::{
    ClosestPoint, ClosestPointInElement, ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement,
    Tet10Element, Tet20Element, Tet4Element,
};
use fenris::error::estimate_element_L2_error;
use fenris::integrate::IntegrationWorkspace;
use fenris::nalgebra::DVector;
use fenris::quadrature;
use fenris_geometry::Triangle;
use fenris_optimize::calculus::{approximate_jacobian, VectorFunctionBuilder};
use itertools::izip;
use matrixcompare::{assert_scalar_eq, prop_assert_matrix_eq};
use nalgebra::{distance, DVectorView, DimName, Dyn, MatrixView, OMatrix, OPoint, Point3, Vector1, Vector3, U1, U10, U20, U3, U4, point};
use proptest::array::uniform3;
use proptest::prelude::*;
use util::assert_approx_matrix_eq;

#[test]
fn tet4_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Tet4Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U4>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn tet10_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Tet10Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U10>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn tet20_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Tet20Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U20>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn tet4_closest_point_failure_case() {
    let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, 0.5]]
        .map(Point3::from);
    let element = Tet4Element::from_vertices(vertices);
    let p = point![0.875, 0.375, 0.375];
    let closest = element.closest_point(&p);

    let xi_closest = closest.point();
    let x_closest = element.map_reference_coords(&xi_closest);
    assert_ne!(x_closest, p);

}

proptest! {
    #[test]
    fn tet4_partition_of_unity(xi in point_in_tet_ref_domain()) {
        let element = Tet4Element::reference();
        let phi = element.evaluate_basis(&xi);
        let phi_sum: f64 = phi.sum();
        prop_assert!( (phi_sum - 1.0f64).abs() <= 1e-12);
    }

    #[test]
    fn tet10_partition_of_unity(xi in point_in_tet_ref_domain()) {
        let element = Tet10Element::reference();
        let phi = element.evaluate_basis(&xi);
        let phi_sum: f64 = phi.sum();
        prop_assert!( (phi_sum - 1.0f64).abs() <= 1e-12);
    }

    #[test]
    fn tet20_partition_of_unity(xi in point_in_tet_ref_domain()) {
        let element = Tet20Element::reference();
        let phi = element.evaluate_basis(&xi);
        let phi_sum: f64 = phi.sum();

        prop_assert!( (phi_sum - 1.0f64).abs() <= 1e-12);
    }

        #[test]
    fn tet4_partition_of_unity_gradient((x, y, z) in (-1.0 ..= 1.0, -1.0 ..= 1.0, -1.0 ..= 1.0)) {
        // Since the sum of basis functions is 1, the sum of the gradients must be 0
        let xi = Point3::new(x, y, z);
        let element = Tet4Element::reference();
        let grad = element.gradients(&xi);
        let grad_sum = grad.column_sum();

        assert_approx_matrix_eq!(grad_sum, Vector3::<f64>::zeros(), abstol=1e-12);
    }

    #[test]
    fn tet10_partition_of_unity_gradient((x, y, z) in (-1.0 ..= 1.0, -1.0 ..= 1.0, -1.0 ..= 1.0)) {
        // Since the sum of basis functions is 1, the sum of the gradients must be 0
        let xi = Point3::new(x, y, z);
        let element = Tet10Element::reference();
        let grad = element.gradients(&xi);
        let grad_sum = grad.column_sum();

        assert_approx_matrix_eq!(grad_sum, Vector3::<f64>::zeros(), abstol=1e-12);
    }

    #[test]
    fn tet20_partition_of_unity_gradient((x, y, z) in (-1.0 ..= 1.0, -1.0 ..= 1.0, -1.0 ..= 1.0)) {
        // Since the sum of basis functions is 1, the sum of the gradients must be 0
        let xi = Point3::new(x, y, z);
        let element = Tet20Element::reference();
        let grad = element.gradients(&xi);
        let grad_sum = grad.column_sum();

        assert_approx_matrix_eq!(grad_sum, Vector3::<f64>::zeros(), abstol=1e-12);
    }

    #[test]
    fn tet4_affine_function_error_is_zero(element in any::<Tet4Element<f64>>()) {
        // If u_exact is an affine function, then we can exactly represent it
        // with a Tet4 element. Then the basis weights of u_h are given by
        // u_exact(x_i), where x_i are the coordinates of each node of the element.
        let u_exact = |p: &Point3<f64>| {
            let x = p.x;
            let y = p.y;
            let z = p.z;
            3.0 * x + 2.0 * y - 3.0 * z + 3.0
        };
        let u_weights = DVector::from_vec(element.vertices()
                                                    .iter()
                                                    .map(|x| u_exact(x))
                                                    .collect::<Vec<_>>());

        let (weights, points) = quadrature::total_order::tetrahedron(10).unwrap();
        let error = estimate_element_L2_error(
            &element, &|x: &Point3<_>| Vector1::new(u_exact(x)),
            MatrixView::from(&u_weights),
            &weights,
            &points,
            &mut IntegrationWorkspace::default());

        assert_scalar_eq!(error, 0.0, comp=abs, tol=element.diameter() * 1e-12);
    }

    #[test]
    fn tet4_element_gradient_is_derivative_of_transform(
        (tet, xi) in (any::<Tet4Element<f64>>(), point_in_tet_ref_domain())
    ) {
        // Finite difference parameter
        let h = 1e-6;
        // Note: Function values are given as row vectors, so we transpose to get the result,
        // and we must also transpose the end result
        let f = VectorFunctionBuilder::with_dimension(4).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U3::name(), U1::name())).clone_owned());
            x.copy_from(&tet.evaluate_basis(&xi).transpose());
        });

        let grad = tet.gradients(&xi);
        let grad_approx = approximate_jacobian(f, &DVectorView::<_, Dyn>::from(&xi.coords).clone_owned(), &h).transpose();

        assert_approx_matrix_eq!(grad, &grad_approx, abstol=1e-5);
    }

    #[test]
    fn tet10_reference_element_gradient_is_derivative_of_transform(
        xi in point_in_tet_ref_domain()
    ) {
        let tet = Tet10Element::reference();
        // Finite difference parameter
        let h = 1e-6;
        // Note: Function values are given as row vectors, so we transpose to get the result,
        // and we must also transpose the end result
        let f = VectorFunctionBuilder::with_dimension(10).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U3::name(), U1::name())).clone_owned());
            x.copy_from(&tet.evaluate_basis(&xi).transpose());
        });

        let grad = tet.gradients(&xi);
        let grad_approx = approximate_jacobian(f, &DVectorView::<_, Dyn>::from(&xi.coords).clone_owned(), &h).transpose();

        assert_approx_matrix_eq!(grad, &grad_approx, abstol=1e-5);
    }

    #[test]
    fn tet20_reference_element_gradient_is_derivative_of_transform(
        xi in point_in_tet_ref_domain()
    ) {
        let tet = Tet20Element::reference();
        // Finite difference parameter
        let h = 1e-6;
        // Note: Function values are given as row vectors, so we transpose to get the result,
        // and we must also transpose the end result
        let f = VectorFunctionBuilder::with_dimension(20).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U3::name(), U1::name())).clone_owned());
            x.copy_from(&tet.evaluate_basis(&xi).transpose());
        });

        let grad = tet.gradients(&xi);
        let grad_approx = approximate_jacobian(f, &DVectorView::<_, Dyn>::from(&xi.coords).clone_owned(), &h).transpose();

        assert_approx_matrix_eq!(grad, &grad_approx, abstol=1e-5);
    }

    #[test]
    fn tet4_element_jacobian_is_derivative_of_transform(
        (tet, xi) in (any::<Tet4Element<f64>>(), point_in_tet_ref_domain())
    ) {
        // Finite difference parameter
        let h = 1e-6;
        // Function is x = f(xi)
        let f = VectorFunctionBuilder::with_dimension(3).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U3::name(), U1::name())).clone_owned());
            x.copy_from(&tet.map_reference_coords(&xi).coords);
        });

        let j = tet.reference_jacobian(&xi);
        let j_approx = approximate_jacobian(f, &DVectorView::<_, Dyn>::from(&xi.coords).clone_owned(), &h);
        assert_approx_matrix_eq!(j, &j_approx, abstol=1e-5);
    }

    #[test]
    fn tet4_element_jacobian_has_non_negative_jacobian(
        (tet, xi) in (any::<Tet4Element<f64>>(), point_in_tet_ref_domain())
    ) {
        let j = tet.reference_jacobian(&xi);
        prop_assert!(j.determinant() >= 0.0);
    }

    #[test]
    fn tet4_interior_voronoi_region_closest_point(
        element: Tet4Element<f64>,
        xi in point_in_tet_ref_domain(),
    ) {
        let x = element.map_reference_coords(&xi);
        let closest = element.closest_point(&x);
        let xi_closest = closest.point();
        let x_closest = element.map_reference_coords(&xi_closest);

        let tol = 1e-6 * element.diameter();
        prop_assert_matrix_eq!(x_closest.coords, x.coords, comp = abs, tol = tol);

        if is_definitely_in_tet_ref_interior(&xi) {
            prop_assert!(matches!(closest, ClosestPoint::InElement(_)));
        }
    }

    #[test]
    fn tet4_face_voronoi_region_closest_point(
        tet_element: Tet4Element<f64>,
        xi_face in point_in_tri_ref_domain(),
        face_idx in 0 .. 4usize,
        normal_factor in 0.0 ..= 5.0)
    {
        let face_conn = Tet4Connectivity([0, 1, 2, 3]).get_face_connectivity(face_idx).unwrap();

        let tet_reference = Tet4Element::reference();
        let xi = face_conn
            .element(tet_reference.vertices())
            .unwrap()
            .map_reference_coords(&xi_face);
        let x0 = tet_element.map_reference_coords(&xi);

        let face_element = face_conn.element(tet_element.vertices()).unwrap();
        let triangle = Triangle(face_element.vertices().clone());

        let x = x0 + normal_factor * triangle.normal_dir();

        let closest = tet_element.closest_point(&x);
        let xi_closest = closest.point();
        prop_assert!(is_likely_in_tet_ref_interior(xi_closest));

        let x_closest = tet_element.map_reference_coords(&xi_closest);
        let tol = f64::max(tet_element.diameter(), distance(&x, &x0)) * 1e-6;
        prop_assert_matrix_eq!(x_closest.coords, x0.coords, comp = abs, tol = tol);

        if distance(&x, &x0) > 1e-6 * tet_element.diameter() {
            // Only if we are sufficiently far away do we check that the classification is correct,
            // since on or very close to the boundary, the point could be classified as interior
            // due to floating point shenanigans
            prop_assert!(matches!(closest, ClosestPoint::ClosestPoint(_)));
        }
    }

    #[test]
    fn tet4_vertex_voronoi_region_closest_point(
        tet_element: Tet4Element<f64>,
        vertex_idx in 0 .. 4usize,
        normal_factors in uniform3(0.0 .. 5.0))
    {
        let face_connectivities = [0, 1, 2, 3]
            .map(|face_idx| Tet4Connectivity([0, 1, 2, 3]).get_face_connectivity(face_idx).unwrap());

        let tet_reference = Tet4Element::<f64>::reference();
        let xi = tet_reference.vertices()[vertex_idx];
        let x0 = tet_element.map_reference_coords(&xi);

        let neighboring_faces: Vec<_> = face_connectivities
            .iter()
            .filter(|conn| conn.vertex_indices().contains(&vertex_idx))
            .map(|conn| conn.element(tet_element.vertices()).unwrap())
            .map(|tri_element| Triangle(tri_element.vertices().clone()))
            .collect();
        assert_eq!(neighboring_faces.len(), 3);

        let mut x = x0.clone();
        for (triangle, normal_factor) in izip!(neighboring_faces, normal_factors) {
            x += normal_factor * triangle.normal_dir();
        }

        let closest = tet_element.closest_point(&x);
        let xi_closest = closest.point();
        prop_assert!(is_likely_in_tet_ref_interior(&xi_closest));

        let x_closest = tet_element.map_reference_coords(&xi_closest);
        let tol = f64::max(tet_element.diameter(), distance(&x, &x0)) * 1e-6;
        prop_assert_matrix_eq!(x_closest.coords, x0.coords, comp = abs, tol = tol);

        if distance(&x, &x0) > 1e-6 * tet_element.diameter() {
            // Only if we are sufficiently far away do we check that the classification is correct,
            // since on or very close to the boundary, the point could be classified as interior
            // due to floating point shenanigans
            prop_assert!(matches!(closest, ClosestPoint::ClosestPoint(_)));
        }
    }
}

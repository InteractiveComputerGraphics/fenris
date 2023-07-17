use fenris::element::{FiniteElement, FixedNodesReferenceFiniteElement, Tet10Element, Tet20Element, Tet4Element};
use fenris::error::estimate_element_L2_error;

use fenris::integrate::IntegrationWorkspace;

use fenris::nalgebra::DVector;
use fenris::quadrature;

use fenris_optimize::calculus::{approximate_jacobian, VectorFunctionBuilder};

use matrixcompare::assert_scalar_eq;
use nalgebra::{
    DVectorView, DimName, Dyn, MatrixView, OMatrix, OPoint, Point3, Vector1, Vector3, U1, U10, U20, U3, U4,
};

use crate::unit_tests::element::point_in_tet_ref_domain;
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
}

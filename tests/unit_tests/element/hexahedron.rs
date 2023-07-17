use fenris::element::{FixedNodesReferenceFiniteElement, Hex20Element, Hex27Element, Hex8Element};
use fenris::error::estimate_element_L2_error;

use fenris::integrate::IntegrationWorkspace;

use fenris::nalgebra::DVector;
use fenris::quadrature;

use fenris_optimize::calculus::{approximate_jacobian, VectorFunctionBuilder};

use matrixcompare::assert_scalar_eq;
use nalgebra::{DVectorView, DimName, Dyn, MatrixView, OMatrix, OPoint, Point3, Vector1, U1, U20, U27, U3, U8};

use crate::unit_tests::element::point_in_hex_ref_domain;
use proptest::prelude::*;
use util::assert_approx_matrix_eq;

#[test]
fn hex8_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Hex8Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U8>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn hex27_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Hex27Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U27>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn hex20_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Hex20Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U20>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn hex27_triquadratic_function_exact_error() {
    let element = Hex27Element::reference();
    // If u_exact is a triquadratic function, then we can exactly represent it
    // with a Hex27 element. Then the basis weights of u_h are given by
    // u_exact(x_i), where x_i are the coordinates of each node of the element.
    let u_exact = |p: &Point3<f64>| {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        // Three arbitrary quadratic functions in each argument x, y, z
        let f = 3.0 * x * x - 2.0 * x + 5.0;
        let g = -2.0 * y * y + 3.0 * y + 1.5;
        let h = 1.5 * z * z + 1.2 * z - 3.0;

        f * g * h
    };
    let u_weights = DVector::from_vec(element.vertices().iter().map(u_exact).collect());

    // TODO: Use lower strength quadrature
    let (weights, points) = quadrature::total_order::hexahedron(11).unwrap();
    let error = estimate_element_L2_error(
        &element,
        &|x: &Point3<_>| Vector1::new(u_exact(x)),
        MatrixView::from(&u_weights),
        &weights,
        &points,
        &mut IntegrationWorkspace::default(),
    );

    // Note: The solution here is obtained by symbolic integration. See
    // the accompanying notebooks
    assert_scalar_eq!(error, 0.0, comp = abs, tol = 1e-12);
}

#[test]
fn hex20_quadratic_function_exact_error() {
    let element = Hex20Element::reference();
    // If u_exact is a quadratic function, then we can exactly represent it
    // with a Hex20 element. Then the basis weights of u_h are given by
    // u_exact(x_i), where x_i are the coordinates of each node of the element.
    // Note that the space of functions spanned by the Hex20 basis functions is larger
    // than the space of quadratic functions, so this is not a sufficient condition
    // for checking correctness
    let u_exact = |p: &Point3<f64>| {
        let x = p[0];
        let y = p[1];
        let z = p[2];

        2.0 * x * x + 4.0 * y * y - 3.0 * z * z + 3.0 * x * y - 5.0 * x * z + 1.5 * y * z + 3.0 * x - 2.0 * y
            + 3.0 * z
            + 9.0
    };
    let u_weights = DVector::from_vec(element.vertices().iter().map(|x| u_exact(x)).collect());

    // TODO: Use lower strength quadrature
    let (weights, points) = quadrature::total_order::hexahedron(11).unwrap();
    let error = estimate_element_L2_error(
        &element,
        &|x: &Point3<_>| Vector1::new(u_exact(x)),
        DVectorView::from(&u_weights),
        &weights,
        &points,
        &mut IntegrationWorkspace::default(),
    );

    // Note: The solution here is obtained by symbolic integration. See
    // the accompanying notebooks
    assert_scalar_eq!(error, 0.0, comp = abs, tol = 1e-12);
}

proptest! {
    #[test]
    fn hex8_partition_of_unity(xi in point_in_hex_ref_domain()) {
        let element = Hex8Element::reference();
        let phi = element.evaluate_basis(&xi);
        let phi_sum: f64 = phi.sum();
        prop_assert!( (phi_sum - 1.0f64).abs() <= 1e-12);
    }

    #[test]
    fn hex27_reference_element_gradient_is_derivative_of_transform(
        xi in point_in_hex_ref_domain()
    ) {
        let hex = Hex27Element::reference();
        // Finite difference parameter
        let h = 1e-6;
        // Note: Function values are given as row vectors, so we transpose to get the result,
        // and we must also transpose the end result
        let f = VectorFunctionBuilder::with_dimension(27).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U3::name(), U1::name())).clone_owned());
            x.copy_from(&hex.evaluate_basis(&xi).transpose());
        });

        let grad = hex.gradients(&xi);
        let xi = DVectorView::<_, Dyn>::from(&xi.coords).clone_owned();
        let grad_approx = approximate_jacobian(f, &xi, &h).transpose();

        assert_approx_matrix_eq!(grad, &grad_approx, abstol=1e-5);
    }

    #[test]
    fn hex20_reference_element_gradient_is_derivative_of_transform(
        xi in point_in_hex_ref_domain()
    ) {
        let hex = Hex20Element::reference();
        // Finite difference parameter
        let h = 1e-6;
        // Note: Function values are given as row vectors, so we transpose to get the result,
        // and we must also transpose the end result
        let f = VectorFunctionBuilder::with_dimension(20).with_function(move |x, xi| {
            let xi = OPoint::from(xi.generic_view((0, 0), (U3::name(), U1::name())).clone_owned());
            x.copy_from(&hex.evaluate_basis(&xi).transpose());
        });

        let grad = hex.gradients(&xi);
        let xi = DVectorView::<_, Dyn>::from(&xi.coords).clone_owned();
        let grad_approx = approximate_jacobian(f, &xi, &h).transpose();

        assert_approx_matrix_eq!(grad, &grad_approx, abstol=1e-5);
    }
}

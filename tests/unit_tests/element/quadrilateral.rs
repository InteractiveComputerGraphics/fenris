use fenris::element::{
    map_physical_coordinates, FiniteElement, FixedNodesReferenceFiniteElement, Quad4d2Element, Quad9d2Element,
};
use fenris::error::estimate_element_L2_error;
use fenris::geometry::proptest::nondegenerate_convex_quad2d_strategy_f64;
use fenris::geometry::Quad2d;
use fenris::integrate::IntegrationWorkspace;

use fenris::nalgebra::DVector;
use fenris::quadrature;

use matrixcompare::assert_scalar_eq;
use nalgebra::{MatrixView, OMatrix, Point2, Vector1, Vector2, U1, U9};

use proptest::prelude::*;
use util::assert_approx_matrix_eq;

#[test]
fn map_reference_coords_quad2d() {
    let vertices = [
        Point2::new(5.0, 3.0),
        Point2::new(10.0, 4.0),
        Point2::new(11.0, 6.0),
        Point2::new(6.0, 4.0),
    ];
    let quad = Quad2d(vertices);
    let quad = Quad4d2Element::from(quad);

    let x0 = quad.map_reference_coords(&Point2::new(-1.0, -1.0));
    assert!(x0.coords.relative_eq(&vertices[0].coords, 1e-10, 1e-10));

    let x1 = quad.map_reference_coords(&Point2::new(1.0, -1.0));
    assert!(x1.coords.relative_eq(&vertices[1].coords, 1e-10, 1e-10));

    let x2 = quad.map_reference_coords(&Point2::new(1.0, 1.0));
    assert!(x2.coords.relative_eq(&vertices[2].coords, 1e-10, 1e-10));

    let x3 = quad.map_reference_coords(&Point2::new(-1.0, 1.0));
    assert!(x3.coords.relative_eq(&vertices[3].coords, 1e-10, 1e-10));
}

#[test]
fn map_physical_coords_quad2d() {
    let vertices = [
        Point2::new(5.0, 3.0),
        Point2::new(10.0, 4.0),
        Point2::new(11.0, 6.0),
        Point2::new(6.0, 4.0),
    ];
    let quad = Quad2d(vertices);
    let quad = Quad4d2Element::from(quad);

    let xi0 = map_physical_coordinates(&quad, &vertices[0]).unwrap();
    assert!(xi0
        .coords
        .relative_eq(&Vector2::new(-1.0, -1.0), 1e-10, 1e-10));

    let xi1 = map_physical_coordinates(&quad, &vertices[1]).unwrap();
    assert!(xi1
        .coords
        .relative_eq(&Vector2::new(1.0, -1.0), 1e-10, 1e-10));

    let xi2 = map_physical_coordinates(&quad, &vertices[2]).unwrap();
    assert!(xi2
        .coords
        .relative_eq(&Vector2::new(1.0, 1.0), 1e-10, 1e-10));

    let xi3 = map_physical_coordinates(&quad, &vertices[3]).unwrap();
    assert!(xi3
        .coords
        .relative_eq(&Vector2::new(-1.0, 1.0), 1e-10, 1e-10));
}

#[test]
fn quad9_lagrange_property() {
    // We expect that N_i(x_j) = delta_ij
    // where N_i is the ith basis function, j is the vertex associated with the ith node,
    // and delta_ij is the Kronecker delta.
    let element = Quad9d2Element::reference();

    for (i, xi) in element.vertices().into_iter().enumerate() {
        let phi = element.evaluate_basis(&xi);

        let mut expected = OMatrix::<f64, U1, U9>::zeros();
        expected[i] = 1.0;

        assert_approx_matrix_eq!(phi, expected, abstol = 1e-12);
    }
}

#[test]
fn quad4_bilinear_function_exact_error() {
    let quad = Quad2d([
        Point2::new(-1.0, -1.0),
        Point2::new(2.0, -2.0),
        Point2::new(4.0, 1.0),
        Point2::new(-2.0, 3.0),
    ]);
    let element = Quad4d2Element::from(quad);
    // If u_exact is a bilinear function, then we can exactly represent it
    // with a Quad4 element. Then the basis weights of u_h are given by
    // u_exact(x_i), where x_i are the coordinates of each node of the element.
    let u_exact = |p: &Point2<f64>| {
        let x = p[0];
        let y = p[1];
        5.0 * x * y + 3.0 * x - 2.0 * y - 5.0
    };
    let u_weights = DVector::from_vec(quad.0.iter().map(u_exact).collect::<Vec<_>>());

    // TODO: Use lower strength quadrature
    let (weights, points) = quadrature::total_order::quadrilateral(11).unwrap();
    let error = estimate_element_L2_error(
        &element,
        &|x: &Point2<_>| Vector1::new(u_exact(x)),
        MatrixView::from(&u_weights),
        &weights,
        &points,
        &mut IntegrationWorkspace::default(),
    );

    // Note: The solution here is obtained by symbolic integration. See
    // the accompanying notebooks
    assert_scalar_eq!(
        error,
        (9955.0f64 / 12.0).sqrt(),
        comp = abs,
        tol = element.diameter() * 1e-12
    );
}

proptest! {
    #[test]
    fn quad4_affine_function_error_is_zero(quad in nondegenerate_convex_quad2d_strategy_f64()) {
        let element = Quad4d2Element::from(quad);
        // If u_exact is an affine function, then we can exactly represent it
        // with a Quad4 element. Then the basis weights of u_h are given by
        // u_exact(x_i), where x_i are the coordinates of each node of the element.
        // Note that this is not true for a general bilinear function unless
        // the quad and the reference quad are related by an affine transformation
        // (i.e. the quad is a parallellogram).
        let u_exact = |x: &Point2<f64>| {
            let y = x[1];
            let x = x[0];
            3.0 * x + 2.0 * y - 3.0
        };
        let u_weights = DVector::from_vec(quad.0
                                                    .iter()
                                                    .map(|x| u_exact(x))
                                                    .collect::<Vec<_>>());

        let (weights, points) = quadrature::total_order::quadrilateral(11).unwrap();
        let error = estimate_element_L2_error(
            &element,
            &|x: &Point2<_>| Vector1::new(u_exact(x)),
            MatrixView::from(&u_weights),
            &weights,
            &points,
            &mut IntegrationWorkspace::default());

        assert_scalar_eq!(error, 0.0, comp=abs, tol=element.diameter() * 1e-12);
    }
}

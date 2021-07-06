use fenris::element::{Tet20Element, Tet4Element, VolumetricFiniteElement};
use fenris::error::{estimate_element_H1_semi_error, estimate_element_L2_error};
use fenris::nalgebra::coordinates::XYZ;
use fenris::nalgebra::{
    DMatrix, DVector, DVectorSlice, Dynamic, MatrixSliceMut, Point3, Vector1, Vector2, U3,
};
use fenris::quadrature;
use fenris::quadrature::{Quadrature, QuadraturePair3d};
use matrixcompare::assert_scalar_eq;
use nalgebra::{Matrix3x2, Vector3};
use std::ops::Deref;
use util::flatten_vertically;

// TODO: Port this to the library proper?
fn transform_quadrature_to_physical_domain<Element>(
    element: &Element,
    weights: &[f64],
    points: &[Point3<f64>],
) -> QuadraturePair3d<f64>
where
    Element: VolumetricFiniteElement<f64, GeometryDim = U3>,
{
    weights
        .iter()
        .zip(points)
        .map(|(w, xi)| {
            let j_det = element.reference_jacobian(xi).determinant().abs();
            (w * j_det, element.map_reference_coords(xi))
        })
        .unzip()
}

fn arbitrary_tet20_element() -> Tet20Element<f64> {
    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);
    Tet20Element::from(&tet4_element)
}

#[test]
#[allow(non_snake_case)]
fn test_element_L2_error_scalar() {
    // We define two functions u1 and u2. u1 is a high order polynomial and u2 is a polynomial
    // that can be exactly represented in some element (Tet20 in this case).
    // Then, given a quadrature rule on the element, we can compute the integral of the function
    //  ||e||^2       e = u1 - u2
    // in two ways:
    //  1. directly with a high order quadrature
    //  2. by computing the L2 error with u = u1 and u_h = u2
    // This allows us to test the L2 error computation routine.
    let element = arbitrary_tet20_element();
    let u_h_element = DVector::from_vec(element.vertices().iter().map(u2_scalar).collect());

    // Use a quadrature rule with sufficient strength such that it can exactly capture the error
    // (since we compute a squared norm, we need double the polynomial degree)
    let (weights, points) = quadrature::total_order::tetrahedron(10).unwrap();
    let mut basis_buffer = vec![3.0; 20];
    let L2_error_computed = estimate_element_L2_error(
        &element,
        |x| Vector1::new(u1_scalar(x)),
        DVectorSlice::from(&u_h_element),
        &weights,
        &points,
        &mut basis_buffer,
    );

    let L2_error_expected = {
        let (weights, points) =
            transform_quadrature_to_physical_domain(&element, &weights, &points);
        let u_squared_norm = |x: &Point3<f64>| u_scalar(x).powi(2);
        (weights, points).integrate(u_squared_norm).sqrt()
    };

    assert_scalar_eq!(
        L2_error_computed,
        L2_error_expected,
        comp = abs,
        tol = 1e-12
    );
}

#[test]
#[allow(non_snake_case)]
fn test_element_L2_error_vector() {
    // This test is completely analogous to the scalar test, it just tests vector-valued
    // functions intead

    let element = arbitrary_tet20_element();
    let u_h_element =
        flatten_vertically(&element.vertices().iter().map(u2_vector).collect::<Vec<_>>()).unwrap();

    // Use a quadrature rule with sufficient strength such that it can exactly capture the error
    // (since we compute a squared norm, we need double the polynomial degree)
    let (weights, points) = quadrature::total_order::tetrahedron(10).unwrap();
    let mut basis_buffer = vec![3.0; 20];
    let L2_error_computed = estimate_element_L2_error(
        &element,
        u1_vector,
        DVectorSlice::from(&u_h_element),
        &weights,
        &points,
        &mut basis_buffer,
    );

    let L2_error_expected = {
        let (weights, points) =
            transform_quadrature_to_physical_domain(&element, &weights, &points);
        let u_squared_norm = |x: &Point3<f64>| u_vector(x).norm_squared();
        (weights, points).integrate(u_squared_norm).sqrt()
    };

    assert_scalar_eq!(
        L2_error_computed,
        L2_error_expected,
        comp = abs,
        tol = 1e-12
    );
}

#[test]
#[allow(non_snake_case)]
fn test_element_H1_seminorm_error_scalar() {
    // We take the same approach as in the L2 tests, except in this case of course we're
    // dealing with gradients of `u` rather than `u` itself.
    // See those tests for comments on what is going on here.
    let element = arbitrary_tet20_element();
    let u_h_element = DVector::from_vec(element.vertices().iter().map(u2_scalar).collect());

    // Use a quadrature rule with sufficient strength such that it can exactly capture the error
    // (since we compute a squared norm, we need double the polynomial degree of the gradient
    // polynomial order)
    let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
    let mut gradient_buffer = DMatrix::repeat(3, 20, 3.0).reshape_generic(U3, Dynamic::new(20));
    let H1_seminorm_computed = estimate_element_H1_semi_error(
        &element,
        u1_scalar_grad,
        DVectorSlice::from(&u_h_element),
        &weights,
        &points,
        MatrixSliceMut::from(&mut gradient_buffer),
    );

    let H1_seminorm_expected = {
        let (weights, points) =
            transform_quadrature_to_physical_domain(&element, &weights, &points);
        let u_squared_norm = |x: &Point3<f64>| u_scalar_grad(x).norm_squared();
        (weights, points).integrate(u_squared_norm).sqrt()
    };

    assert_scalar_eq!(
        H1_seminorm_computed,
        H1_seminorm_expected,
        comp = abs,
        tol = 1e-12
    );
}

#[test]
#[allow(non_snake_case)]
fn test_element_H1_seminorm_error_vector() {
    // This test is completely analogous to the scalar test, it just tests vector-valued
    // functions instead
    let element = arbitrary_tet20_element();
    let u_h_element =
        flatten_vertically(&element.vertices().iter().map(u2_vector).collect::<Vec<_>>()).unwrap();

    // Use a quadrature rule with sufficient strength such that it can exactly capture the error
    // (since we compute a squared norm, we need double the polynomial degree)
    let (weights, points) = quadrature::total_order::tetrahedron(10).unwrap();
    let mut gradient_buffer = DMatrix::repeat(3, 20, 3.0).reshape_generic(U3, Dynamic::new(20));
    let H1_seminorm_computed = estimate_element_H1_semi_error(
        &element,
        u1_vector_grad,
        DVectorSlice::from(&u_h_element),
        &weights,
        &points,
        MatrixSliceMut::from(&mut gradient_buffer),
    );

    let H1_seminorm_expected = {
        let (weights, points) =
            transform_quadrature_to_physical_domain(&element, &weights, &points);
        let u_squared_norm = |x: &Point3<f64>| u_vector_grad(x).norm_squared();
        (weights, points).integrate(u_squared_norm).sqrt()
    };

    assert_scalar_eq!(
        H1_seminorm_computed,
        H1_seminorm_expected,
        comp = abs,
        tol = 1e-12
    );
}

/// An arbitrary multi-variate scalar function used in tests.
fn u1_scalar(x: &Point3<f64>) -> f64 {
    let &XYZ { x, y, z } = x.deref();
    // A polynomial of total order 5
    6.0 * x.powi(5) + 2.0 * y.powi(5) - 2.0 * z.powi(5) + 3.0 * x * y.powi(3) * z - 2.0 * y.powi(3)
        + x.powi(2) * y.powi(2)
        + 3.0 * x
        + 2.0 * y
        - 3.0 * z
        - 6.0
}

/// An arbitrary multi-variate scalar function used in tests.
fn u2_scalar(x: &Point3<f64>) -> f64 {
    let &XYZ { x, y, z } = x.deref();
    // A polynomial of total order 3
    6.0 * x.powi(3) - 2.0 * y.powi(3)
        + 4.0 * z.powi(3)
        + 2.0 * x.powi(2) * y
        + 4.0 * y.powi(2) * z
        + x.powi(2)
        - y.powi(3)
        + 5.0 * x * y * z
        + 2.0 * x
        + 3.0 * y
        - 5.0 * z
        + 2.0
}

/// An arbitrary multi-variate scalar function used in tests.
fn u_scalar(x: &Point3<f64>) -> f64 {
    u1_scalar(x) - u2_scalar(x)
}

/// An arbitrary multi-variate vector function used in tests.
fn u1_vector(x: &Point3<f64>) -> Vector2<f64> {
    let &XYZ { x, y, z } = x.deref();
    // A polynomial of total order 5
    let u1_1 = 6.0 * x.powi(5) + 2.0 * y.powi(5) - 2.0 * z.powi(5) + 3.0 * x * y.powi(3) * z
        - 2.0 * y.powi(3)
        + x.powi(2) * y.powi(2)
        + 3.0 * x
        + 2.0 * y
        - 3.0 * z
        - 6.0;
    let u1_2 = 3.0 * x.powi(5) - 3.0 * y.powi(5)
        + 2.0 * z.powi(5)
        + 3.0 * x.powi(3) * y * z
        + 4.0 * x
        + 2.0 * y
        + 15.0;
    Vector2::new(u1_1, u1_2)
}

/// An arbitrary multi-variate vector function used in tests.
fn u2_vector(x: &Point3<f64>) -> Vector2<f64> {
    let &XYZ { x, y, z } = x.deref();
    // A polynomial of total order 3
    let u2_1 = 6.0 * x.powi(3) - 2.0 * y.powi(3)
        + 4.0 * z.powi(3)
        + 2.0 * x.powi(2) * y
        + 4.0 * y.powi(2) * z
        + x.powi(2)
        - y.powi(3)
        + 5.0 * x * y * z
        + 2.0 * x
        + 3.0 * y
        - 5.0 * z
        + 2.0;
    let u2_2 =
        3.0 * x.powi(3) - 4.0 * y.powi(3) + 2.0 * z.powi(3) + 2.0 * x.powi(2) * z + 3.0 * y.powi(2)
            - 2.0 * x
            + 3.0 * y
            - 5.0 * z
            + 9.0;
    Vector2::new(u2_1, u2_2)
}

/// An arbitrary multi-variate vector function used in tests.
fn u_vector(x: &Point3<f64>) -> Vector2<f64> {
    u1_vector(x) - u2_vector(x)
}

fn u1_scalar_grad(x: &Point3<f64>) -> Vector3<f64> {
    let &XYZ { x, y, z } = x.deref();
    let u1_x = 30.0 * x.powi(5) + 3.0 * y.powi(3) * z + 2.0 * x + 3.0;
    let u1_y = 10.0 * y.powi(4) + 9.0 * x * y.powi(2) * z - 6.0 * y + 2.0 * x.powi(2) * y + 2.0;
    let u1_z = -10.0 * z.powi(4) + 3.0 * x * y.powi(3) - 3.0;
    Vector3::new(u1_x, u1_y, u1_z)
}

fn u2_scalar_grad(x: &Point3<f64>) -> Vector3<f64> {
    let &XYZ { x, y, z } = x.deref();
    let u2_x = 18.0 * x.powi(2) + 4.0 * y * x + 2.0 * x + 5.0 * y * z + 2.0;
    let u2_y =
        -6.0 * y.powi(2) + 2.0 * x.powi(2) + 8.0 * z * y - 3.0 * y.powi(2) + 5.0 * x * z + 3.0;
    let u2_z = 12.0 * z.powi(2) + 4.0 * y.powi(2) + 5.0 * x * y - 5.0;
    Vector3::new(u2_x, u2_y, u2_z)
}

fn u_scalar_grad(x: &Point3<f64>) -> Vector3<f64> {
    u1_scalar_grad(x) - u2_scalar_grad(x)
}

fn u1_vector_grad(x: &Point3<f64>) -> Matrix3x2<f64> {
    let &XYZ { x, y, z } = x.deref();
    let u1_1x = 30.0 * x.powi(4) + 3.0 * y.powi(3) * z + 2.0 * x * y.powi(2) + 3.0;
    let u1_1y =
        10.0 * y.powi(4) + 9.0 * x * z * y.powi(2) - 6.0 * y.powi(2) + 2.0 * x.powi(2) * y + 2.0;
    let u1_1z = -10.0 * z.powi(4) + 3.0 * x * y.powi(3) - 3.0;

    let u1_2x = 15.0 * x.powi(4) + 9.0 * y * z * x.powi(2) + 4.0;
    let u1_2y = -15.0 * y.powi(4) + 3.0 * x.powi(3) * z + 2.0;
    let u1_2z = 10.0 * z.powi(4) + 3.0 * x.powi(3) * y;

    let u1_1_grad = Vector3::new(u1_1x, u1_1y, u1_1z);
    let u1_2_grad = Vector3::new(u1_2x, u1_2y, u1_2z);

    Matrix3x2::from_columns(&[u1_1_grad, u1_2_grad])
}

fn u2_vector_grad(x: &Point3<f64>) -> Matrix3x2<f64> {
    let &XYZ { x, y, z } = x.deref();
    let u2_1x = 18.0 * x.powi(2) + 4.0 * y * x + 2.0 * x + 5.0 * y * z + 2.0;
    let u2_1y =
        -6.0 * y.powi(2) + 2.0 * x.powi(2) + 8.0 * z * y - 3.0 * y.powi(2) + 5.0 * x * z + 3.0;
    let u2_1z = 12.0 * z.powi(2) + 4.0 * y.powi(2) + 5.0 * x * y - 5.0;

    let u2_2x = 9.0 * x.powi(2) + 4.0 * x * z - 2.0;
    let u2_2y = -12.0 * y.powi(2) + 6.0 * y + 3.0;
    let u2_2z = 6.0 * z.powi(2) + 2.0 * x.powi(2) - 5.0;

    let u2_1_grad = Vector3::new(u2_1x, u2_1y, u2_1z);
    let u2_2_grad = Vector3::new(u2_2x, u2_2y, u2_2z);

    Matrix3x2::from_columns(&[u2_1_grad, u2_2_grad])
}

fn u_vector_grad(x: &Point3<f64>) -> Matrix3x2<f64> {
    u1_vector_grad(x) - u2_vector_grad(x)
}

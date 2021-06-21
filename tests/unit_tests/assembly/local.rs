use fenris::allocators::{BiDimAllocator, SmallDimAllocator};
use fenris::assembly::local::{
    assemble_element_elliptic_matrix, assemble_element_elliptic_vector,
    assemble_element_source_vector, assemble_generalized_element_mass,
    compute_element_elliptic_energy, SourceFunction,
};
use fenris::assembly::operators::{
    EllipticContraction, EllipticEnergy, EllipticOperator, Operator,
};
use fenris::element::{
    MatrixSlice, MatrixSliceMut, Quad4d2Element, ReferenceFiniteElement, Tet10Element, Tet4Element,
    VolumetricFiniteElement,
};
use fenris::geometry::Quad2d;
use fenris::nalgebra::coordinates::{XY, XYZ};
use fenris::nalgebra::{
    DMatrix, DVector, DVectorSliceMut, DefaultAllocator, DimName, Dynamic, Matrix3x2, Matrix4,
    MatrixMN, MatrixN, Point, Point2, RealField, VectorN, U1, U2, U3, U8,
};
use fenris::quadrature;
use fenris::quadrature::{Quadrature, QuadraturePair};
use fenris_optimize::calculus::{approximate_gradient_fd, approximate_jacobian_fd};
use itertools::izip;
use matrixcompare::{assert_matrix_eq, assert_scalar_eq};
use nalgebra::{DVectorSlice, Matrix2, Point3, Vector1, Vector2, Vector3};
use num::Zero;
use std::ops::Deref;

fn reference_quad<T>() -> Quad2d<T>
where
    T: RealField,
{
    Quad2d([
        Point2::new(-T::one(), -T::one()),
        Point2::new(T::one(), -T::one()),
        Point2::new(T::one(), T::one()),
        Point2::new(-T::one(), T::one()),
    ])
}

#[test]
fn analytic_comparison_of_element_mass_matrix_for_reference_element() {
    let density = 3.0;

    let quadrature = quadrature::total_order::quadrilateral(5).unwrap();
    let quad = Quad4d2Element::from(reference_quad());

    let ndof = 8;
    let mut m = DMatrix::zeros(ndof, ndof);
    assemble_generalized_element_mass::<_, U2, _, _>(
        MatrixSliceMut::from(&mut m),
        &quad,
        density,
        &quadrature,
    );

    #[rustfmt::skip]
    let expected4x4 = (density / 9.0) * Matrix4::new(
        4.0, 2.0, 1.0, 2.0,
        2.0, 4.0, 2.0, 1.0,
        1.0, 2.0, 4.0, 2.0,
        2.0, 1.0, 2.0, 4.0);

    let mut expected8x8: MatrixN<f64, U8> = MatrixN::zero();
    expected8x8
        .slice_with_steps_mut((0, 0), (4, 4), (1, 1))
        .copy_from(&expected4x4);
    expected8x8
        .slice_with_steps_mut((1, 1), (4, 4), (1, 1))
        .copy_from(&expected4x4);

    let diff = m - expected8x8;
    assert!(diff.norm() <= 1e-6);
}

struct MockScalarEllipticEnergy;

impl Operator for MockScalarEllipticEnergy {
    type SolutionDim = U1;
    type Parameters = ();
}

impl EllipticEnergy<f64, U2> for MockScalarEllipticEnergy {
    fn compute_energy(
        &self,
        gradient: &MatrixMN<f64, U2, Self::SolutionDim>,
        _parameters: &Self::Parameters,
    ) -> f64 {
        3.0 * gradient[0] - 2.0 * gradient[1]
    }
}

fn u_scalar_linear(x: &Point2<f64>) -> Vector1<f64> {
    let &XY { x, y } = x.deref();
    Vector1::new(4.0 * x - 2.0 * y + 7.0)
}

fn u_scalar_linear_grad(_x: &Point2<f64>) -> Vector2<f64> {
    Vector2::new(4.0, -2.0)
}

fn u_scalar_bilinear(x: &Point2<f64>) -> Vector1<f64> {
    let &XY { x, y } = x.deref();
    Vector1::new((4.0 * x + 3.0) * (-3.0 * y + 2.0))
}

fn u_scalar_bilinear_grad(x: &Point2<f64>) -> Vector2<f64> {
    let &XY { x, y } = x.deref();
    Vector2::new(4.0 * (-3.0 * y + 2.0), -3.0 * (4.0 * x + 3.0))
}

/// Constructs a specific quadrature rule for the given element by transforming an input
/// quadrature rule for the reference element.
fn construct_quadrature_rule_for_element<Element>(
    element: &Element,
    reference_rule: &QuadraturePair<f64, Element::GeometryDim>,
) -> QuadraturePair<f64, Element::GeometryDim>
where
    Element: VolumetricFiniteElement<f64>,
    DefaultAllocator: SmallDimAllocator<f64, Element::GeometryDim>,
{
    // Construct a quadrature rule for this particular element
    let (weights, points) = reference_rule;
    izip!(weights, points)
        .map(|(w, p)| {
            let x = element.map_reference_coords(&p);
            let j = element.reference_jacobian(&p);
            let new_w = w * j.determinant().abs();
            (new_w, x)
        })
        .unzip()
}

fn compute_expected_energy_integral<Element, Energy, UGrad>(
    element: &Element,
    energy: &Energy,
    u_grad: UGrad,
    reference_rule: &QuadraturePair<f64, Element::GeometryDim>,
) -> f64
where
    Element: VolumetricFiniteElement<f64>,
    Energy: EllipticEnergy<f64, Element::GeometryDim, Parameters = ()>,
    UGrad: Fn(
        &Point<f64, Element::GeometryDim>,
    ) -> MatrixMN<f64, Element::GeometryDim, Energy::SolutionDim>,
    DefaultAllocator: BiDimAllocator<f64, Element::GeometryDim, Energy::SolutionDim>,
{
    let quadrature_rule = construct_quadrature_rule_for_element(element, reference_rule);
    // Assuming f is a polynomial function (i.e. the energy is a polynomial in terms of the
    // components of u_grad and u_grad is polynomial), then we can hopefully compute this integral
    // exactly provided that the reference rule is sufficiently accurate
    let f = |x: &Point<f64, Element::GeometryDim>| energy.compute_energy(&u_grad(x), &());
    let integral_expected = quadrature_rule.integrate(f);
    integral_expected
}

fn compute_energy_integral<Element, Energy>(
    element: &Element,
    energy: &Energy,
    u_element: DVectorSlice<f64>,
    quadrature: &QuadraturePair<f64, Element::GeometryDim>,
) -> f64
where
    Element: VolumetricFiniteElement<f64>,
    Energy: EllipticEnergy<f64, Element::GeometryDim, Parameters = ()>,
    DefaultAllocator: BiDimAllocator<f64, Element::GeometryDim, Energy::SolutionDim>,
{
    let (weights, points) = quadrature;
    let quadrature_params = vec![(); weights.len()];

    let d = Element::GeometryDim::dim();
    let mut gradient_buffer = MatrixMN::from_vec_generic(
        Element::GeometryDim::name(),
        Dynamic::new(element.num_nodes()),
        // Fill buffer with something non-zero to check that the implementation isn't
        // somehow dependent on this
        vec![3.0; d * element.num_nodes()],
    );
    compute_element_elliptic_energy(
        element,
        energy,
        DVectorSlice::from(u_element),
        &weights,
        &points,
        &quadrature_params,
        MatrixSliceMut::from(&mut gradient_buffer),
    )
    .unwrap()
}

fn u_element_from_vertices_and_u_exact<D, S>(
    vertices: &[Point<f64, D>],
    u_exact: impl Fn(&Point<f64, D>) -> VectorN<f64, S>,
) -> DVector<f64>
where
    D: DimName,
    S: DimName,
    DefaultAllocator: BiDimAllocator<f64, D, S>,
{
    let mut entries = Vec::with_capacity(D::dim());
    for v in vertices {
        let u = u_exact(v);
        entries.extend(u.iter());
    }
    DVector::from_vec(entries)
}

#[test]
fn compute_element_energy_scalar_quad4() {
    // Test that we can exactly integrate a mock elliptic energy over a given element.
    // By choosing an a priori known function `u` we can determine the expected answer with
    // high-order numerical integration.
    // Then, by interpolating `u` on the vertices of the element with the standard nodal
    // interpolation, we would expect to get the same result from our element-wise energy
    // computation routines, provided that the element is able to reproduce `u` exactly.

    // TODO: Include parameters in the test? Currently we don't test this at all...

    {
        // Rectangular element: here we should be able to reproduce any bilinear function
        let element = Quad4d2Element::from_vertices([
            Point2::new(-6.0, -2.0),
            Point2::new(2.0, -2.0),
            Point2::new(2.0, 4.0),
            Point2::new(-6.0, 4.0),
        ]);

        let u_element = u_element_from_vertices_and_u_exact(element.vertices(), u_scalar_bilinear);

        let quadrature = quadrature::tensor::quadrilateral_gauss(2);
        let integral_computed = compute_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            DVectorSlice::from(&u_element),
            &quadrature,
        );

        let reference_quadrature = quadrature::total_order::quadrilateral(8).unwrap();
        let integral_expected = compute_expected_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            u_scalar_bilinear_grad,
            &reference_quadrature,
        );

        assert_scalar_eq!(
            integral_computed,
            integral_expected,
            comp = abs,
            tol = 1e-12
        );
    }

    {
        // General quadrilateral element: here we can not reproduce any arbitrary bilinear function,
        // so we instead use a linear one
        let element = Quad4d2Element::from_vertices([
            Point2::new(-6.0, -2.0),
            Point2::new(2.0, 0.0),
            Point2::new(4.0, 6.0),
            Point2::new(-8.0, 4.0),
        ]);

        let u_element = u_element_from_vertices_and_u_exact(element.vertices(), u_scalar_linear);

        let quadrature = quadrature::tensor::quadrilateral_gauss(2);
        let integral_computed = compute_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            DVectorSlice::from(&u_element),
            &quadrature,
        );

        let reference_quadrature = quadrature::total_order::quadrilateral(8).unwrap();
        let integral_expected = compute_expected_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            u_scalar_linear_grad,
            &reference_quadrature,
        );

        assert_scalar_eq!(
            integral_computed,
            integral_expected,
            comp = abs,
            tol = 1e-12
        );
    }
}

struct MockVectorEllipticEnergy;

impl Operator for MockVectorEllipticEnergy {
    type SolutionDim = U2;
    type Parameters = ();
}

impl EllipticEnergy<f64, U3> for MockVectorEllipticEnergy {
    fn compute_energy(&self, gradient: &Matrix3x2<f64>, _parameters: &Self::Parameters) -> f64 {
        // Use the log here to make sure our function is not so simple that the
        // contraction is independent of the gradient
        gradient.dot(&(gradient)).ln()
    }
}

impl EllipticOperator<f64, U3> for MockVectorEllipticEnergy {
    fn compute_elliptic_term(
        &self,
        gradient: &MatrixMN<f64, U3, Self::SolutionDim>,
        _data: &Self::Parameters,
    ) -> MatrixMN<f64, U3, Self::SolutionDim> {
        2.0 * gradient / (gradient.dot(&gradient))
    }
}

impl EllipticContraction<f64, U3> for MockVectorEllipticEnergy {
    #[allow(non_snake_case)]
    fn contract(
        &self,
        gradient: &MatrixMN<f64, U3, Self::SolutionDim>,
        _data: &Self::Parameters,
        a: &VectorN<f64, U3>,
        b: &VectorN<f64, U3>,
    ) -> MatrixMN<f64, Self::SolutionDim, Self::SolutionDim> {
        let G = gradient;
        let G_dot_G = G.dot(&G);

        let t = a.dot(&b) * G_dot_G * Matrix2::identity();
        let u = 2.0 * G.transpose() * a * b.transpose() * G;
        (2.0 / G_dot_G.powi(2)) * (t - u)
    }
}

fn u_vector_quadratic(x: &Point3<f64>) -> Vector2<f64> {
    let &XYZ { x, y, z } = x.deref();
    Vector2::new(
        2.0 * x * x + 3.0 * y - 4.0 * x * y - z * x + z * z + 3.0,
        3.0 * x * z + 4.0 * y * y - y * z + 2.0,
    )
}

fn u_vector_quadratic_grad(x: &Point3<f64>) -> Matrix3x2<f64> {
    let &XYZ { x, y, z } = x.deref();
    let u_1_grad = Vector3::new(4.0 * x - 4.0 * y - z, 3.0 - 4.0 * x, -x + 2.0 * z);
    let u_2_grad = Vector3::new(3.0 * z, 8.0 * y - z, 3.0 * x - y);
    Matrix3x2::from_columns(&[u_1_grad, u_2_grad])
}

#[test]
fn compute_element_energy_vector_tet10() {
    // Test that we can exactly integrate a mock elliptic energy over a given element.
    // By choosing an a priori known function `u` we can determine the expected answer with
    // high-order numerical integration.
    // Then, by interpolating `u` on the vertices of the element with the standard nodal
    // interpolation, we would expect to get the same result from our element-wise energy
    // computation routines, provided that the element is able to reproduce `u` exactly.

    // TODO: Include parameters in the test? Currently we don't test this at all...
    {
        let a = Point3::new(2.0, 0.0, 1.0);
        let b = Point3::new(3.0, 4.0, 1.0);
        let c = Point3::new(1.0, 1.0, 2.0);
        let d = Point3::new(3.0, 1.0, 4.0);
        let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

        // A Tet10 element can reproduce any quadratic solution field exactly
        let element = Tet10Element::from(&tet4_element);

        let u_element = u_element_from_vertices_and_u_exact(element.vertices(), u_vector_quadratic);

        let quadrature = quadrature::total_order::tetrahedron(8).unwrap();
        let integral_computed = compute_energy_integral(
            &element,
            &MockVectorEllipticEnergy,
            DVectorSlice::from(&u_element),
            &quadrature,
        );

        let reference_quadrature = quadrature::total_order::tetrahedron(8).unwrap();
        let integral_expected = compute_expected_energy_integral(
            &element,
            &MockVectorEllipticEnergy,
            u_vector_quadratic_grad,
            &reference_quadrature,
        );

        assert_scalar_eq!(integral_computed, integral_expected, comp = abs, tol = 1e-8);
    }
}

#[test]
fn elliptic_element_vector_is_gradient_of_energy_tet10() {
    // The element vector associated with an elliptic operator should work out
    // to be exactly the gradient of the associated elliptic energy. We check this here
    // using finite differences

    // TODO: Test that it works with parameters?

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // A Tet10 element can reproduce any quadratic solution field exactly
    let element = Tet10Element::from(&tet4_element);
    let u_element = u_element_from_vertices_and_u_exact(element.vertices(), u_vector_quadratic);

    // Let f(u_element) = energy(grad u). Then we can compute an approximate derivative with finite
    // differences and use this to compare with our output from the assembly, which should
    // be exactly the same.
    let finite_diff_result = {
        let quadrature = quadrature::total_order::tetrahedron(8).unwrap();
        let f = |u: DVectorSlice<f64>| {
            compute_energy_integral(&element, &MockVectorEllipticEnergy, u, &quadrature)
        };
        // TODO: What to use as h?
        let mut u_element = u_element.clone();
        approximate_gradient_fd(f, &mut u_element, 1e-6)
    };

    let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
    let quadrature_data = vec![(); weights.len()];
    let mut output = DVector::repeat(2 * element.num_nodes(), 3.0);
    let mut gradient_buffer = DMatrix::repeat(3, element.num_nodes(), 3.0)
        .reshape_generic(U3, Dynamic::new(element.num_nodes()));
    assemble_element_elliptic_vector(
        MatrixSliceMut::from(&mut output),
        &element,
        &MockVectorEllipticEnergy,
        MatrixSlice::from(&u_element),
        &weights,
        &points,
        &quadrature_data,
        MatrixSliceMut::from(&mut gradient_buffer),
    )
    .unwrap();

    assert_matrix_eq!(output, finite_diff_result, comp = abs, tol = 1e-6);
}

#[test]
fn elliptic_element_matrix_is_jacobian_of_vector_tet10() {
    // The element matrix associated with an elliptic operator should work out
    // to be exactly the Jacobian of the associated element vector.
    // We check this with finite differences

    // TODO: Test that it works with parameters?

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // A Tet10 element can reproduce any quadratic solution field exactly
    let element = Tet10Element::from(&tet4_element);
    let u_element = u_element_from_vertices_and_u_exact(element.vertices(), u_vector_quadratic);

    // Let f(u_element) = energy(grad u). Then we can compute an approximate derivative with finite
    // differences and use this to compare with our output from the assembly, which should
    // be exactly the same.
    let finite_diff_result = {
        let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
        let quadrature_data = vec![(); weights.len()];

        // Set up a function f = f(u) that corresponds to the element vector given state u
        let f = |u: DVectorSlice<f64>, output: DVectorSliceMut<f64>| {
            let mut gradient_buffer = DMatrix::repeat(3, element.num_nodes(), 3.0)
                .reshape_generic(U3, Dynamic::new(element.num_nodes()));
            assemble_element_elliptic_vector(
                output,
                &element,
                &MockVectorEllipticEnergy,
                u,
                &weights,
                &points,
                &quadrature_data,
                MatrixSliceMut::from(&mut gradient_buffer),
            )
            .unwrap();
        };

        // TODO: What should h be?
        approximate_jacobian_fd(2 * element.num_nodes(), f, &mut u_element.clone(), 1e-6)
    };

    let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
    let quadrature_data = vec![(); weights.len()];
    let mut output = DMatrix::repeat(2 * element.num_nodes(), 2 * element.num_nodes(), 3.0);
    let mut gradient_buffer = DMatrix::repeat(3, element.num_nodes(), 3.0)
        .reshape_generic(U3, Dynamic::new(element.num_nodes()));
    assemble_element_elliptic_matrix(
        MatrixSliceMut::from(&mut output),
        &element,
        &MockVectorEllipticEnergy,
        MatrixSlice::from(&u_element),
        &weights,
        &points,
        &quadrature_data,
        MatrixSliceMut::from(&mut gradient_buffer),
    )
    .unwrap();

    assert_matrix_eq!(output, finite_diff_result, comp = abs, tol = 1e-6);
}

#[test]
fn element_source_vector_reproduces_inner_product() {
    // We wish to test our procedure for computing element source vectors stemming from the
    // weak form term (f, v) for a smooth function f = f(x) and test function v. The routine
    // produces a vector f_I associated with each node in the element K corresponding to the
    // integral
    //  f_I := int_K f phi_I dx
    // where phi_I is the basis function associated with node I. It's cumbersome to compute the
    // integral manually for verification, so instead we adopt the following procedure:
    //
    // Let u = u(x) be a smooth function that can be exactly reproduced by the nodal interpolation
    // on the element K. For example if u is a quadratic function, we can use a quadratic
    // tetrahedral element to reproduce it. Let u_h denote the nodal interpolation. We can write
    //  u_h = sum_I u_I phi_I
    // where u_I is the "weight" associated with each basis function phi_I.
    //
    // Then we observe that
    //  int_K f dot u dx = int_K f dot u_h dx
    //                   = \sum_I u_I dot \int_K f \phi_I dx
    //                   = \sum_I u_I dot f_I
    //                   = u_K dot f_K
    // where the vectors u_K and f_K correspond to the vectors u_K = [u_1, ..., u_n] and
    // f_K = [f_1, ..., f_n] is the element source vector for n nodes in element K.
    //
    // Since we can compute the integral on the left hand side by high-order numerical quadrature
    // we directly have a verifiable criteria to check the element source vector f_K against.
    let u = |x: &Point3<f64>| {
        let &XYZ { x, y, z } = x.deref();
        let u1 = 3.0 * x * x - 4.0 * x * y + 3.0 * x * z - z * z + 5.0;
        let u2 = 3.0 * x + 3.0 * y * z - 2.0 * y + z * y - 3.0;
        Vector2::new(u1, u2)
    };

    fn f(x: &Point3<f64>) -> Vector2<f64> {
        let &XYZ { x, y, z } = x.deref();
        let f1 = 6.0 * x * x - 4.0 * x * z + 3.0 - z * z - x + y - 3.0;
        let f2 = 2.0 * x + 3.0 * x * (y - z) - x * y - 2.0 * z * z + 5.0;
        Vector2::new(f1, f2)
    }

    struct MockSourceFunction;

    impl Operator for MockSourceFunction {
        type SolutionDim = U2;
        type Parameters = ();
    }

    impl SourceFunction<f64, U3> for MockSourceFunction {
        fn evaluate(&self, coords: &Point<f64, U3>, _data: &Self::Parameters) -> Vector2<f64> {
            f(coords)
        }
    }

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // A Tet10 element can reproduce any quadratic solution field exactly
    let element = Tet10Element::from(&tet4_element);
    let u_element = u_element_from_vertices_and_u_exact(element.vertices(), u);

    let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
    let quadrature_data = vec![(); weights.len()];
    let mut basis_buffer = vec![0.0; element.num_nodes()];
    let mut f_element = DVector::repeat(u_element.len(), 2.0);
    assemble_element_source_vector(
        DVectorSliceMut::from(&mut f_element),
        &element,
        &MockSourceFunction,
        &weights,
        &points,
        &quadrature_data,
        &mut basis_buffer,
    );

    // Compute the inner product (u, f) on the element with high order quadrature
    let expected_inner_product = {
        // u and f are both quadratic functions, so the product is of order 4
        let reference_rule = quadrature::total_order::tetrahedron(4).unwrap();
        let quadrature_rule = construct_quadrature_rule_for_element(&element, &reference_rule);
        quadrature_rule.integrate(|x| f(x).dot(&u(x)))
    };

    let computed_inner_product = u_element.dot(&f_element);

    assert_scalar_eq!(
        computed_inner_product,
        expected_inner_product,
        comp = abs,
        tol = 1e-12
    );
}

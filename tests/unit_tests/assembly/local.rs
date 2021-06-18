use fenris::assembly::local::{assemble_generalized_element_mass, compute_element_elliptic_energy};
use fenris::assembly::operators::{EllipticEnergy, Operator};
use fenris::element::{MatrixSliceMut, Quad4d2Element, VolumetricFiniteElement, Tet10Element, Tet4Element};
use fenris::geometry::Quad2d;
use fenris::nalgebra::coordinates::{XY, XYZ};
use fenris::nalgebra::{DMatrix, DVector, Dynamic, Matrix4, MatrixMN, MatrixN, Point2, RealField, U1, U2, U8, U3, Matrix3x2, DimName, DefaultAllocator, Point, VectorN};
use fenris::quadrature;
use fenris::quadrature::{Quadrature, QuadraturePair};
use itertools::izip;
use matrixcompare::assert_scalar_eq;
use nalgebra::{DVectorSlice, Vector2, Point3, Vector1, Vector3};
use num::Zero;
use std::ops::Deref;
use fenris::allocators::{SmallDimAllocator, BiDimAllocator};
use fenris::nalgebra_sparse::na::Matrix3;

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
    DefaultAllocator: SmallDimAllocator<f64, Element::GeometryDim>
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
    UGrad: Fn(&Point<f64, Element::GeometryDim>) -> MatrixMN<f64, Element::GeometryDim, Energy::SolutionDim>,
    DefaultAllocator: BiDimAllocator<f64, Element::GeometryDim, Energy::SolutionDim>
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
    u_element: &DVector<f64>,
    quadrature: &QuadraturePair<f64, Element::GeometryDim>,
) -> f64
where
    Element: VolumetricFiniteElement<f64>,
    Energy: EllipticEnergy<f64, Element::GeometryDim, Parameters = ()>,
    DefaultAllocator: BiDimAllocator<f64, Element::GeometryDim, Energy::SolutionDim>
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
    DefaultAllocator: BiDimAllocator<f64, D, S>
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
        let integral_computed =
            compute_energy_integral(&element, &MockScalarEllipticEnergy, &u_element, &quadrature);

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
        let integral_computed =
            compute_energy_integral(&element, &MockScalarEllipticEnergy, &u_element, &quadrature);

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
    fn compute_energy(
        &self,
        gradient: &Matrix3x2<f64>,
        _parameters: &Self::Parameters,
    ) -> f64 {
        // TODO: Use matrix! macro when we've been able to upgrade nalgebra...
        let a = Matrix3::new(-1.0, 2.0, 4.0,
                             5.0, -2.0, 0.5,
                             3.0, 2.0, 0.0);

        // An arbitrary quadratic function G^T A G
        let e = gradient.transpose() * a * gradient;
        // e : e
        e.dot(&e)
    }
}

fn u_vector_quadratic(x: &Point3<f64>) -> Vector2<f64> {
    let &XYZ { x, y, z } = x.deref();
    Vector2::new(2.0 * x * x + 3.0 * y - 4.0 * x * y - z * x + z * z + 3.0,
                 3.0 * x * z + 4.0 * y * y - y * z + 2.0)
}

fn u_vector_quadratic_grad(x: &Point3<f64>) -> Matrix3x2<f64> {
    let &XYZ { x, y, z } = x.deref();
    let u_1_grad = Vector3::new(4.0 * x - 4.0 * y - z,
                                3.0 - 4.0 * x,
                                - x + 2.0 * z);
    let u_2_grad = Vector3::new(3.0 * z,
                                8.0 * y - z,
                                3.0 * x - y);
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
        let integral_computed =
            compute_energy_integral(&element, &MockVectorEllipticEnergy, &u_element, &quadrature);

        let reference_quadrature = quadrature::total_order::tetrahedron(8).unwrap();
        let integral_expected = compute_expected_energy_integral(
            &element,
            &MockVectorEllipticEnergy,
            u_vector_quadratic_grad,
            &reference_quadrature,
        );

        assert_scalar_eq!(
            integral_computed,
            integral_expected,
            comp = abs,
            tol = 1e-8
        );
    }
}

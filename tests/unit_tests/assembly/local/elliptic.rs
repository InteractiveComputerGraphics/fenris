use std::ops::Deref;

use matrixcompare::{assert_matrix_eq, assert_scalar_eq};

use fenris::allocators::BiDimAllocator;
use fenris::assembly::local::{
    assemble_element_elliptic_matrix, assemble_element_elliptic_vector, compute_element_elliptic_energy,
    ElementEllipticAssemblerBuilder, ElementMatrixAssembler, ElementScalarAssembler, ElementVectorAssembler,
    GeneralQuadratureTable,
};
use fenris::assembly::operators::{EllipticContraction, EllipticEnergy, EllipticOperator, Operator};
use fenris::element::{
    ElementConnectivity, FiniteElement, MatrixSlice, MatrixSliceMut, Quad4d2Element, ReferenceFiniteElement,
    Tet10Element, Tet4Element, VolumetricFiniteElement,
};
use fenris::nalgebra::base::coordinates::{XY, XYZ};
use fenris::nalgebra::{
    DMatrix, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, DimName, Dynamic, Matrix2, Matrix3x2, OMatrix,
    OPoint, OVector, Point2, Point3, Vector1, Vector2, Vector3, U1, U2, U3,
};
use fenris::quadrature::{Quadrature, QuadraturePair};
use fenris::{quadrature, Symmetry};
use fenris_optimize::calculus::{approximate_gradient_fd, approximate_jacobian_fd};

use crate::unit_tests::assembly::local;
use crate::unit_tests::assembly::local::density;
use fenris::assembly::global::gather_global_to_local;
use fenris::connectivity::Connectivity;
use fenris::mesh::procedural::create_unit_square_uniform_quad_mesh_2d;
use fenris::mesh::QuadMesh2d;
use fenris_nested_vec::NestedVec;
use nalgebra::DMatrixSliceMut;

struct MockScalarEllipticEnergy;

impl Operator<f64, U2> for MockScalarEllipticEnergy {
    type SolutionDim = U1;
    /// Basically density to check that parameters are taken into account during assembly
    type Parameters = f64;
}

impl EllipticEnergy<f64, U2> for MockScalarEllipticEnergy {
    fn compute_energy(&self, gradient: &OMatrix<f64, U2, Self::SolutionDim>, _parameters: &Self::Parameters) -> f64 {
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

#[test]
fn compute_element_energy_scalar_quad4() {
    // Test that we can exactly integrate a mock elliptic energy over a given element.
    // By choosing an a priori known function `u` we can determine the expected answer with
    // high-order numerical integration.
    // Then, by interpolating `u` on the vertices of the element with the standard nodal
    // interpolation, we would expect to get the same result from our element-wise energy
    // computation routines, provided that the element is able to reproduce `u` exactly.

    {
        // Rectangular element: here we should be able to reproduce any bilinear function
        let element = Quad4d2Element::from_vertices([
            Point2::new(-6.0, -2.0),
            Point2::new(2.0, -2.0),
            Point2::new(2.0, 4.0),
            Point2::new(-6.0, 4.0),
        ]);

        let u_element = local::u_element_from_vertices_and_u_exact(element.vertices(), u_scalar_bilinear);

        let quadrature = quadrature::tensor::quadrilateral_gauss(2);
        let quadrature_params = local::evaluate_density_at_quadrature_points(&element, &quadrature.1, local::density);
        let integral_computed = compute_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            DVectorSlice::from(&u_element),
            &quadrature,
            &quadrature_params,
        );

        let reference_quadrature = quadrature::total_order::quadrilateral(8).unwrap();
        let integral_expected = compute_expected_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            u_scalar_bilinear_grad,
            &reference_quadrature,
        );

        assert_scalar_eq!(integral_computed, integral_expected, comp = abs, tol = 1e-12);
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

        let u_element = local::u_element_from_vertices_and_u_exact(element.vertices(), u_scalar_linear);

        let quadrature = quadrature::tensor::quadrilateral_gauss(2);
        let quadrature_params = local::evaluate_density_at_quadrature_points(&element, &quadrature.1, local::density);
        let integral_computed = compute_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            DVectorSlice::from(&u_element),
            &quadrature,
            &quadrature_params,
        );

        let reference_quadrature = quadrature::total_order::quadrilateral(8).unwrap();
        let integral_expected = compute_expected_energy_integral(
            &element,
            &MockScalarEllipticEnergy,
            u_scalar_linear_grad,
            &reference_quadrature,
        );

        assert_scalar_eq!(integral_computed, integral_expected, comp = abs, tol = 1e-12);
    }
}

struct MockVectorEllipticEnergy;

impl Operator<f64, U3> for MockVectorEllipticEnergy {
    type SolutionDim = U2;
    // A sort of "density" whose purpose is to test that the quadrature parameters are taken into
    // account during assembly
    type Parameters = f64;
}

impl EllipticEnergy<f64, U3> for MockVectorEllipticEnergy {
    fn compute_energy(&self, gradient: &Matrix3x2<f64>, density: &Self::Parameters) -> f64 {
        // Use the log here to make sure our function is not so simple that the
        // contraction is independent of the gradient
        density * gradient.dot(&(gradient)).ln()
    }
}

impl EllipticOperator<f64, U3> for MockVectorEllipticEnergy {
    fn compute_elliptic_operator(
        &self,
        gradient: &OMatrix<f64, U3, Self::SolutionDim>,
        density: &Self::Parameters,
    ) -> OMatrix<f64, U3, Self::SolutionDim> {
        *density * (2.0 * gradient / (gradient.dot(&gradient)))
    }
}

impl EllipticContraction<f64, U3> for MockVectorEllipticEnergy {
    #[allow(non_snake_case)]
    fn contract(
        &self,
        gradient: &OMatrix<f64, U3, Self::SolutionDim>,
        a: &OVector<f64, U3>,
        b: &OVector<f64, U3>,
        density: &Self::Parameters,
    ) -> OMatrix<f64, Self::SolutionDim, Self::SolutionDim> {
        let G = gradient;
        let G_dot_G = G.dot(&G);

        let t = a.dot(&b) * G_dot_G * Matrix2::identity();
        let u = 2.0 * G.transpose() * a * b.transpose() * G;
        *density * (2.0 / G_dot_G.powi(2)) * (t - u)
    }
}

/// Same as the non-symmetric version, except we explicitly override `symmetry()` to indicate
/// that the operator is in fact symmetric
struct MockVectorSymmetricEllipticEnergy;

impl Operator<f64, U3> for MockVectorSymmetricEllipticEnergy {
    type SolutionDim = U2;
    type Parameters = f64;
}

impl EllipticOperator<f64, U3> for MockVectorSymmetricEllipticEnergy {
    fn compute_elliptic_operator(
        &self,
        gradient: &OMatrix<f64, U3, Self::SolutionDim>,
        data: &Self::Parameters,
    ) -> OMatrix<f64, U3, Self::SolutionDim> {
        MockVectorEllipticEnergy.compute_elliptic_operator(gradient, data)
    }
}

impl EllipticContraction<f64, U3> for MockVectorSymmetricEllipticEnergy {
    fn contract(
        &self,
        gradient: &OMatrix<f64, U3, Self::SolutionDim>,
        a: &OVector<f64, U3>,
        b: &OVector<f64, U3>,
        parameters: &Self::Parameters,
    ) -> OMatrix<f64, Self::SolutionDim, Self::SolutionDim> {
        MockVectorEllipticEnergy.contract(gradient, a, b, parameters)
    }

    fn symmetry(&self) -> Symmetry {
        Symmetry::Symmetric
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

    {
        let a = Point3::new(2.0, 0.0, 1.0);
        let b = Point3::new(3.0, 4.0, 1.0);
        let c = Point3::new(1.0, 1.0, 2.0);
        let d = Point3::new(3.0, 1.0, 4.0);
        let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

        // A Tet10 element can reproduce any quadratic solution field exactly
        let element = Tet10Element::from(&tet4_element);

        let u_element = local::u_element_from_vertices_and_u_exact(element.vertices(), u_vector_quadratic);

        let quadrature = quadrature::total_order::tetrahedron(8).unwrap();
        let quadrature_params = local::evaluate_density_at_quadrature_points(&element, &quadrature.1, local::density);
        let integral_computed = compute_energy_integral(
            &element,
            &MockVectorEllipticEnergy,
            DVectorSlice::from(&u_element),
            &quadrature,
            &quadrature_params,
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

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // A Tet10 element can reproduce any quadratic solution field exactly
    let element = Tet10Element::from(&tet4_element);
    let u_element = local::u_element_from_vertices_and_u_exact(element.vertices(), u_vector_quadratic);

    // Let f(u_element) = energy(grad u). Then we can compute an approximate derivative with finite
    // differences and use this to compare with our output from the assembly, which should
    // be exactly the same.
    let finite_diff_result = {
        let quadrature = quadrature::total_order::tetrahedron(8).unwrap();
        let quadrature_params = local::evaluate_density_at_quadrature_points(&element, &quadrature.1, local::density);
        let f = |u: DVectorSlice<f64>| {
            compute_energy_integral(&element, &MockVectorEllipticEnergy, u, &quadrature, &quadrature_params)
        };
        // TODO: What to use as h?
        let mut u_element = u_element.clone();
        approximate_gradient_fd(f, &mut u_element, 1e-6)
    };

    let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
    let quadrature_data = local::evaluate_density_at_quadrature_points(&element, &points, local::density);
    let mut output = DVector::repeat(2 * element.num_nodes(), 3.0);
    let mut gradient_buffer =
        DMatrix::repeat(3, element.num_nodes(), 3.0).reshape_generic(U3::name(), Dynamic::new(element.num_nodes()));
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

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // A Tet10 element can reproduce any quadratic solution field exactly
    let element = Tet10Element::from(&tet4_element);
    let u_element = local::u_element_from_vertices_and_u_exact(element.vertices(), u_vector_quadratic);

    // Let f(u_element) = energy(grad u). Then we can compute an approximate derivative with finite
    // differences and use this to compare with our output from the assembly, which should
    // be exactly the same.
    let finite_diff_result = {
        let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
        let quadrature_data = local::evaluate_density_at_quadrature_points(&element, &points, local::density);

        // Set up a function f = f(u) that corresponds to the element vector given state u
        let f = |u: DVectorSlice<f64>, output: DVectorSliceMut<f64>| {
            let mut gradient_buffer = DMatrix::repeat(3, element.num_nodes(), 3.0)
                .reshape_generic(U3::name(), Dynamic::new(element.num_nodes()));
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
    let quadrature_data = local::evaluate_density_at_quadrature_points(&element, &points, local::density);
    let mut output = DMatrix::repeat(2 * element.num_nodes(), 2 * element.num_nodes(), 3.0);
    let mut gradient_buffer =
        DMatrix::repeat(3, element.num_nodes(), 3.0).reshape_generic(U3::name(), Dynamic::new(element.num_nodes()));
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
fn symmetric_elliptic_element_matrix_is_jacobian_of_vector_tet10() {
    // This test is the same as the non-symmetric test, except we use an operator
    // that is explicitly symmetric

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // A Tet10 element can reproduce any quadratic solution field exactly
    let element = Tet10Element::from(&tet4_element);
    let u_element = local::u_element_from_vertices_and_u_exact(element.vertices(), u_vector_quadratic);

    // Let f(u_element) = energy(grad u). Then we can compute an approximate derivative with finite
    // differences and use this to compare with our output from the assembly, which should
    // be exactly the same.
    let finite_diff_result = {
        let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
        let quadrature_data = local::evaluate_density_at_quadrature_points(&element, &points, local::density);

        // Set up a function f = f(u) that corresponds to the element vector given state u
        let f = |u: DVectorSlice<f64>, output: DVectorSliceMut<f64>| {
            let mut gradient_buffer = DMatrix::repeat(3, element.num_nodes(), 3.0)
                .reshape_generic(U3::name(), Dynamic::new(element.num_nodes()));
            assemble_element_elliptic_vector(
                output,
                &element,
                &MockVectorSymmetricEllipticEnergy,
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
    let quadrature_data = local::evaluate_density_at_quadrature_points(&element, &points, local::density);
    let mut output = DMatrix::repeat(2 * element.num_nodes(), 2 * element.num_nodes(), 3.0);
    let mut gradient_buffer =
        DMatrix::repeat(3, element.num_nodes(), 3.0).reshape_generic(U3::name(), Dynamic::new(element.num_nodes()));
    assemble_element_elliptic_matrix(
        MatrixSliceMut::from(&mut output),
        &element,
        &MockVectorSymmetricEllipticEnergy,
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
fn elliptic_element_assembler_matches_individual_element_assembly() {
    // Create a mesh with a small number of elements
    let mesh: QuadMesh2d<f64> = create_unit_square_uniform_quad_mesh_2d(3);
    let mesh = mesh.keep_cells(&[0, 1, 2, 3]);

    // Then give each element its own quadrature rule
    let (weights, points): (Vec<_>, Vec<_>) = vec![
        quadrature::tensor::quadrilateral_gauss::<f64>(2),
        quadrature::tensor::quadrilateral_gauss::<f64>(3),
        quadrature::tensor::quadrilateral_gauss::<f64>(4),
        quadrature::tensor::quadrilateral_gauss::<f64>(5),
    ]
    .into_iter()
    .unzip();

    let params: Vec<Vec<_>> = points
        .iter()
        .zip(mesh.connectivity())
        .map(|(points_for_element, conn)| {
            let element = conn.element(mesh.vertices()).unwrap();
            let density_per_point = points_for_element
                .iter()
                .map(|xi| element.map_reference_coords(xi))
                .map(|x| density(&x))
                .collect();
            density_per_point
        })
        .collect();

    let qtable = GeneralQuadratureTable::from_points_weights_and_data(
        NestedVec::from(&points),
        NestedVec::from(&weights),
        NestedVec::from(&params),
    );

    // And set up a simple mock operator
    struct MockEllipticOperator;

    impl Operator<f64, U2> for MockEllipticOperator {
        type SolutionDim = U2;
        type Parameters = f64;
    }

    impl EllipticEnergy<f64, U2> for MockEllipticOperator {
        fn compute_energy(&self, gradient: &OMatrix<f64, U2, Self::SolutionDim>, &density: &Self::Parameters) -> f64 {
            density * gradient.norm_squared()
        }
    }

    impl EllipticOperator<f64, U2> for MockEllipticOperator {
        fn compute_elliptic_operator(&self, gradient: &Matrix2<f64>, &density: &Self::Parameters) -> Matrix2<f64> {
            density * gradient
        }
    }

    impl EllipticContraction<f64, U2> for MockEllipticOperator {
        fn contract(
            &self,
            gradient: &Matrix2<f64>,
            a: &Vector2<f64>,
            b: &Vector2<f64>,
            &density: &Self::Parameters,
        ) -> Matrix2<f64> {
            // This is *not* the correct contraction with respect to the elliptic operator,
            // but it doesn't matter for our subsequent testing
            density * (a.dot(&b)) * gradient.transpose() * gradient
        }
    }

    let u_global: Vec<_> = (0..2 * mesh.vertices().len()).map(|i| i as f64).collect();
    let u_global = DVector::from_vec(u_global);

    let assembler = ElementEllipticAssemblerBuilder::new()
        .with_finite_element_space(&mesh)
        .with_quadrature_table(&qtable)
        .with_operator(&MockEllipticOperator)
        .with_u(&u_global)
        .build();

    // Now check that the result returned by the assembler matches that returned
    // by using a low-level routine
    let mut element_vector = DVector::repeat(8, 3.0);
    let mut element_matrix = DMatrix::repeat(8, 8, 3.0);
    let mut element_vector_expected = DVector::repeat(8, 4.0);
    let mut element_matrix_expected = DMatrix::repeat(8, 8, 4.0);
    let mut u_element = DVector::repeat(8, 5.0);
    let mut basis_gradients_buffer = DMatrix::repeat(2, 4, 6.0).reshape_generic(U2::name(), Dynamic::new(4));
    for (i, conn) in mesh.connectivity().iter().enumerate() {
        assembler
            .assemble_element_vector_into(i, DVectorSliceMut::from(&mut element_vector))
            .unwrap();

        assembler
            .assemble_element_matrix_into(i, DMatrixSliceMut::from(&mut element_matrix))
            .unwrap();

        let element_scalar = assembler.assemble_element_scalar(i).unwrap();

        // Compute expected element scalar, vector and matrix for this element
        let element_scalar_expected;
        {
            let element = conn.element(mesh.vertices()).unwrap();
            let weights = &weights[i];
            let points = &points[i];
            let data = &params[i];
            gather_global_to_local(&u_global, &mut u_element, conn.vertex_indices(), 2);
            element_scalar_expected = compute_element_elliptic_energy(
                &element,
                &MockEllipticOperator,
                DVectorSlice::from(&u_element),
                weights,
                points,
                &data,
                MatrixSliceMut::from(&mut basis_gradients_buffer),
            )
            .unwrap();

            assemble_element_elliptic_vector(
                DVectorSliceMut::from(&mut element_vector_expected),
                &element,
                &MockEllipticOperator,
                DVectorSlice::from(&u_element),
                weights,
                points,
                &data,
                MatrixSliceMut::from(&mut basis_gradients_buffer),
            )
            .unwrap();
            assemble_element_elliptic_matrix(
                DMatrixSliceMut::from(&mut element_matrix_expected),
                &element,
                &MockEllipticOperator,
                DVectorSlice::from(&u_element),
                weights,
                points,
                &data,
                MatrixSliceMut::from(&mut basis_gradients_buffer),
            )
            .unwrap();
        }

        assert_scalar_eq!(element_scalar, element_scalar_expected);
        assert_matrix_eq!(element_vector, element_vector_expected);
        assert_matrix_eq!(element_matrix, element_matrix_expected);
    }
}

fn compute_energy_integral<Element, Energy>(
    element: &Element,
    energy: &Energy,
    u_element: DVectorSlice<f64>,
    quadrature: &QuadraturePair<f64, Element::GeometryDim>,
    quadrature_params: &[Energy::Parameters],
) -> f64
where
    Element: VolumetricFiniteElement<f64>,
    Energy: EllipticEnergy<f64, Element::GeometryDim>,
    DefaultAllocator: BiDimAllocator<f64, Element::GeometryDim, Energy::SolutionDim>,
{
    let (weights, points) = quadrature;

    let d = Element::GeometryDim::dim();
    let mut gradient_buffer = OMatrix::from_vec_generic(
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
        quadrature_params,
        MatrixSliceMut::from(&mut gradient_buffer),
    )
    .unwrap()
}

fn compute_expected_energy_integral<Element, Energy, UGrad>(
    element: &Element,
    energy: &Energy,
    u_grad: UGrad,
    reference_rule: &QuadraturePair<f64, Element::GeometryDim>,
) -> f64
where
    Element: VolumetricFiniteElement<f64>,
    Energy: EllipticEnergy<f64, Element::GeometryDim, Parameters = f64>,
    UGrad: Fn(&OPoint<f64, Element::GeometryDim>) -> OMatrix<f64, Element::GeometryDim, Energy::SolutionDim>,
    DefaultAllocator: BiDimAllocator<f64, Element::GeometryDim, Energy::SolutionDim>,
{
    let quadrature_rule = local::construct_quadrature_rule_for_element(element, reference_rule);
    // Assuming f is a polynomial function (i.e. the energy is a polynomial in terms of the
    // components of u_grad and u_grad is polynomial), then we can hopefully compute this integral
    // exactly provided that the reference rule is sufficiently accurate
    let f = |x: &OPoint<f64, Element::GeometryDim>| energy.compute_energy(&u_grad(x), &local::density(x));
    let integral_expected = quadrature_rule.integrate(f);
    integral_expected
}

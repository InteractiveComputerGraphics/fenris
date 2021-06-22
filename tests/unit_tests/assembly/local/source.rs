use crate::unit_tests::assembly::local;
use fenris::assembly::local::{assemble_element_source_vector, SourceFunction};
use fenris::assembly::operators::Operator;
use fenris::element::{ReferenceFiniteElement, Tet10Element, Tet4Element};
use fenris::nalgebra::base::coordinates::XYZ;
use fenris::nalgebra::{DVector, DVectorSliceMut, Point, Point3, Vector2, U2, U3};
use fenris::quadrature;
use fenris::quadrature::Quadrature;
use matrixcompare::assert_scalar_eq;
use std::ops::Deref;

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
        // We give each point in space a "density" in order to test correct parameter evaluation
        type Parameters = f64;
    }

    impl SourceFunction<f64, U3> for MockSourceFunction {
        fn evaluate(&self, coords: &Point<f64, U3>, density: &Self::Parameters) -> Vector2<f64> {
            // The actual function is rho(x) * f(x), where rho(x) is a scalar implicitly
            // determined by the parameters
            *density * f(coords)
        }
    }

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // A Tet10 element can reproduce any quadratic solution field exactly
    let element = Tet10Element::from(&tet4_element);
    let u_element = local::u_element_from_vertices_and_u_exact(element.vertices(), u);

    let (weights, points) = quadrature::total_order::tetrahedron(8).unwrap();
    let quadrature_data =
        local::evaluate_density_at_quadrature_points(&element, &points, local::density);
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
        // u is a quadratic function and f is together with the density function of order 4,
        // so the product is of order 6
        let reference_rule = quadrature::total_order::tetrahedron(6).unwrap();
        let quadrature_rule =
            local::construct_quadrature_rule_for_element(&element, &reference_rule);
        quadrature_rule.integrate(|x| local::density(x) * f(x).dot(&u(x)))
    };

    let computed_inner_product = u_element.dot(&f_element);

    assert_scalar_eq!(
        computed_inner_product,
        expected_inner_product,
        comp = abs,
        tol = 1e-12
    );
}

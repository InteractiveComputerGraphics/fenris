use fenris::quadrature::subdivide::subdivide_univariate;
use fenris::quadrature::univariate::gauss;
use fenris::quadrature::Quadrature;
use matrixcompare::assert_scalar_eq;

#[test]
fn subdivided_gauss_rules_exactly_integrate_monomials() {
    // This test is almost exactly the same as a test in fenris-quadrature, but modified
    // to test *subdivided* Gauss quadrature rules (as opposed to standard Gauss quadrature)
    // The gist of this test is that it should be able to exactly integrate the same monomials as
    // the original quadrature rule
    for n in 1..=20 {
        let expected_polynomial_degree = 2 * n - 1;
        let reference_rule = gauss(n);

        for num_subdivs in 1..6 {
            let rule = subdivide_univariate(&reference_rule, num_subdivs);
            assert_eq!(rule.weights().len(), num_subdivs * reference_rule.weights().len());
            assert_eq!(rule.points().len(), num_subdivs * reference_rule.points().len());

            // Integrate all monomials of degree <= expected polynomial degree that can be
            // exactly integrated
            for alpha in 0..=expected_polynomial_degree as i32 {
                let monomial = |x: f64| x.powi(alpha);
                let monomial_integral = (1.0 - (-1.0f64).powi(alpha + 1)) / (alpha as f64 + 1.0);
                let estimated_integral = rule.integrate(|x| monomial(x[0]));

                assert_scalar_eq!(estimated_integral, monomial_integral, comp = abs, tol = 1e-14);
            }
        }
    }
}

#[test]
fn subdivided_gauss_rules_have_periodic_weights() {
    // We check that the weights are repeating in a periodic pattern.
    // This ensures that the weights are not unevenly distributed. For example, an erronous implementation
    // might consist of the initial quadrature rule and additional zero-weighted points. This would
    // still pass the polynomial accuracy test

    for n in 1..=20 {
        let reference_rule = gauss::<f64>(n);
        let reference_size = reference_rule.weights().len();

        for num_subdivs in 0..6 {
            let rule = subdivide_univariate(&reference_rule, num_subdivs);

            for i in 0..reference_size {
                for s in 0..num_subdivs {
                    assert_scalar_eq!(rule.weights()[i], rule.weights()[s * reference_size + i], comp = float);
                }
            }
        }
    }
}

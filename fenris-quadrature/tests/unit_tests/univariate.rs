use fenris_quadrature::integrate;
use fenris_quadrature::univariate::gauss;

use matrixcompare::assert_scalar_eq;

#[test]
fn gauss_rules_satisfy_expected_accuracy() {
    for n in 1..=200 {
        let expected_polynomial_degree = 2 * n - 1;
        let rule = gauss(n);

        // Also test that weights are positive
        assert!(rule.0.iter().all(|&w| w > 0.0));

        // Integrate all monomials of degree <= expected polynomial degree that can be
        // exactly integrated
        for alpha in 0..=expected_polynomial_degree as i32 {
            let monomial = |x: f64| x.powi(alpha);
            let monomial_integral = (1.0 - (-1.0f64).powi(alpha + 1)) / (alpha as f64 + 1.0);
            let estimated_integral = integrate(&rule, |x| monomial(x[0]));

            assert_scalar_eq!(
                estimated_integral,
                monomial_integral,
                comp = abs,
                tol = 1e-14
            );
        }
    }
}

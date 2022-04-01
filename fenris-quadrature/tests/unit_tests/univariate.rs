use fenris_quadrature::integrate;
use fenris_quadrature::univariate::{gauss, try_gauss_lobatto};

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

            assert_scalar_eq!(estimated_integral, monomial_integral, comp = abs, tol = 1e-14);
        }
    }
}

#[test]
fn gauss_lobatto_rules_satisfy_expected_accuracy() {
    assert!(try_gauss_lobatto(0).is_none());
    assert!(try_gauss_lobatto(1).is_none());

    let available_n = (2..=32).chain([64, 128, 256, 512]);

    for n in available_n {
        let expected_polynomial_degree = 2 * n - 3;
        let rule = try_gauss_lobatto(n).unwrap();

        // Check that rule contains endpoints, like Gauss-Lobatto should
        assert_eq!(rule.1.first().unwrap(), &[-1.0]);
        assert_eq!(rule.1.last().unwrap(), &[1.0]);

        // Also test that weights are positive
        assert!(rule.0.iter().all(|&w| w > 0.0));

        // Integrate all monomials of degree <= expected polynomial degree that can be
        // exactly integrated
        for alpha in 0..=expected_polynomial_degree as i32 {
            let monomial = |x: f64| x.powi(alpha);
            let monomial_integral = (1.0 - (-1.0f64).powi(alpha + 1)) / (alpha as f64 + 1.0);
            let estimated_integral = integrate(&rule, |x| monomial(x[0]));

            assert_scalar_eq!(estimated_integral, monomial_integral, comp = abs, tol = 1e-14);
        }
    }
}

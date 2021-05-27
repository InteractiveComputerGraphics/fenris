use fenris_quadrature::tensor::{quadrilateral_gauss, hexahedron_gauss};
use fenris_quadrature::integrate;
use matrixcompare::assert_scalar_eq;

#[test]
fn quadrilateral_gauss_rules_satisfy_expected_accuracy() {
    // Number of points in each dimension of rule
    for n in 1..=20 {
        // Expected polynomial degree that the rule can exactly integrate *along each dimension*
        let expected_polynomial_degree = 2 * n - 1;
        let rule = quadrilateral_gauss(n);

        // Also test that weights are positive
        assert!(rule.0.iter().all(|&w| w > 0.0));

        // Integrate all monomials of per-dimension degree <= expected polynomial degree that
        // can be exactly integrated
        for alpha in 0..=expected_polynomial_degree as i32 {
            for beta in 0 ..=expected_polynomial_degree as i32 {
                let monomial = |x: f64, y: f64| x.powi(alpha) * y.powi(beta);
                let monomial_integral_1d = |alpha| (1.0 - (-1.0f64).powi(alpha + 1)) / (alpha as f64 + 1.0);
                let monomial_integral_2d = monomial_integral_1d(alpha) * monomial_integral_1d(beta);
                let estimated_integral = integrate(&rule, |&[x, y]| monomial(x, y));

                assert_scalar_eq!(
                    estimated_integral,
                    monomial_integral_2d,
                    comp = abs,
                    tol = 1e-14
                );
            }
        }
    }
}

#[test]
fn hexahedral_gauss_rules_satisfy_expected_accuracy() {
    // Number of points in each dimension of rule
    for n in 1..=10 {
        // Expected polynomial degree that the rule can exactly integrate *along each dimension*
        let expected_polynomial_degree = 2 * n - 1;
        let rule = hexahedron_gauss(n);

        // Also test that weights are positive
        assert!(rule.0.iter().all(|&w| w > 0.0));

        // Integrate all monomials of per-dimension degree <= expected polynomial degree that
        // can be exactly integrated
        for alpha in 0..=expected_polynomial_degree as i32 {
            for beta in 0 ..=expected_polynomial_degree as i32 {
                for gamma in 0 ..=expected_polynomial_degree as i32 {
                    let monomial = |x: f64, y: f64, z: f64| x.powi(alpha) * y.powi(beta) * z.powi(gamma);
                    let monomial_integral_1d = |alpha| (1.0 - (-1.0f64).powi(alpha + 1)) / (alpha as f64 + 1.0);
                    let monomial_integral_2d = monomial_integral_1d(alpha) * monomial_integral_1d(beta) * monomial_integral_1d(gamma);
                    let estimated_integral = integrate(&rule, |&[x, y, z]| monomial(x, y, z));

                    assert_scalar_eq!(
                    estimated_integral,
                    monomial_integral_2d,
                    comp = abs,
                    tol = 1e-13
                );
                }
            }
        }
    }
}
use fenris::nalgebra::Point2;
use fenris::quadrature::subdivide::subdivide_univariate;
use fenris::quadrature::univariate::gauss;
use fenris::quadrature::{subdivide::subdivide_triangle, total_order, Quadrature};
use itertools::izip;
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

#[test]
fn subdivide_triangle_error() {
    // Here we estimate error and check manually that they make sense, then compare
    // errors against expected errors (as determined by previous run)
    let cos = |x: f64| x.cos();
    let sin = |x: f64| x.sin();
    let exp = |x: f64| x.exp();
    let f = |x, y| cos(x) * sin(y * x) + exp(x + y);
    let f = |p: &Point2<f64>| f(p.x, p.y);

    let base_quadrature = total_order::triangle::<f64>(5).unwrap();
    let integral_reference = subdivide_triangle(&base_quadrature, 20).integrate(f);

    let errors: Vec<f64> = (1..=10)
        .map(|subdivs| subdivide_triangle(&base_quadrature, subdivs))
        .map(|quadrature| quadrature.integrate(f))
        .map(|integral| (integral - integral_reference).abs())
        .collect();
    let expected_errors = vec![
        0.00032406989918110085,
        2.2805424527705398e-5,
        2.069177391428312e-6,
        3.679345033091863e-7,
        9.622548069465608e-8,
        3.2158293583606223e-8,
        1.2724713949197053e-8,
        5.693470583878479e-9,
        2.7950108894003733e-9,
        1.4738197329222658e-9,
    ];

    assert_eq!(errors.len(), expected_errors.len());
    for (error, expected) in izip!(errors, expected_errors) {
        assert!((error - expected).abs() / expected.abs() <= 1e-5);
    }
}

#[test]
fn subdivide_triangle_has_same_polynomial_strength_as_base() {
    let create_monomial = |i: usize, j: usize| move |p: &Point2<f64>| p.x.powi(i as i32) * p.y.powi(j as i32);
    for subdivs in 1..=10 {
        for strength in 1..=10 {
            let base_quadrature = total_order::triangle(strength).unwrap();
            let subdivided_quadrature = subdivide_triangle(&base_quadrature, subdivs);
            let num_subtriangles = subdivs * subdivs;
            let expected_num_points = num_subtriangles * base_quadrature.points().len();
            assert_eq!(subdivided_quadrature.points().len(), expected_num_points);
            assert_eq!(
                subdivided_quadrature.points().len(),
                subdivided_quadrature.weights().len()
            );
            for i in 0..=strength {
                for j in 0..=strength {
                    if i + j <= strength {
                        let monomial = create_monomial(i, j);
                        let base_integral = base_quadrature.integrate(monomial);
                        let subdivided_integral = subdivided_quadrature.integrate(monomial);
                        assert_scalar_eq!(subdivided_integral, base_integral, comp = abs, tol = 1e-12);
                    }
                }
            }
        }
    }
}

//! Quadrature rules for the one-dimensional domain `[-1, 1]`.

use crate::Rule;
use std::f64::consts::PI;

/// Recurrence relation for Legendre polynomials.
///
/// Note: we use a formula for which derivatives are *not* defined at |x| == 1, so it is only
/// suitable for evaluation in the open interval (-1, 1).
#[derive(Debug, Default)]
struct LegendreRecurrence {
    n: usize,
    x: f64,
    // The current value, i.e. p_n(x)
    p1: f64,
    // The previous value in the recurrence, i.e. p_{n - 1}(x)
    p2: f64,
}

impl LegendreRecurrence {
    pub fn evaluate(n: usize, x: f64) -> Self {
        // Use recurrence relation
        //  m P_m(x) = (2m - 1) * x P_{m - 1}(x) - (m - 1) P_{m - 2}(x)
        let mut p1 = 1.0;
        let mut p2 = 0.0;
        let mut p3;
        for m in 1..=n {
            let m = m as f64;
            p3 = p2;
            p2 = p1;
            p1 = ((2.0 * m - 1.0) * x * p2 - (m - 1.0) * p3) / m;
        }

        Self { n, x, p1, p2 }
    }

    fn value(&self) -> f64 {
        self.p1
    }

    fn derivative(&self) -> f64 {
        let Self { n, x, p1, p2 } = &self;
        let n = *n as f64;
        // Use the standard recurrence relation
        // dp_n/dx (x) = n * (x * p_n(x) - p_{n - 1}(x)) / (x^2 - 1)
        n * (x * p1 - p2) / (x * x - 1.0)
    }

    fn value_and_derivative(&self) -> (f64, f64) {
        (self.value(), self.derivative())
    }
}

/// Gauss quadrature for the reference interval [-1, 1].
///
/// Returns the [Gauss quadrature rule] with the given number of points. Given `n` points,
/// the rule integrates polynomials of order up to `2 n - 1` exactly.
///
/// # Panics
///
/// Panics if zero points are requested.
///
/// [Gauss quadrature rule]: https://en.wikipedia.org/wiki/Gaussian_quadrature
pub fn gauss(num_points: usize) -> Rule<1> {
    let n = num_points;
    assert!(n > 0, "number of points must be positive");

    // Loosely based on the procedure used in
    // Numerical Recipes, The art of Scientific Computing, Third Edition (2007)
    let num_roots = n;
    let m = (num_roots + 1) / 2;

    let mut points = Vec::with_capacity(num_roots);
    let mut weights = Vec::with_capacity(num_roots);

    // Only find the first m roots. The remaining roots can be found by symmetry
    for i in 0..m {
        // Compute a fairly accurate initial guess
        let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let (mut p, mut dp) = LegendreRecurrence::evaluate(n, x).value_and_derivative();

        // Newton's method
        // TODO: Can we end up in a situation where we loop indefinitely? Might want to
        // add some maximum expected iteration and return error?
        // Note: We don't seem to have any such problems in our tests, at least
        'newton: loop {
            let dx = -p / dp;
            x += dx;
            let (p_new, dp_new) = LegendreRecurrence::evaluate(n, x).value_and_derivative();
            p = p_new;
            dp = dp_new;
            if dx.abs() <= 1e-15 {
                break 'newton;
            }
        }

        // Once a root is known, its corresponding weight is given explicitly by a standard
        // formula
        let w = 2.0 / ((1.0 - x * x) * dp * dp);

        points.push([x]);
        weights.push(w);
    }

    // Recover the remaining points and weights by symmetry
    for i in m..n {
        let mirror_idx = n - i - 1;
        points.push([-points[mirror_idx][0]]);
        weights.push(weights[mirror_idx]);
    }

    assert_eq!(points.len(), weights.len());
    assert_eq!(points.len(), n, "Internal error: incorrect number of points produced");

    (weights, points)
}

#[cfg(test)]
mod tests {
    use crate::univariate::LegendreRecurrence;
    use matrixcompare::assert_scalar_eq;

    #[test]
    fn legendre_recurrence() {
        let num_samples = 2;

        // Actual Legendre polynomials, p[n]
        let p: Vec<fn(f64) -> f64> = vec![
            |_| 1.0,
            |x| x,
            |x| 0.5 * (3.0 * x.powi(2) - 1.0),
            |x| 0.5 * (5.0 * x.powi(3) - 3.0 * x),
            |x| (1.0 / 8.0) * (35.0 * x.powi(4) - 30.0 * x.powi(2) + 3.0),
        ];
        let dp: Vec<fn(f64) -> f64> = vec![|_| 0.0, |_| 1.0, |x| 3.0 * x, |x| 0.5 * (15.0 * x.powi(2) - 3.0), |x| {
            (1.0 / 8.0) * (35.0 * 4.0 * x.powi(3) - 60.0 * x)
        }];

        for n in 0..p.len() {
            for i in 1..num_samples {
                let x_i = -1.0 + (i as f64) * 2.0 / (num_samples as f64);
                let recurrence = LegendreRecurrence::evaluate(n, x_i);
                assert_scalar_eq!(recurrence.value(), p[n](x_i), comp = abs, tol = 1e-14);
                assert_scalar_eq!(recurrence.derivative(), dp[n](x_i), comp = abs, tol = 1e-14);
            }
        }
    }
}

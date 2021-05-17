use fenris_quadrature::{Error, Rule2d, Rule3d};
use fenris_quadrature::polyquad::{triangle, hexahedron, tetrahedron, pyramid, quadrilateral, prism};

use matrixcompare::assert_scalar_eq;

macro_rules! assert_quadrature_size {
    ($quadrature_fn:ident, strength = $strength:expr, size = $size:expr) => {
        {
            let (points, weights) = $quadrature_fn($strength)
                .expect("Expected valid quadrature rule");
            assert_eq!(points.len(), weights.len());
            assert_eq!(points.len(), $size);
        }
    }
}

macro_rules! assert_no_quadrature_for_strength {
    ($quadrature_fn:ident, strength = $strength:expr) => {
        assert_eq!($quadrature_fn($strength), Err(Error::NoRuleAvailable));
    }
}

#[test]
fn triangle_rules_have_expected_numbers_of_points() {
    assert_quadrature_size!(triangle, strength = 0, size = 1);
    assert_quadrature_size!(triangle, strength = 1, size = 1);
    assert_quadrature_size!(triangle, strength = 2, size = 3);
    assert_quadrature_size!(triangle, strength = 3, size = 6);
    assert_quadrature_size!(triangle, strength = 4, size = 6);
    assert_quadrature_size!(triangle, strength = 5, size = 7);
    assert_quadrature_size!(triangle, strength = 6, size = 12);
    assert_quadrature_size!(triangle, strength = 7, size = 15);
    assert_quadrature_size!(triangle, strength = 8, size = 16);
    assert_quadrature_size!(triangle, strength = 9, size = 19);
    assert_quadrature_size!(triangle, strength = 10, size = 25);
    assert_quadrature_size!(triangle, strength = 11, size = 28);
    assert_quadrature_size!(triangle, strength = 12, size = 33);
    assert_quadrature_size!(triangle, strength = 13, size = 37);
    assert_quadrature_size!(triangle, strength = 14, size = 42);
    assert_quadrature_size!(triangle, strength = 15, size = 49);
    assert_quadrature_size!(triangle, strength = 16, size = 55);
    assert_quadrature_size!(triangle, strength = 17, size = 60);
    assert_quadrature_size!(triangle, strength = 18, size = 67);
    assert_quadrature_size!(triangle, strength = 19, size = 73);
    assert_quadrature_size!(triangle, strength = 20, size = 79);

    assert_no_quadrature_for_strength!(triangle, strength = 21);
    assert_no_quadrature_for_strength!(triangle, strength = 22);
    assert_no_quadrature_for_strength!(triangle, strength = 23);
    assert_no_quadrature_for_strength!(triangle, strength = 24);
    assert_no_quadrature_for_strength!(triangle, strength = 25);
    assert_no_quadrature_for_strength!(triangle, strength = 26);
}

#[test]
fn quadrilateral_rules_have_expected_numbers_of_points() {
    assert_quadrature_size!(quadrilateral, strength = 0, size = 1);
    assert_quadrature_size!(quadrilateral, strength = 1, size = 1);
    assert_quadrature_size!(quadrilateral, strength = 2, size = 4);
    assert_quadrature_size!(quadrilateral, strength = 3, size = 4);
    assert_quadrature_size!(quadrilateral, strength = 4, size = 8);
    assert_quadrature_size!(quadrilateral, strength = 5, size = 8);
    assert_quadrature_size!(quadrilateral, strength = 6, size = 12);
    assert_quadrature_size!(quadrilateral, strength = 7, size = 12);
    assert_quadrature_size!(quadrilateral, strength = 8, size = 20);
    assert_quadrature_size!(quadrilateral, strength = 9, size = 20);
    assert_quadrature_size!(quadrilateral, strength = 10, size = 28);
    assert_quadrature_size!(quadrilateral, strength = 11, size = 28);
    assert_quadrature_size!(quadrilateral, strength = 12, size = 37);
    assert_quadrature_size!(quadrilateral, strength = 13, size = 37);
    assert_quadrature_size!(quadrilateral, strength = 14, size = 48);
    assert_quadrature_size!(quadrilateral, strength = 15, size = 48);
    assert_quadrature_size!(quadrilateral, strength = 16, size = 60);
    assert_quadrature_size!(quadrilateral, strength = 17, size = 60);
    assert_quadrature_size!(quadrilateral, strength = 18, size = 72);
    assert_quadrature_size!(quadrilateral, strength = 19, size = 72);
    assert_quadrature_size!(quadrilateral, strength = 20, size = 85);
    assert_quadrature_size!(quadrilateral, strength = 21, size = 85);

    assert_no_quadrature_for_strength!(quadrilateral, strength = 22);
    assert_no_quadrature_for_strength!(quadrilateral, strength = 23);
    assert_no_quadrature_for_strength!(quadrilateral, strength = 24);
    assert_no_quadrature_for_strength!(quadrilateral, strength = 25);
    assert_no_quadrature_for_strength!(quadrilateral, strength = 26);
    assert_no_quadrature_for_strength!(quadrilateral, strength = 27);
    assert_no_quadrature_for_strength!(quadrilateral, strength = 28);
}

#[test]
fn tetrahedron_rules_have_expected_number_of_points() {
    assert_quadrature_size!(tetrahedron, strength = 0, size = 1);
    assert_quadrature_size!(tetrahedron, strength = 1, size = 1);
    assert_quadrature_size!(tetrahedron, strength = 2, size = 4);
    assert_quadrature_size!(tetrahedron, strength = 3, size = 8);
    assert_quadrature_size!(tetrahedron, strength = 4, size = 14);
    assert_quadrature_size!(tetrahedron, strength = 5, size = 14);
    assert_quadrature_size!(tetrahedron, strength = 6, size = 24);
    assert_quadrature_size!(tetrahedron, strength = 7, size = 35);
    assert_quadrature_size!(tetrahedron, strength = 8, size = 46);
    assert_quadrature_size!(tetrahedron, strength = 9, size = 59);
    assert_quadrature_size!(tetrahedron, strength = 10, size = 81);

    assert_no_quadrature_for_strength!(tetrahedron, strength = 11);
    assert_no_quadrature_for_strength!(tetrahedron, strength = 12);
    assert_no_quadrature_for_strength!(tetrahedron, strength = 13);
    assert_no_quadrature_for_strength!(tetrahedron, strength = 14);

}

#[test]
fn hexahedron_rules_have_expected_numbers_of_points() {
    assert_quadrature_size!(hexahedron, strength = 0, size = 1);
    assert_quadrature_size!(hexahedron, strength = 1, size = 1);
    assert_quadrature_size!(hexahedron, strength = 2, size = 6);
    assert_quadrature_size!(hexahedron, strength = 3, size = 6);
    assert_quadrature_size!(hexahedron, strength = 4, size = 14);
    assert_quadrature_size!(hexahedron, strength = 5, size = 14);
    assert_quadrature_size!(hexahedron, strength = 6, size = 34);
    assert_quadrature_size!(hexahedron, strength = 7, size = 34);
    assert_quadrature_size!(hexahedron, strength = 8, size = 58);
    assert_quadrature_size!(hexahedron, strength = 9, size = 58);
    assert_quadrature_size!(hexahedron, strength = 10, size = 90);
    assert_quadrature_size!(hexahedron, strength = 11, size = 90);

    assert_no_quadrature_for_strength!(hexahedron, strength = 12);
    assert_no_quadrature_for_strength!(hexahedron, strength = 13);
    assert_no_quadrature_for_strength!(hexahedron, strength = 14);
}

#[test]
fn pyramid_rules_have_expected_numbers_of_points() {
    assert_quadrature_size!(pyramid, strength = 0, size = 1);
    assert_quadrature_size!(pyramid, strength = 1, size = 1);
    assert_quadrature_size!(pyramid, strength = 2, size = 5);
    assert_quadrature_size!(pyramid, strength = 3, size = 6);
    assert_quadrature_size!(pyramid, strength = 4, size = 10);
    assert_quadrature_size!(pyramid, strength = 5, size = 15);
    assert_quadrature_size!(pyramid, strength = 6, size = 24);
    assert_quadrature_size!(pyramid, strength = 7, size = 31);
    assert_quadrature_size!(pyramid, strength = 8, size = 47);
    assert_quadrature_size!(pyramid, strength = 9, size = 62);
    assert_quadrature_size!(pyramid, strength = 10, size = 83);

    assert_no_quadrature_for_strength!(pyramid, strength = 11);
    assert_no_quadrature_for_strength!(pyramid, strength = 12);
    assert_no_quadrature_for_strength!(pyramid, strength = 13);
}

#[test]
fn prism_rules_have_expected_numbers_of_points() {
    assert_quadrature_size!(prism, strength = 0, size = 1);
    assert_quadrature_size!(prism, strength = 1, size = 1);
    assert_quadrature_size!(prism, strength = 2, size = 5);
    assert_quadrature_size!(prism, strength = 3, size = 8);
    assert_quadrature_size!(prism, strength = 4, size = 11);
    assert_quadrature_size!(prism, strength = 5, size = 16);
    assert_quadrature_size!(prism, strength = 6, size = 28);
    assert_quadrature_size!(prism, strength = 7, size = 35);
    assert_quadrature_size!(prism, strength = 8, size = 46);
    assert_quadrature_size!(prism, strength = 9, size = 60);
    assert_quadrature_size!(prism, strength = 10, size = 85);

    assert_no_quadrature_for_strength!(prism, strength = 11);
    assert_no_quadrature_for_strength!(prism, strength = 12);
    assert_no_quadrature_for_strength!(prism, strength = 13);
}

fn is_even(i: usize) -> bool {
    i % 2 == 0
}

fn test_2d_rules_satisfy_prescribed_accuracy(max_strength: usize,
                                             rule_generator: impl Fn(usize) -> Rule2d,
                                             monomial_integral: impl Fn(usize, usize) -> f64) {
    for i in 0 ..= max_strength {
        for j in 0 ..= max_strength {
            if i + j <= max_strength {
                let monomial = |x: f64, y: f64| x.powi(i as i32) * y.powi(j as i32);
                let monomial_integral = monomial_integral(i, j);
                let strength = (i + j) as usize;
                let (weights, points) = rule_generator(strength);
                let estimated_integral: f64 = weights.into_iter()
                    .zip(&points)
                    .map(|(w, p)| w * monomial(p[0], p[1]))
                    .sum();

                // The magnitude of the integrals is either 0 or somewhere in the neighborhood of 1,
                // at least within a couple of orders of magnitude, which is why the below
                // absolute tolerance seems reasonable
                assert_scalar_eq!(estimated_integral, monomial_integral, comp = abs, tol = 1e-14);
            }
        }
    }
}

fn test_3d_rules_satisfy_prescribed_accuracy(max_strength: usize,
                                             rule_generator: impl Fn(usize) -> Rule3d,
                                             monomial_integral: impl Fn(usize, usize, usize) -> f64) {
    for i in 0 ..= max_strength {
        for j in 0 ..= max_strength {
            for k in 0 ..= max_strength {
                if i + j + k <= max_strength {
                    let monomial = |x: f64, y: f64, z:f64| x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32);
                    let monomial_integral = monomial_integral(i, j, k);
                    let strength = (i + j + k) as usize;
                    let (weights, points) = rule_generator(strength);
                    let estimated_integral: f64 = weights.into_iter()
                        .zip(&points)
                        .map(|(w, p)| w * monomial(p[0], p[1], p[2]))
                        .sum();

                    // The magnitude of the integrals is either 0 or somewhere in the neighborhood of 1,
                    // at least within a couple of orders of magnitude, which is why the below
                    // absolute tolerance seems reasonable
                    assert_scalar_eq!(estimated_integral, monomial_integral, comp = abs, tol = 1e-14);
                }
            }
        }
    }
}

#[test]
fn quadrilateral_rules_satisfy_prescribed_accuracy() {
    let max_strength = 21;
    let rule_generator = |strength| quadrilateral(strength).unwrap();

    let monomial_integral = |i, j| if is_even(i) && is_even(j) {
        let (i, j) = (i as f64, j as f64);
        4.0 / ((i + 1.0) * (j + 1.0))
    } else {
        0.0
    };

    test_2d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}

#[test]
fn triangle_rules_satisfy_prescribed_accuracy() {
    let max_strength = 20;
    let rule_generator = |strength| triangle(strength).unwrap();

    let monomial_integral = |i, j| {
        let (i, j) = (i as f64, j as f64);
        // TODO: Is there a simpler expression?
        (-1.0f64).powf(j + 1.0) / (j + 1.0)
        * ( (1.0 - (-1.0f64).powf(i + j))/(i + j + 2.0)
            - (1.0 - (-1.0f64).powf(i + 1.0))/(i + 1.0))
    };

    test_2d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}

#[test]
fn hexahedron_rules_satisfy_prescribed_accuracy() {
    let max_strength = 11;
    let rule_generator = |strength| hexahedron(strength).unwrap();

    let monomial_integral = |i, j, k| if is_even(i) && is_even(j) && is_even(k) {
        let (i, j, k) = (i as f64, j as f64, k as f64);
        8.0 / ((i + 1.0) * (j + 1.0) * (k + 1.0))
    } else {
        0.0
    };

    test_3d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}
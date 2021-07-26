use fenris_quadrature::polyquad::{hexahedron, prism, pyramid, quadrilateral, tetrahedron, triangle};
use fenris_quadrature::{Error, Rule};

use matrixcompare::assert_scalar_eq;
use nalgebra::{SVector, Vector3};
use std::convert::TryFrom;

macro_rules! assert_quadrature_size {
    ($quadrature_fn:ident, strength = $strength:expr, size = $size:expr) => {{
        let (points, weights) = $quadrature_fn($strength).expect("Expected valid quadrature rule");
        assert_eq!(points.len(), weights.len());
        assert_eq!(points.len(), $size);
    }};
}

macro_rules! assert_no_quadrature_for_strength {
    ($quadrature_fn:ident, strength = $strength:expr) => {
        assert_eq!($quadrature_fn($strength), Err(Error::NoRuleAvailable));
    };
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

fn test_2d_rules_satisfy_prescribed_accuracy(
    max_strength: usize,
    rule_generator: impl Fn(usize) -> Rule<2>,
    monomial_integral: impl Fn(usize, usize) -> f64,
) {
    for strength in 0..=max_strength {
        for i in 0..=strength {
            for j in 0..=strength {
                if i + j <= strength {
                    let monomial = |x: f64, y: f64| x.powi(i as i32) * y.powi(j as i32);
                    let monomial_integral = monomial_integral(i, j);
                    let (weights, points) = rule_generator(strength);
                    let estimated_integral: f64 = weights
                        .into_iter()
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
}

fn test_3d_rules_satisfy_prescribed_accuracy(
    max_strength: usize,
    rule_generator: impl Fn(usize) -> Rule<3>,
    monomial_integral: impl Fn(usize, usize, usize) -> f64,
) {
    for strength in 0..=max_strength {
        for i in 0..=strength {
            for j in 0..=strength {
                for k in 0..=strength {
                    if i + j + k <= strength {
                        let monomial = |x: f64, y: f64, z: f64| x.powi(i as i32) * y.powi(j as i32) * z.powi(k as i32);
                        let monomial_integral = monomial_integral(i, j, k);
                        let (weights, points) = rule_generator(strength);
                        let estimated_integral: f64 = weights
                            .into_iter()
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
}

#[test]
fn quadrilateral_rules_satisfy_prescribed_accuracy() {
    let max_strength = 21;
    let rule_generator = |strength| quadrilateral(strength).unwrap();

    let monomial_integral = |i, j| {
        if is_even(i) && is_even(j) {
            let (i, j) = (i as f64, j as f64);
            4.0 / ((i + 1.0) * (j + 1.0))
        } else {
            0.0
        }
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
            * ((1.0 - (-1.0f64).powf(i + j)) / (i + j + 2.0) - (1.0 - (-1.0f64).powf(i + 1.0)) / (i + 1.0))
    };

    test_2d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}

#[test]
fn hexahedron_rules_satisfy_prescribed_accuracy() {
    let max_strength = 11;
    let rule_generator = |strength| hexahedron(strength).unwrap();

    let monomial_integral = |i, j, k| {
        if is_even(i) && is_even(j) && is_even(k) {
            let (i, j, k) = (i as f64, j as f64, k as f64);
            8.0 / ((i + 1.0) * (j + 1.0) * (k + 1.0))
        } else {
            0.0
        }
    };

    test_3d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}

/// Integrate function on 3D planar polygon.
///
/// The function `f` to be integrated is a vector function, i.e. it maps `R^3 -> R^OUTDIM`.
/// The integration relies on quadrature rules with the provided `strength`.
fn integrate_polygon_3d<const OUTDIM: usize>(
    strength: usize,
    vertices: &[Vector3<f64>],
    f: impl Fn(&Vector3<f64>) -> SVector<f64, OUTDIM>,
) -> SVector<f64, OUTDIM> {
    assert!(vertices.len() >= 3, "Polygon must have at least 3 vertices");
    let mut result = SVector::zeros();

    let (weights, points) = triangle(strength).expect("Must have quadrature rule available for requested strength");

    // Decompose the polygon into a triangle fan and sum the integrals over each triangle
    let num_triangles = vertices.len() - 2;
    for i in 0..num_triangles {
        let x1 = vertices[0];
        let x2 = vertices[i + 1];
        let x3 = vertices[i + 2];

        // We consider a counter-clockwise oriented triangle defined by its vertices x1, x2 and x3.
        // The transformation from the reference triangle (with corners in (-1, -1), (1, -1) and
        // (-1, 1)) is then given by
        //  r(s, t) = (x2 - x1)/2 * s + (x3 - x1)/2 * t + (x2 + x3)/2
        let a = (x2 - x1) / 2.0;
        let b = (x3 - x1) / 2.0;
        let c = (x2 + x3) / 2.0;
        // Transformation from reference element to triangle defined by the three vertices.
        let r = |s: f64, t: f64| a * s + b * t + c;
        let h = |s, t| f(&r(s, t));
        let dr_ds = a;
        let dr_dt = b;
        let q = dr_ds.cross(&dr_dt).norm();
        // The surface integral of a scalar function f over a 3D triangle is given by
        //  integrate_S f([x, y, z]) dS = integrate_R f(r(xi, eta)) || dr/ds x dr/dt || ds dt
        // where R is the reference triangle.
        for (&w, p) in weights.iter().zip(&points) {
            result += w * h(p[0], p[1]) * q;
        }
    }

    result
}

pub struct Polyhedron {
    vertices: Vec<Vector3<f64>>,
    /// Each face is represented by a collection of indices into the list of vertices.
    /// Vertices of each face must be oriented counter-clockwise for consistent outwards normal
    /// direction.
    faces: Vec<Vec<usize>>,
}

impl Polyhedron {
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    pub fn extract_face_vertices(&self, face_index: usize) -> Vec<Vector3<f64>> {
        self.faces[face_index]
            .iter()
            .map(|&vertex_idx| self.vertices[vertex_idx])
            .collect()
    }
}

/// Integrate a monomial in the given polyhedron.
fn integrate_monomial_polyhedron(polyhedron: &Polyhedron, alpha: &[usize; 3]) -> f64 {
    // The non-Hex 3D shapes are much more difficult to verify. It is very hard to obtain a
    // symbolic expression for the monomial integral, so instead we choose a different strategy.
    // In short, we use the divergence theorem to convert the monomial integral to
    // a sum of integrals over each face. Since each face is a planar polygon,
    // we can compute the integral over each face by decomposing the polygon into triangles
    // and sum the integral over each triangle, which we can compute with our already
    // verified triangle quadrature rules.

    // Let f: R^3 -> R be a monomial function, i.e.
    //  f(x, y, z) = x^alpha * y^alpha * z^alpha.
    // Let g: R^3 -> R^3 be a (vector) function such that
    //  div(g) = f.
    // Then we can use the divergence theorem to rewrite the volume integral as a surface integral:
    //  integral f dV = integral div(g) dV = sum_i integral_{S_i} dot(g, n) dS
    // where S_i is the ith face of our shape. Since the faces are polygonal/planar, we can
    // further write
    //   ... = sum_i dot(n_i, integral_{S_i} g dS)
    // in which we can evaluate
    //   integral_{S_i} g dS
    // numerically.

    // We need one higher order strength because we'll be integrating a monomial of
    // one order higher than the initial monomial
    let [alpha, beta, gamma] = *alpha;
    let strength = alpha + beta + gamma + 1;
    let (alpha, beta, gamma) = (alpha as i32, beta as i32, gamma as i32);

    // g satisfies div(g) = f (note the 1/3 factor at the end)
    let g = |x: f64, y: f64, z: f64| {
        Vector3::new(
            x.powi(alpha + 1) * y.powi(beta) * z.powi(gamma) / (alpha + 1) as f64,
            x.powi(alpha) * y.powi(beta + 1) * z.powi(gamma) / (beta + 1) as f64,
            x.powi(alpha) * y.powi(beta) * z.powi(gamma + 1) / (gamma + 1) as f64,
        ) / 3.0
    };
    let g = |v: &Vector3<f64>| g(v[0], v[1], v[2]);

    let mut surface_integral = 0.0;

    // Compute volume integral by way of divergence theorem
    for i in 0..polyhedron.num_faces() {
        let face_vertices = polyhedron.extract_face_vertices(i);
        let [x1, x2, x3, ..] = <[_; 3]>::try_from(&face_vertices[0..3]).unwrap();
        let n = (x2 - x1).cross(&(x3 - x1)).normalize();
        surface_integral += n.dot(&integrate_polygon_3d(strength, &face_vertices, g));
    }

    surface_integral
}

#[test]
fn tetrahedron_rules_satisfy_prescribed_accuracy() {
    let max_strength = 10;
    let rule_generator = |strength| tetrahedron(strength).unwrap();

    let tetrahedron = Polyhedron {
        vertices: vec![
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, -1.0, 1.0),
        ],
        // (note: normals must point outwards)
        faces: vec![
            // Face on z = -1
            vec![0, 2, 1],
            // Face on y = -1
            vec![0, 1, 3],
            // Face on x = -1
            vec![0, 3, 2],
            // Face on x + y + z = -1
            vec![3, 1, 2],
        ],
    };

    let monomial_integral =
        |alpha: usize, beta: usize, gamma: usize| integrate_monomial_polyhedron(&tetrahedron, &[alpha, beta, gamma]);

    test_3d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}

#[test]
fn prism_rules_satisfy_prescribed_accuracy() {
    let max_strength = 10;
    let rule_generator = |strength| prism(strength).unwrap();

    let prism = Polyhedron {
        vertices: vec![
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
        ],
        // (note: normals must point outwards)
        faces: vec![
            vec![0, 2, 1],
            vec![0, 1, 4, 3],
            vec![1, 2, 5, 4],
            vec![0, 3, 5, 2],
            vec![4, 5, 3],
        ],
    };

    let monomial_integral =
        |alpha: usize, beta: usize, gamma: usize| integrate_monomial_polyhedron(&prism, &[alpha, beta, gamma]);

    test_3d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}

#[test]
fn pyramid_rules_satisfy_prescribed_accuracy() {
    let max_strength = 10;
    let rule_generator = |strength| pyramid(strength).unwrap();

    let pyramid = Polyhedron {
        vertices: vec![
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(1.0, 1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(0.0, 0.0, 1.0),
        ],
        // (note: normals must point outwards)
        faces: vec![
            vec![0, 3, 2, 1],
            vec![0, 1, 4],
            vec![1, 2, 4],
            vec![2, 3, 4],
            vec![0, 4, 3],
        ],
    };

    let monomial_integral =
        |alpha: usize, beta: usize, gamma: usize| integrate_monomial_polyhedron(&pyramid, &[alpha, beta, gamma]);

    test_3d_rules_satisfy_prescribed_accuracy(max_strength, rule_generator, monomial_integral);
}

use fenris::element::{Tet10Element, Tet4Element};
use fenris::nalgebra;
use fenris::nalgebra::{matrix, Matrix2, Matrix3, Point3};
use fenris_solid::materials::LameParameters;

mod material_elliptic_operator;
mod materials;

fn lame_parameters() -> LameParameters<f64> {
    LameParameters {
        mu: 384.0,
        lambda: 577.0,
    }
}

fn deformation_gradient_2d() -> Matrix2<f64> {
    // Note: this is deliberately chosen so that it has det(F) > 0
    matrix![2.0, 1.0;
            3.0, 4.0]
}

fn deformation_gradient_3d() -> Matrix3<f64> {
    // Note: this is deliberately chosen so that it has det(F) > 0
    matrix![2.0, 1.0, 3.0;
            4.0, 6.0, 5.0;
            2.0, 8.0, 9.0]
}

/// An arbitrary Tet10 element used in tests.
fn tet10_element() -> Tet10Element<f64> {
    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);
    Tet10Element::from(&tet4_element)
}

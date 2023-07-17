use fenris::element::{
    FixedNodesReferenceFiniteElement, Hex20Element, Hex27Element, Quad4d2Element, Quad9d2Element, Tri3d2Element,
    Tri6d2Element,
};

use fenris_traits::Real;

use nalgebra::{Point2, Point3};
use numeric_literals::replace_float_literals;
use proptest::prelude::*;
use util::assert_approx_matrix_eq;

mod hexahedron;
mod quadrilateral;
mod segment;
mod tetrahedron;
mod triangle;

fn point_in_tri_ref_domain() -> impl Strategy<Value = Point2<f64>> {
    // Generate points x, y in [-1, 1]^2 such that
    // x + y <= 0
    (-1.0..=1.0)
        .prop_flat_map(|x: f64| (Just(x), -1.0..=-x))
        .prop_map(|(x, y)| Point2::new(x, y))
}

fn point_in_quad_ref_domain() -> impl Strategy<Value = Point2<f64>> {
    // Generate points x, y, z in [-1, 1]^3
    let r = -1.0..=1.0;
    [r.clone(), r].prop_map(|[x, y]| Point2::new(x, y))
}

fn point_in_hex_ref_domain() -> impl Strategy<Value = Point3<f64>> {
    // Generate points x, y, z in [-1, 1]^3
    let r = -1.0..=1.0;
    [r.clone(), r.clone(), r].prop_map(|[x, y, z]| Point3::new(x, y, z))
}

fn point_in_tet_ref_domain() -> impl Strategy<Value = Point3<f64>> {
    // Generate points x, y, z in [-1, 1]^3 such that
    // x + y + z <= 0
    (-1.0..=1.0)
        .prop_flat_map(|x: f64| (Just(x), -1.0..=-x))
        .prop_flat_map(|(x, y)| {
            let z_range = -1.0..=(-(x + y));
            (Just(x), Just(y), z_range)
        })
        .prop_map(|(x, y, z)| Point3::new(x, y, z))
}

macro_rules! partition_of_unity_test {
    ($test_name:ident, $ref_domain_strategy:expr, $ref_element:expr) => {
        proptest! {
            #[test]
            fn $test_name(xi in $ref_domain_strategy) {
                let xi = xi;
                let element = $ref_element;
                let phi = element.evaluate_basis(&xi);
                let phi_sum: f64 = phi.sum();

                prop_assert!( (phi_sum - 1.0f64).abs() <= 1e-12);
            }
        }
    };
}

macro_rules! partition_of_unity_gradient_test {
    ($test_name:ident, $ref_domain_strategy:expr, $ref_element:expr) => {
        proptest! {
            #[test]
            fn $test_name(xi in $ref_domain_strategy) {
                // Since the sum of basis functions is 1, the sum of the gradients must be 0
                let xi = xi;
                let element = $ref_element;
                let grad = element.gradients(&xi);
                let grad_sum = grad.column_sum();

                let mut zero = grad_sum.clone();
                zero.fill(0.0);

                assert_approx_matrix_eq!(grad_sum, zero, abstol=1e-12);
            }
        }
    };
}

partition_of_unity_test!(
    tri3d2_partition_of_unity,
    point_in_tri_ref_domain(),
    Tri3d2Element::reference()
);
partition_of_unity_test!(
    tri6d2_partition_of_unity,
    point_in_tri_ref_domain(),
    Tri6d2Element::reference()
);
partition_of_unity_test!(
    quad4_partition_of_unity,
    point_in_quad_ref_domain(),
    Tri6d2Element::reference()
);
partition_of_unity_test!(
    quad9_partition_of_unity,
    point_in_quad_ref_domain(),
    Tri6d2Element::reference()
);

partition_of_unity_test!(
    hex27_partition_of_unity,
    point_in_hex_ref_domain(),
    Hex27Element::reference()
);

partition_of_unity_test!(
    hex20_partition_of_unity,
    point_in_hex_ref_domain(),
    Hex20Element::reference()
);

partition_of_unity_gradient_test!(
    tri3d2_partition_of_unity_gradient,
    point_in_tri_ref_domain(),
    Tri3d2Element::reference()
);
partition_of_unity_gradient_test!(
    tri6d2_partition_of_unity_gradient,
    point_in_tri_ref_domain(),
    Tri6d2Element::reference()
);
partition_of_unity_gradient_test!(
    quad4_partition_of_unity_gradient,
    point_in_quad_ref_domain(),
    Quad4d2Element::reference()
);
partition_of_unity_gradient_test!(
    quad9_partition_of_unity_gradient,
    point_in_quad_ref_domain(),
    Quad9d2Element::reference()
);

partition_of_unity_gradient_test!(
    hex27_partition_of_unity_gradient,
    point_in_hex_ref_domain(),
    Hex27Element::reference()
);

partition_of_unity_gradient_test!(
    hex20_partition_of_unity_gradient,
    point_in_hex_ref_domain(),
    Hex20Element::reference()
);

// TODO: This is copied from fenris code base. Don't want to make it part of public API,
// but it's unfortunate to duplicate it
#[replace_float_literals(T::from_f64(literal).unwrap())]
fn is_likely_in_tri_ref_interior<T: Real>(xi: &Point2<T>) -> bool {
    let eps = 4.0 * T::default_epsilon();
    xi.x >= -1.0 - eps && xi.y >= -1.0 - eps && xi.x + xi.y <= eps
}

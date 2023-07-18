use itertools::{Itertools, izip};
use fenris::element::{FixedNodesReferenceFiniteElement, Hex20Element, Hex27Element, Quad4d2Element, Quad9d2Element, Tet4Element, Tri3d2Element, Tri6d2Element};
use fenris_traits::Real;
use nalgebra::{Point2, Point3, Vector3};
use num::clamp;
use numeric_literals::replace_float_literals;
use proptest::array::uniform4;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use fenris::connectivity::Tet4Connectivity;
use fenris::mesh::Tet4Mesh;
use util::assert_approx_matrix_eq;
use crate::export_mesh_vtk;

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
    uniform4(0.0 ..= 1.0f64)
        .prop_map(|mut barycentric_coords| {
            let mut sum = 0.0;
            for lambda_i in &mut barycentric_coords {
                assert!(0.0 <= sum && sum <= 1.0);
                if *lambda_i + sum > 1.0 {
                    *lambda_i = clamp(1.0 - sum, 0.0, 1.0);
                }
                sum += *lambda_i;
            }
            let min_lambda_idx = barycentric_coords
                .iter()
                .copied()
                .position_min_by(f64::total_cmp)
                .unwrap();
            let lambda_min = &mut barycentric_coords[min_lambda_idx];
            *lambda_min = clamp(*lambda_min + 1.0 - sum, 0.0, 1.0);

            if cfg!(debug_assertions) {
                let sum = barycentric_coords.iter().sum();
                debug_assert!(1.0 - 1e-12 <= sum && sum <= 1.0 + 1e-12);
                debug_assert!(barycentric_coords.iter().all(|&lambda_i| 0.0 <= lambda_i && lambda_i <= 1.0));
            }
            barycentric_coords
        })
        .prop_shuffle()
        .prop_map(|barycentric_coords| {
            let tet_ref = Tet4Element::reference();
            let mut xi = Vector3::zeros();
            for (&lambda_i, &x_i) in izip!(&barycentric_coords, tet_ref.vertices()) {
                xi += lambda_i * x_i.coords;
            }
            Point3::from(xi)
        })
}

// This is copied from fenris source in order to prevent having this in the public API
#[replace_float_literals(T::from_f64(literal).unwrap())]
fn is_likely_in_tet_ref_interior<T: Real>(xi: &Point3<T>) -> bool {
    let eps = 4.0 * T::default_epsilon();
    xi.x >= -1.0 - eps && xi.y >= -1.0 - eps && xi.z >= -1.0 - eps && xi.x + xi.y + xi.z <= -1.0 + eps
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn is_definitely_in_tet_ref_interior<T: Real>(xi: &Point3<T>) -> bool {
    let eps = T::default_epsilon().sqrt();
    xi.x >= -1.0 + eps && xi.y >= -1.0 + eps && xi.z >= -1.0 + eps && xi.x + xi.y + xi.z <= -1.0 - eps
}

proptest! {
    #[test]
    fn point_in_tet_ref_domain_inside_ref_tet(xi in point_in_tet_ref_domain()) {
        prop_assert!(is_likely_in_tet_ref_interior(&xi));
    }
}

#[test]
fn sample_points_in_tet_ref_domain() {
    let mut runner = TestRunner::deterministic();
    let tet_point_strategy = point_in_tet_ref_domain();
    let points: Vec<_> = (0 .. 1000)
        .map(|_| tet_point_strategy.new_tree(&mut runner).unwrap().current().clone())
        .collect();

    // TODO: Actually export a point cloud and not a "fake" tet mesh
    let mesh = Tet4Mesh::from_vertices_and_connectivity(points, vec![]);
    export_mesh_vtk("sample_points_in_tet_ref_domain", "sampled_points", &mesh);

    let reference_tet = Tet4Element::reference();
    let reference_tet_mesh = Tet4Mesh::from_vertices_and_connectivity(
        reference_tet.vertices().to_vec(),
        vec![Tet4Connectivity([0, 1, 2, 3])]);
    export_mesh_vtk("sample_points_in_tet_ref_domain", "reference_tet", &reference_tet_mesh);
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

use fenris_geometry::util::compute_orthonormal_vectors_3d;
use matrixcompare::prop_assert_scalar_eq;
use nalgebra::{UnitVector3, Vector3};
use proptest::prelude::*;
use std::f64::consts::PI;

fn unit_vector3() -> impl Strategy<Value = UnitVector3<f64>> {
    // Generate spherical coordinates on the unit sphere
    (0.0..PI, 0.0..2.0 * PI).prop_map(|(theta, phi)| {
        let x = phi.cos() * theta.sin();
        let y = phi.sin() * theta.sin();
        let z = theta.cos();
        UnitVector3::new_normalize(Vector3::new(x, y, z))
    })
}

proptest! {
    #[test]
    fn compute_orthogonal_vectors_3d_vectors_are_orthogonal(v in unit_vector3()) {
        let [t1, t2] = compute_orthonormal_vectors_3d(&v);

        let tol = 1e-14;
        prop_assert_scalar_eq!(t1.norm(), 1.0, comp = abs, tol = tol);
        prop_assert_scalar_eq!(t2.norm(), 1.0, comp = abs, tol = tol);
        prop_assert_scalar_eq!(v.dot(&t1), 0.0, comp = abs, tol = tol);
        prop_assert_scalar_eq!(v.dot(&t2), 0.0, comp = abs, tol = tol);
        prop_assert_scalar_eq!(t1.dot(&t2), 0.0, comp = abs, tol = tol);
    }
}

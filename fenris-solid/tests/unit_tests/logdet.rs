use fenris::nalgebra;
use fenris::nalgebra::{matrix, vector, Matrix3, Rotation3};
use fenris_solid::{log_det_F, u_grad_from_F};
use matrixcompare::assert_scalar_eq;

#[allow(non_snake_case)]
fn arbitrary_F() -> Matrix3<f64> {
    // We construct F by an ad-hoc SVD construction. This way we can control the singular values
    // and determinant (since rotations have determinant 1)
    let rot1 = Rotation3::from_scaled_axis(vector![1.0, 2.0, -3.0])
        .matrix()
        .clone_owned();
    let rot2 = Rotation3::from_scaled_axis(vector![4.0, -5.0, 6.0])
        .matrix()
        .clone_owned();
    let sigma = matrix![1.0 + 0.1, 0.0, 0.0;
                            0.0, 1.0 - 0.2, 0.0;
                            0.0, 0.0, 1.0 + 0.3];
    rot1 * sigma * rot2
}

#[test]
#[allow(non_snake_case)]
fn test_log_det_F() {
    let F = arbitrary_F();
    let du_dX = F - Matrix3::identity();
    assert_scalar_eq!(
        log_det_F(&du_dX).unwrap(),
        F.determinant().ln(),
        comp = abs,
        tol = 1e-12
    );
}

#[test]
#[allow(non_snake_case)]
fn log_det_negative_determinant() {
    let rot1 = Rotation3::from_scaled_axis(vector![1.0, 2.0, -3.0])
        .matrix()
        .clone_owned();
    let rot2 = Rotation3::from_scaled_axis(vector![4.0, -5.0, 6.0])
        .matrix()
        .clone_owned();
    let sigma = matrix![-1e-6, 0.0, 0.0;
                            0.0, 1.0, 0.0;
                            0.0, 0.0, 1.0 + 0.3];
    let F = rot1 * sigma * rot2;
    let du_dX = u_grad_from_F(&F).transpose();
    assert!(log_det_F(&du_dX).is_none());
}

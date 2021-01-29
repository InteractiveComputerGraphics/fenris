use nalgebra::{DMatrix, DVector};
use sparse::cg::{ConjugateGradient, IdentityOperator, RelativeResidualCriterion};
use util::assert_approx_matrix_eq;

#[test]
fn solve_identity() {
    let operator = IdentityOperator;
    let mut x = DVector::zeros(4);
    let b = DVector::from_element(4, 5.0);
    ConjugateGradient::new()
        .with_operator(operator)
        .with_stopping_criterion(RelativeResidualCriterion::default())
        .solve_with_guess(&b, &mut x)
        .unwrap();
    assert_eq!(x, b);
}

#[test]
fn solve_arbitrary() {
    // Use an arbitrary symmetric, positive definite matrix (diagonally dominant in this case)
    // and the identity preconditioner
    let a = DMatrix::from_fn(3, 3, |r, c| if r == c { 7.0 } else { 2.0 });

    let x0 = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
    let b = &a * &x0;
    let mut x = DVector::zeros(3);
    let output = ConjugateGradient::new()
        .with_operator(&a)
        .with_stopping_criterion(RelativeResidualCriterion::new(1e-14))
        .solve_with_guess(&b, &mut x)
        .unwrap();

    assert!(output.num_iterations > 0 && output.num_iterations <= 3);
    assert_approx_matrix_eq!(&x, &x0, abstol = 1e-12);
}

#[test]
fn solve_arbitrary_preconditioned() {
    // Take some arbitrary positive definite matrices as system matrix and preconditioner
    let a = DMatrix::from_row_slice(
        3,
        3,
        &[21.0, -1.0, -5.0, -1.0, 11.0, -4.0, -5.0, -4.0, 26.0],
    );
    let p = DMatrix::from_row_slice(3, 3, &[17.0, 6.0, 3.0, 6.0, 14.0, 9.0, 3.0, 9.0, 10.0]);
    let x0 = DVector::from_column_slice(&[1.0, 3.0, 2.0]);
    let b = &a * &x0;

    // Arbitrary initial guess
    let mut x = DVector::from_column_slice(&[2.0, 1.0, 0.0]);
    let output = ConjugateGradient::new()
        .with_operator(&a)
        .with_preconditioner(&p)
        .with_stopping_criterion(RelativeResidualCriterion::new(1e-14))
        .solve_with_guess(&b, &mut x)
        .unwrap();

    // We use the fact that CG converges in exact arithmetic in at most n iterations
    // for an n x n matrix. For such a small matrix, we should reach high precision immediately.
    assert_eq!(output.num_iterations, 3);
    assert_approx_matrix_eq!(&x, &x0, abstol = 1e-12);
}

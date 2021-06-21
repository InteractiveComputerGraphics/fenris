use fenris_optimize::calculus::*;
use matrixcompare::assert_matrix_eq;
use nalgebra::{DMatrix, DVector, DVectorSlice, DVectorSliceMut};

#[test]
fn approximate_jacobian_simple_function() {
    struct SimpleTwoDimensionalPolynomial;

    // TODO: Rewrite with VectorFunctionBuilder

    impl VectorFunction<f64> for SimpleTwoDimensionalPolynomial {
        fn dimension(&self) -> usize {
            2
        }

        fn eval_into(&mut self, f: &mut DVectorSliceMut<f64>, x: &DVectorSlice<f64>) {
            assert_eq!(x.len(), 2);
            assert_eq!(f.len(), x.len());
            let x1 = x[0];
            let x2 = x[1];
            f[0] = x1 * x2 + 3.0;
            f[1] = x1 * x1 + x2 * x2 + x1 + 5.0;
        }
    }

    let h = 1e-6;
    let x = DVector::from_column_slice(&[3.0, 4.0]);
    let j = approximate_jacobian(SimpleTwoDimensionalPolynomial, &x, &h);

    // J = [   x2           x1 ]
    //     [ 2*x1 + 1     2*x2 ]

    #[rustfmt::skip]
    let expected = DMatrix::from_row_slice(2, 2,
                                           &[4.0, 3.0,
                                             7.0, 8.0]);

    let diff = expected - j;
    assert!(diff.norm() < 1e-6);
}

#[test]
fn test_approximate_gradient_fd() {
    // Define some function f and its gradient
    let f = |x: DVectorSlice<f64>| {
        let (x, y, z) = (x[0], x[1], x[2]);
        3.0 * x * x * x + 3.0 * x * y - 5.0 * z * z + 2.0
    };
    let f_grad = |x: DVectorSlice<f64>| {
        let (x, y, z) = (x[0], x[1], x[2]);
        DVector::from_column_slice(&[9.0 * x * x + 3.0 * y, 3.0 * x, -10.0 * z])
    };

    let x = DVector::from_column_slice(&[3.0, 4.0, 5.0]);
    let f_grad_fd = approximate_gradient_fd(f, &x, 1e-6);

    assert_matrix_eq!(
        f_grad_fd,
        f_grad(DVectorSlice::from(&x)),
        comp = abs,
        tol = 1e-6
    );
}

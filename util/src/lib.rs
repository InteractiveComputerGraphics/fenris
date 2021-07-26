use nalgebra::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use nalgebra::storage::{Storage, StorageMut};
use nalgebra::{Dim, Dynamic, Matrix, MatrixMN, Scalar};
use num::Zero;

/// Poor man's approx assertion for matrices
#[macro_export]
macro_rules! assert_approx_matrix_eq {
    ($x:expr, $y:expr, abstol = $tol:expr) => {{
        let diff = $x - $y;

        let max_absdiff = diff.abs().max();
        let approx_eq = max_absdiff <= $tol;

        if !approx_eq {
            println!("abstol: {:e}", $tol);
            println!("left: {}", $x);
            println!("right: {}", $y);
            println!("diff: {:e}", diff);
        }
        assert!(approx_eq);
    }};
}

#[macro_export]
macro_rules! assert_panics {
    ($e:expr) => {{
        use std::panic::catch_unwind;
        use std::stringify;
        let expr_string = stringify!($e);
        let result = catch_unwind(|| $e);
        if result.is_ok() {
            panic!("assert_panics!({}) failed.", expr_string);
        }
    }};
}

pub fn flatten_vertically_into<T, R1, C1, S1, R2, C2, S2>(
    output: &mut Matrix<T, R2, C2, S2>,
    matrices: &[Matrix<T, R1, C1, S1>],
) where
    T: Scalar,
    R1: Dim,
    C1: Dim,
    S1: Storage<T, R1, C1>,
    R2: Dim,
    C2: Dim,
    S2: StorageMut<T, R2, C2>,
    ShapeConstraint: SameNumberOfColumns<C2, C1> + SameNumberOfRows<Dynamic, R1>,
{
    if let Some(first) = matrices.first() {
        let cols = first.ncols();
        let mut rows = 0;

        for matrix in matrices {
            assert_eq!(matrix.ncols(), cols, "All matrices must have same number of columns.");
            output.rows_mut(rows, matrix.nrows()).copy_from(matrix);
            rows += matrix.nrows();
        }
        assert_eq!(
            rows,
            output.nrows(),
            "Number of rows in output must match number of total rows in input."
        );
    } else {
        assert_eq!(
            output.nrows(),
            0,
            "Can only vertically flatten empty slice of matrices into a matrix with 0 rows."
        );
    }
}

pub fn flatten_vertically<T, R, C, S>(matrices: &[Matrix<T, R, C, S>]) -> Option<MatrixMN<T, Dynamic, C>>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    ShapeConstraint: SameNumberOfRows<Dynamic, R>,
{
    if let Some(first) = matrices.first() {
        let rows = matrices.iter().map(Matrix::nrows).sum();
        let mut output = MatrixMN::zeros_generic(Dynamic::new(rows), first.data.shape().1);
        flatten_vertically_into(&mut output, matrices);
        Some(output)
    } else {
        None
    }
}

pub fn prefix_sum(counts: impl IntoIterator<Item = usize>, x0: usize) -> impl Iterator<Item = usize> {
    counts.into_iter().scan(x0, |sum, x| {
        let current = *sum;
        *sum += x;
        Some(current)
    })
}

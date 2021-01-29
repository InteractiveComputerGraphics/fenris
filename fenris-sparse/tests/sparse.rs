use fenris_sparse::{CooMatrix, CsrMatrix};

use std::ops::Add;

use nalgebra::{DMatrix, DVector, Vector3};

use ::proptest::collection::vec;
use ::proptest::prelude::*;
use itertools::Itertools;
use fenris_sparse::{spmm_csr, spmm_csr_pattern, spmv_csr, CscMatrix};
use fenris_sparse_proptest;
use fenris_sparse_proptest::{CsrStrategy, SparsityPatternStrategy};
use std::sync::Arc;
use util::assert_approx_matrix_eq;
use util::assert_panics;
use util::flatten_vertically;

#[test]
fn coo_to_csr_sorted_no_duplicates() {
    let mut coo = CooMatrix::new(4, 3);
    coo.push(0, 1, 1);
    coo.push(2, 0, 2);
    coo.push(2, 1, 3);
    coo.push(2, 2, 4);
    coo.push(3, 2, 5);

    let csr = coo.to_csr(Add::add);

    let expected_csr = CsrMatrix::from_csr_data(
        4,
        3,
        vec![0, 1, 1, 4, 5],
        vec![1, 0, 1, 2, 2],
        vec![1, 2, 3, 4, 5],
    );

    assert_eq!(csr, expected_csr);
}

#[test]
fn coo_to_csr_minimal_example() {
    let mut coo = CooMatrix::new(1, 2);
    coo.push(0, 1, 0);
    coo.push(0, 0, 0);
    coo.push(0, 0, 1);

    let csr = coo.to_csr(Add::add);
    let expected_csr = CsrMatrix::from_csr_data(1, 2, vec![0, 2], vec![0, 1], vec![1, 0]);

    assert_eq!(csr, expected_csr);
}

#[test]
fn coo_to_csr_sorted_with_duplicates() {
    let mut coo = CooMatrix::new(4, 3);
    coo.push(0, 1, 1);
    coo.push(2, 0, 2);
    coo.push(2, 0, -1);
    coo.push(2, 1, 3);
    coo.push(2, 2, 4);
    coo.push(3, 2, 5);

    let csr = coo.to_csr(Add::add);

    let expected_csr = CsrMatrix::from_csr_data(
        4,
        3,
        vec![0, 1, 1, 4, 5],
        vec![1, 0, 1, 2, 2],
        vec![1, 1, 3, 4, 5],
    );

    assert_eq!(csr, expected_csr);
}

#[test]
fn coo_to_csr_unsorted_without_duplicates() {
    let mut coo = CooMatrix::new(4, 3);
    coo.push(3, 2, 5);
    coo.push(2, 1, 3);
    coo.push(2, 2, 4);
    coo.push(0, 1, 1);
    coo.push(2, 0, 2);

    let csr = coo.to_csr(Add::add);

    let expected_csr = CsrMatrix::from_csr_data(
        4,
        3,
        vec![0, 1, 1, 4, 5],
        vec![1, 0, 1, 2, 2],
        vec![1, 2, 3, 4, 5],
    );

    assert_eq!(csr, expected_csr);
}

#[test]
fn csc_to_dense() {
    // Dense matrix:
    let dense = DMatrix::from_row_slice(
        4,
        5,
        &vec![1, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 7],
    );

    let column_offsets = vec![0, 2, 4, 6, 6, 7];
    let row_indices = vec![0, 3, 1, 3, 0, 3, 3];
    let values = vec![1, 4, 3, 5, 2, 6, 7];

    let csc = CscMatrix::from_csc_data(4, 5, column_offsets, row_indices, values);
    assert_eq!(csc.to_dense(), dense);
}

#[test]
fn spmv() {
    let matrix = DMatrix::from_row_slice(4, 3, &vec![1, 0, 2, 0, 0, 0, 3, 4, 0, 5, 6, 0]);
    let csr = CsrMatrix::from(&matrix);

    let x = DVector::from_iterator(3, vec![1, 2, 3]);
    let mut y = DVector::from_iterator(4, vec![4, 5, 6, 7]);
    let alpha = 2;
    let beta = -1;

    spmv_csr(beta, &mut y, alpha, &csr, &x);

    let y_expected = DVector::from_iterator(4, vec![10, -5, 16, 27]);

    assert_eq!(y, y_expected);
}

fn default_csr_i32() -> impl Strategy<Value = CsrMatrix<i32>> {
    CsrStrategy::new()
        .with_shapes((0usize..5, 0usize..5))
        .with_cols_per_row(0usize..5)
        .with_elements(-9i32..9i32)
}

#[test]
fn csr_from_diagonal() {
    let a = CsrMatrix::from_diagonal(&Vector3::new(2.0, -1.0, -4.0));

    let a_diag: Vec<_> = a.diag_iter().collect();
    let a_expected_diag = vec![2.0, -1.0, -4.0];
    assert_eq!(a.nnz(), 3);
    assert_eq!(a.nrows(), a.ncols());
    assert_eq!(a.nrows(), 3);
    assert_eq!(a_diag, a_expected_diag);
}

#[test]
fn scale_rows_cols() {
    let a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let a = CsrMatrix::from(&a);

    let d = DVector::from_column_slice(&[2.0, -4.0, 3.0]);

    // Scale rows
    {
        let mut a = a.clone();
        a.scale_rows(&d);
        let a_dense = a.build_dense();
        let a_expected = DMatrix::from_row_slice(
            3,
            3,
            &[2.0, 4.0, 6.0, -16.0, -20.0, -24.0, 21.0, 24.0, 27.0],
        );
        assert_approx_matrix_eq!(&a_dense, &a_expected, abstol = 1e-12);
    }

    // Scale cols
    {
        let mut a = a.clone();
        a.scale_cols(&d);
        let a_dense = a.build_dense();
        let a_expected =
            DMatrix::from_row_slice(3, 3, &[2.0, -8.0, 9.0, 8.0, -20.0, 18.0, 14.0, -32.0, 27.0]);
        assert_approx_matrix_eq!(&a_dense, &a_expected, abstol = 1e-12);
    }
}

#[test]
fn csr_row_mut() {
    let mut csr = {
        let mut coo = CooMatrix::new(4, 4);
        coo.push(0, 0, 1);
        coo.push(0, 1, 2);
        coo.push(0, 3, 3);
        coo.push(1, 1, 4);
        coo.push(1, 3, 5);
        coo.push(2, 0, 6);
        coo.push(2, 2, 7);
        coo.push(3, 1, 8);
        coo.push(3, 2, 9);
        coo.to_csr(Add::add)
    };

    // Reads only
    {
        let row = csr.row_mut(0);
        assert_eq!(row.value_at_local_index(0), &1);
        assert_eq!(row.value_at_local_index(1), &2);
        assert_eq!(row.value_at_local_index(2), &3);
        assert_panics!(row.value_at_local_index(3));
        assert_eq!(row.col_at_local_index(0), 0);
        assert_eq!(row.col_at_local_index(1), 1);
        assert_eq!(row.col_at_local_index(2), 3);
        assert_panics!(row.col_at_local_index(3));

        let row = csr.row_mut(1);
        assert_eq!(row.value_at_local_index(0), &4);
        assert_eq!(row.value_at_local_index(1), &5);
        assert_panics!(row.value_at_local_index(2));
        assert_eq!(row.col_at_local_index(0), 1);
        assert_eq!(row.col_at_local_index(1), 3);
        assert_panics!(row.col_at_local_index(2));

        let row = csr.row_mut(2);
        assert_eq!(row.value_at_local_index(0), &6);
        assert_eq!(row.value_at_local_index(1), &7);
        assert_panics!(row.value_at_local_index(2));
        assert_eq!(row.col_at_local_index(0), 0);
        assert_eq!(row.col_at_local_index(1), 2);
        assert_panics!(row.col_at_local_index(2));

        let row = csr.row_mut(3);
        assert_eq!(row.value_at_local_index(0), &8);
        assert_eq!(row.value_at_local_index(1), &9);
        assert_panics!(row.value_at_local_index(2));
        assert_eq!(row.col_at_local_index(0), 1);
        assert_eq!(row.col_at_local_index(1), 2);
        assert_panics!(row.col_at_local_index(2));
    }

    // TODO: More tests
}

#[test]
fn test_spmm_csr_pattern() {
    let a = DMatrix::from_row_slice(
        4,
        5,
        &[1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    );
    let b = DMatrix::from_row_slice(5, 3, &[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]);
    let c = DMatrix::from_row_slice(4, 3, &[2, 1, 3, 0, 0, 1, 0, 0, 0, 2, 1, 3]);

    let a_pattern = CsrMatrix::from(&a).sparsity_pattern();
    let b_pattern = CsrMatrix::from(&b).sparsity_pattern();
    let c_pattern = spmm_csr_pattern(&a_pattern, &b_pattern);

    let c_pattern_expected = CsrMatrix::from(&c).sparsity_pattern();

    assert_eq!(&c_pattern, c_pattern_expected.as_ref());
    assert_eq!(c_pattern.nnz(), 7);
}

#[test]
fn test_spmm_csr() {
    {
        let a = DMatrix::from_row_slice(2, 2, &[1, 0, 0, 3]);
        let b = DMatrix::from_row_slice(2, 1, &[3, -1]);
        let c = DMatrix::from_row_slice(2, 1, &[1, 1]);

        let a = CsrMatrix::from(&a);
        let b = CsrMatrix::from(&b);
        let mut c = CsrMatrix::from(&c);

        let alpha = 1;
        let beta = 0;

        spmm_csr(beta, &mut c, alpha, &a, &b).unwrap();

        let expected = DMatrix::from_row_slice(2, 1, &[3, -3]);
        let expected = CsrMatrix::from(&expected);

        assert_eq!(c, expected);
        assert_eq!(c.nnz(), 2);
    }

    {
        let a = DMatrix::from_row_slice(3, 2, &[1, 0, 0, 0, 2, 0]);
        let b = DMatrix::from_row_slice(2, 3, &[3, 0, 1, -1, 1, 0]);
        let c = DMatrix::from_row_slice(3, 3, &[3, 0, 1, -3, 3, 0, 6, 0, 2]);

        let a = CsrMatrix::from(&a);
        let b = CsrMatrix::from(&b);
        let mut c = CsrMatrix::from(&c);

        let alpha = 2;
        let beta = 2;

        spmm_csr(beta, &mut c, alpha, &a, &b).unwrap();

        let expected = DMatrix::from_row_slice(3, 3, &[12, 0, 4, -6, 6, 0, 24, 0, 8]);
        let expected = CsrMatrix::from(&expected);

        assert_eq!(c, expected);
        assert_eq!(c.nnz(), 6);
    }

    // Minimal example found by proptest
    {
        let c_offsets = vec![0, 0, 3];
        let c_indices = vec![0, 1, 2];
        let c_values = vec![0, 0, 0];
        let mut c = CsrMatrix::from_csr_data(2, 3, c_offsets, c_indices, c_values);

        let a_offsets = vec![0, 0, 2];
        let a_indices = vec![0, 1];
        let a_values = vec![8, 5];
        let a = CsrMatrix::from_csr_data(2, 2, a_offsets, a_indices, a_values);

        let b_offsets = vec![0, 0, 3];
        let b_indices = vec![0, 1, 2];
        let b_values = vec![0, 0, -1];
        let b = CsrMatrix::from_csr_data(2, 3, b_offsets, b_indices, b_values);

        let alpha = 3;
        let beta = 2;

        spmm_csr(beta, &mut c, alpha, &a, &b).unwrap();

        let expected_offsets = vec![0, 0, 3];
        let expected_indices = vec![0, 1, 2];
        let expected_values = vec![0, 0, -15];
        let expected =
            CsrMatrix::from_csr_data(2, 3, expected_offsets, expected_indices, expected_values);

        assert_eq!(c, expected);
    }

    {
        let a = DMatrix::from_row_slice(
            4,
            5,
            &[1, 3, 0, 0, 4, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 0, 0, 2],
        );
        let b = DMatrix::from_row_slice(5, 3, &[2, 0, 1, 0, 0, 5, -1, 0, 0, 0, -4, 0, 1, 3, 1]);
        let c = DMatrix::from_row_slice(4, 3, &[2, 1, 3, 0, 0, 1, 0, 0, 0, 2, 1, 3]);

        let a = CsrMatrix::from(&a);
        let b = CsrMatrix::from(&b);
        let mut c = CsrMatrix::from(&c);

        let alpha = 2;
        let beta = 3;

        spmm_csr(beta, &mut c, alpha, &a, &b).unwrap();

        let expected = DMatrix::from_row_slice(4, 3, &[18, 27, 49, 0, 0, -47, 0, 0, 0, 14, 15, -5]);
        let expected = CsrMatrix::from(&expected);

        assert_eq!(c, expected);
        assert_eq!(c.nnz(), 7);
    }
}

/// Strategy for generating matrices A, B, C such that C = A * B is a valid expression.
fn spmm_compatible_csr_matrices() -> impl Strategy<Value = (CsrMatrix<i32>, [CsrMatrix<i32>; 2])> {
    let a_strategy = CsrStrategy::new()
        .with_shapes((0usize..5, 0usize..5))
        .with_cols_per_row(0usize..7)
        .with_elements(-9i32..9);

    a_strategy
        .prop_flat_map(|a| {
            let b_strategy = CsrStrategy::new()
                .with_shapes((Just(a.ncols()), 0usize..5))
                .with_cols_per_row(0usize..7)
                .with_elements(-9i32..9);
            (Just(a), b_strategy)
        })
        .prop_flat_map(|(a, b)| {
            let c_pattern = Arc::new(spmm_csr_pattern(
                &a.sparsity_pattern(),
                &b.sparsity_pattern(),
            ));

            let nnz = c_pattern.nnz();
            let c_strategy = vec(-9i32..9, nnz).prop_map(move |values| {
                CsrMatrix::from_pattern_and_values(c_pattern.clone(), values)
            });

            (c_strategy, [Just(a), Just(b)])
        })
}

proptest! {
    #[test]
    fn coo_csr_identical_dense_representations(
        coo in fenris_sparse_proptest::coo(4, 4, 10, -5..5)
    ) {
        let coo_as_dense = coo.build_dense();
        let csr = coo.to_csr(Add::add);
        let csr_as_dense = csr.build_dense();

        prop_assert_eq!(coo_as_dense, csr_as_dense)
    }

    #[test]
    fn csr_spmv_same_as_dense_gemv(
        (coo, x, y, alpha, beta) in (
            fenris_sparse_proptest::coo(6, 6, 10, -5..5),
            vec(-5..5, 6),
            vec(-5..5, 6),
            -5..5,
            -5..5)
    ) {
        // TODO: It seems `nalgebra` does not compute the correct gemv result
        // for zero-sized matrices. Should make an issue with a reproducible test case!
        prop_assume!(coo.nrows() > 0);
        prop_assume!(coo.ncols() > 0);

        let x = DVector::from_iterator(coo.ncols(), x.into_iter().take(coo.ncols()));
        let y = DVector::from_iterator(coo.nrows(), y.into_iter().take(coo.nrows()));
        let dense = coo.build_dense();
        let csr = coo.to_csr(Add::add);

        let mut y_gemv = y.clone();
        y_gemv.gemv(alpha, &dense, &x, beta);

        let mut y_spmv = y.clone();
        spmv_csr(beta, &mut y_spmv, alpha, &csr, &x);

        prop_assert_eq!(&y_spmv, &y_gemv);
    }

    #[test]
    fn csr_mat_mul_vec_same_as_dense(
        (coo, x) in (
            fenris_sparse_proptest::coo(6, 6, 10, -5..5),
            vec(-5..5, 6))
    ) {
        // TODO: It seems `nalgebra` does not compute the correct matrix-vector product
        // for zero-sized matrices. Should make an issue with a reproducible test case!
        prop_assume!(coo.nrows() > 0);
        prop_assume!(coo.ncols() > 0);

        let x = DVector::from_iterator(coo.ncols(), x.into_iter().take(coo.ncols()));
        let dense = coo.build_dense();
        let csr = coo.to_csr(Add::add);

        let y_dense = &dense * &x;
        let y_csr = &csr * &x;

        prop_assert_eq!(&y_csr, &y_dense);
    }

    #[test]
    fn csr_to_dense(
        csr in CsrStrategy::new()
                .with_shapes((0usize..7, 0usize.. 7))
                .with_cols_per_row(0usize..7)
                .with_elements(-9i32..9))
    {
        println!("{}", csr.build_dense());
    }

    #[test]
    fn csr_append_csr_rows(
        (csr1, csr2) in (0usize .. 8).prop_flat_map(|cols| {
            // Generate pairs of CSR matrices with the same number of columns
            let shape_strategy = (0usize .. 6, Just(cols));
            let csr_strategy = CsrStrategy::new().with_shapes(shape_strategy)
                .with_cols_per_row(0usize ..= 5)
                .with_elements(-9i32..9);
            (csr_strategy.clone(), csr_strategy)
        }))
    {
        // Sanity check for test generation
        prop_assert_eq!(csr1.ncols(), csr2.ncols());

        let mut csr_result = csr1.clone();
        csr_result.append_csr_rows(&csr2);

        prop_assert_eq!(csr_result.ncols(), csr1.ncols());
        prop_assert_eq!(csr_result.nrows(), csr1.nrows() + csr2.nrows());
        prop_assert_eq!(csr_result.nnz(), csr1.nnz() + csr2.nnz());

        // Check that the result agrees with the dense concatenation
        let dense1 = csr1.build_dense();
        let dense2 = csr2.build_dense();
        let dense_result = csr_result.build_dense();
        let dense_expected = flatten_vertically(&[dense1, dense2]).unwrap();

        prop_assert_eq!(dense_result, dense_expected);
    }

    #[test]
    fn csr_to_csc(
        csr in CsrStrategy::new().with_shapes((0usize .. 5, 0usize .. 5))
                                 .with_cols_per_row(0usize .. 5)
                                 .with_elements(-9i32..9i32)
    )
    {
        let csc = csr.to_csc();

        prop_assert_eq!(csc.nrows(), csr.nrows());
        prop_assert_eq!(csc.ncols(), csr.ncols());
        prop_assert_eq!(csc.nnz(), csr.nnz());

        prop_assert_eq!(csr.build_dense(), csc.to_dense());
    }

    #[test]
    fn concat_diagonally(matrices in vec(default_csr_i32(), 0..5)) {
        let concatenated = CsrMatrix::concat_diagonally(&matrices);

        let expected_nnz = matrices.iter().map(CsrMatrix::nnz).sum();
        let expected_nrows = matrices.iter().map(CsrMatrix::nrows).sum();
        let expected_ncols = matrices.iter().map(CsrMatrix::ncols).sum();

        prop_assert_eq!(concatenated.nnz(), expected_nnz);
        prop_assert_eq!(concatenated.nrows(), expected_nrows);
        prop_assert_eq!(concatenated.ncols(), expected_ncols);

        // TODO: Would be nice to have a method for diagonally concatenating matrices in
        // nalgebra, then we could simply compare with the result of the dense concatenation
        // and we'd be done
        let concat_dense = concatenated.build_dense();
        assert_eq!(concat_dense.nrows(), concatenated.nrows());
        assert_eq!(concat_dense.ncols(), concatenated.ncols());

        let mut row_offset = 0;
        let mut col_offset = 0;

        for matrix in &matrices {
            // TODO: It seems as if the .get and .index methods cannot take
            // inputs of the form 0 .. 0. Should make an issue about this.
            // For now, we work around this by explicitly checking nrows and ncols
            if matrix.nrows() > 0 && matrix.ncols() > 0 {
                let region = (row_offset .. row_offset + matrix.nrows(),
                          col_offset .. col_offset + matrix.ncols());
                let concat_slice = concat_dense.index(region);
                let matrix_dense = matrix.build_dense();

                prop_assert_eq!(matrix_dense, concat_slice.clone_owned());
            }

            row_offset += matrix.nrows();
            col_offset += matrix.ncols();
        }
    }

    #[test]
    fn csr_spmm_matches_dense_results(
        (mut c, [a, b]) in spmm_compatible_csr_matrices()
    )
    {
        let a_dense = a.build_dense();
        let b_dense = b.build_dense();
        let mut c_dense = c.build_dense();

        let alpha = 3;
        let beta = 2;

        c_dense.gemm(alpha, &a_dense, &b_dense, beta);
        spmm_csr(beta, &mut c, alpha, &a, &b).expect("Matrices should be compatible by definition");

        prop_assert_eq!(c.build_dense(), c_dense);
    }

    #[test]
    fn csr_mul_csr_matches_dense_results(
        (_, [a, b]) in spmm_compatible_csr_matrices()
    )
    {
        let a_dense = a.build_dense();
        let b_dense = b.build_dense();
        let c_dense = &a_dense * &b_dense;

        let c = &a * &b;

        prop_assert_eq!(&c.build_dense(), &c_dense);
    }

    #[test]
    fn sparsity_pattern_strategy_respects_strategies(
        pattern in SparsityPatternStrategy::new()
                    .with_shapes((Just(5), 2usize..=3))
                    .with_num_minors_per_major(1usize ..= 2))
    {
        prop_assert_eq!(pattern.major_dim(), 5);
        prop_assert!(pattern.minor_dim() >= 2);
        prop_assert!(pattern.minor_dim() <= 3);

        let counts: Vec<_> = pattern.major_offsets()
            .iter()
            .tuple_windows()
            .map(|(prev, next)| next - prev)
            .collect();

        prop_assert!(counts.iter().cloned().all(|c| c >= 1 && c <= 2));
    }

    #[test]
    fn csr_strategy_respects_strategies(
        matrix in CsrStrategy::new()
                    .with_shapes((Just(5), 2usize..=3))
                    .with_cols_per_row(1usize..=2)
                    .with_elements(0i32..5))
    {
        prop_assert_eq!(matrix.nrows(), 5);
        prop_assert!(matrix.ncols() >= 2);
        prop_assert!(matrix.ncols() <= 3);
        prop_assert!(matrix.values().iter().cloned().all(|x| x >= 0 && x <= 5));

        let counts: Vec<_> = matrix.row_offsets()
            .iter()
            .tuple_windows()
            .map(|(prev, next)| next - prev)
            .collect();

        prop_assert!(counts.iter().cloned().all(|c| c >= 1 && c <= 2));
    }
}

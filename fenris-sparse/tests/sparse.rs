use ::proptest::prelude::*;

// TODO: Remove all these tests. Currently we keep them here in case we want to use them to test
// some similar functionality that we can port to nalgebra-sparse

// #[test]
// fn csr_from_diagonal() {
//     let a = CsrMatrix::from_diagonal(&Vector3::new(2.0, -1.0, -4.0));
//
//     let a_diag: Vec<_> = a.diag_iter().collect();
//     let a_expected_diag = vec![2.0, -1.0, -4.0];
//     assert_eq!(a.nnz(), 3);
//     assert_eq!(a.nrows(), a.ncols());
//     assert_eq!(a.nrows(), 3);
//     assert_eq!(a_diag, a_expected_diag);
// }

// #[test]
// fn scale_rows_cols() {
//     let a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
//     let a = CsrMatrix::from(&a);
//
//     let d = DVector::from_column_slice(&[2.0, -4.0, 3.0]);
//
//     // Scale rows
//     {
//         let mut a = a.clone();
//         a.scale_rows(&d);
//         let a_dense = a.build_dense();
//         let a_expected = DMatrix::from_row_slice(
//             3,
//             3,
//             &[2.0, 4.0, 6.0, -16.0, -20.0, -24.0, 21.0, 24.0, 27.0],
//         );
//         assert_approx_matrix_eq!(&a_dense, &a_expected, abstol = 1e-12);
//     }
//
//     // Scale cols
//     {
//         let mut a = a.clone();
//         a.scale_cols(&d);
//         let a_dense = a.build_dense();
//         let a_expected =
//             DMatrix::from_row_slice(3, 3, &[2.0, -8.0, 9.0, 8.0, -20.0, 18.0, 14.0, -32.0, 27.0]);
//         assert_approx_matrix_eq!(&a_dense, &a_expected, abstol = 1e-12);
//     }
// }

proptest! {
    // #[test]
    // fn csr_append_csr_rows(
    //     (csr1, csr2) in (0usize .. 8).prop_flat_map(|cols| {
    //         // Generate pairs of CSR matrices with the same number of columns
    //         let shape_strategy = (0usize .. 6, Just(cols));
    //         let csr_strategy = CsrStrategy::new().with_shapes(shape_strategy)
    //             .with_cols_per_row(0usize ..= 5)
    //             .with_elements(-9i32..9);
    //         (csr_strategy.clone(), csr_strategy)
    //     }))
    // {
    //     // Sanity check for test generation
    //     prop_assert_eq!(csr1.ncols(), csr2.ncols());
    //
    //     let mut csr_result = csr1.clone();
    //     csr_result.append_csr_rows(&csr2);
    //
    //     prop_assert_eq!(csr_result.ncols(), csr1.ncols());
    //     prop_assert_eq!(csr_result.nrows(), csr1.nrows() + csr2.nrows());
    //     prop_assert_eq!(csr_result.nnz(), csr1.nnz() + csr2.nnz());
    //
    //     // Check that the result agrees with the dense concatenation
    //     let dense1 = csr1.build_dense();
    //     let dense2 = csr2.build_dense();
    //     let dense_result = csr_result.build_dense();
    //     let dense_expected = flatten_vertically(&[dense1, dense2]).unwrap();
    //
    //     prop_assert_eq!(dense_result, dense_expected);
    // }
    //
    // #[test]
    // fn concat_diagonally(matrices in vec(default_csr_i32(), 0..5)) {
    //     let concatenated = CsrMatrix::concat_diagonally(&matrices);
    //
    //     let expected_nnz = matrices.iter().map(CsrMatrix::nnz).sum();
    //     let expected_nrows = matrices.iter().map(CsrMatrix::nrows).sum();
    //     let expected_ncols = matrices.iter().map(CsrMatrix::ncols).sum();
    //
    //     prop_assert_eq!(concatenated.nnz(), expected_nnz);
    //     prop_assert_eq!(concatenated.nrows(), expected_nrows);
    //     prop_assert_eq!(concatenated.ncols(), expected_ncols);
    //
    //     // TODO: Would be nice to have a method for diagonally concatenating matrices in
    //     // nalgebra, then we could simply compare with the result of the dense concatenation
    //     // and we'd be done
    //     let concat_dense = concatenated.build_dense();
    //     assert_eq!(concat_dense.nrows(), concatenated.nrows());
    //     assert_eq!(concat_dense.ncols(), concatenated.ncols());
    //
    //     let mut row_offset = 0;
    //     let mut col_offset = 0;
    //
    //     for matrix in &matrices {
    //         // TODO: It seems as if the .get and .index methods cannot take
    //         // inputs of the form 0 .. 0. Should make an issue about this.
    //         // For now, we work around this by explicitly checking nrows and ncols
    //         if matrix.nrows() > 0 && matrix.ncols() > 0 {
    //             let region = (row_offset .. row_offset + matrix.nrows(),
    //                       col_offset .. col_offset + matrix.ncols());
    //             let concat_slice = concat_dense.index(region);
    //             let matrix_dense = matrix.build_dense();
    //
    //             prop_assert_eq!(matrix_dense, concat_slice.clone_owned());
    //         }
    //
    //         row_offset += matrix.nrows();
    //         col_offset += matrix.ncols();
    //     }
    // }
}

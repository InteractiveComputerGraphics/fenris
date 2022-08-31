//! Functionality for sparse linear algebra.
//!
//! Some of it is intended to be ported to `nalgebra-sparse` later.
use fenris_paradis::{ParallelIndexedAccess, ParallelIndexedCollection};
use nalgebra_sparse::pattern::SparsityPattern;
use nalgebra_sparse::CsrMatrix;
use std::slice;

// // TODO: Do we want to try to remove duplicates? Probably not...
// pub fn from_offsets_and_unsorted_indices(
//     major_dim: usize,
//     minor_dim: usize,
//     major_offsets: Vec<usize>,
//     mut minor_indices: Vec<usize>,
// ) -> Self {
//     assert_eq!(major_offsets.len(), major_dim + 1);
//     assert_eq!(*major_offsets.last().unwrap(), minor_indices.len());
//     if major_offsets
//         .iter()
//         .tuple_windows()
//         .any(|(prev, next)| prev > next)
//     {
//         panic!("Offsets must be non-decreasing.");
//     }
//
//     for (major_begin, major_end) in major_offsets.iter().tuple_windows() {
//         let minor = &mut minor_indices[*major_begin..*major_end];
//         minor.sort_unstable();
//         if minor
//             .iter()
//             .tuple_windows()
//             .any(|(prev, next)| prev >= next)
//         {
//             panic!("Minor indices contain duplicates");
//         }
//     }
//
//     Self {
//         major_offsets,
//         minor_indices,
//         minor_dim,
//     }
// }

// /// Appends another sparsity pattern to this one, in the sense that it is extended
// /// along its major dimension.
// ///
// /// Panics if `self` and `other` have different minor dimensions.
// pub fn append_pattern(&mut self, other: &SparsityPattern) {
//     assert_eq!(self.minor_dim(), other.minor_dim());
//
//     let offset_begin = *self.major_offsets.last().unwrap();
//     let new_offsets_iter = other
//         .major_offsets()
//         .iter()
//         .map(|offset| offset + offset_begin);
//
//     self.major_offsets.pop();
//     self.major_offsets.extend(new_offsets_iter);
//     self.minor_indices.extend_from_slice(&other.minor_indices);
// }

// // TODO: Write tests
// pub fn diag_iter<'a>(&'a self) -> impl 'a + Iterator<Item = T>
// where
//     T: Zero + Clone,
// {
//     let ia = self.row_offsets();
//     let ja = self.column_indices();
//     (0..self.nrows()).map(move |i| {
//         let row_begin = ia[i];
//         let row_end = ia[i + 1];
//         let columns_in_row = &ja[row_begin..row_end];
//         if let Ok(idx) = columns_in_row.binary_search(&i) {
//             self.values()[row_begin + idx].clone()
//         } else {
//             T::zero()
//         }
//     })
// }

// pub fn from_diagonal<'a>(diagonal: impl Into<DVectorSlice<'a, T>>) -> Self
// where
//     T: Scalar,
// {
//     let diagonal = diagonal.into();
//     let vals = diagonal.iter().cloned().collect();
//     let num_rows = diagonal.len();
//     let ia = (0..(num_rows + 1)).collect();
//     let ja = (0..num_rows).collect();
//     Self::from_csr_data(num_rows, num_rows, ia, ja, vals)
// }
//
// pub fn from_pattern_and_values(pattern: Arc<SparsityPattern>, values: Vec<T>) -> Self {
//     assert_eq!(pattern.nnz(), values.len());
//     Self {
//         sparsity_pattern: pattern,
//         v: values,
//     }
// }

// /// Computes `self += a*x` where `x` is another matrix. Panics if the matrices are of different size.
// pub fn add_assign_scaled(&mut self, a: T, x: &Self)
// where
//     T: Clone + ClosedAdd + ClosedMul,
// {
//     assert_eq!(self.values_mut().len(), x.values().len());
//     for (v_i, x_i) in self.values_mut().iter_mut().zip(x.values().iter()) {
//         *v_i += a.clone() * x_i.clone();
//     }
// }

// pub fn append_csr_rows(&mut self, other: &CsrMatrix<T>)
// where
//     T: Clone,
// {
//     Arc::make_mut(&mut self.sparsity_pattern).append_pattern(&other.sparsity_pattern());
//     self.v.extend_from_slice(other.values());
// }

// pub fn concat_diagonally(matrices: &[CsrMatrix<T>]) -> CsrMatrix<T>
// where
//     T: Clone,
// {
//     let mut num_rows = 0;
//     let mut num_cols = 0;
//     let mut nnz = 0;
//
//     // This first pass over the matrices is cheap, since we don't access any of the data.
//     // We use this to be able to pre-allocate enough capacity so that no further
//     // reallocation will be necessary.
//     for matrix in matrices {
//         num_rows += matrix.nrows();
//         num_cols += matrix.ncols();
//         nnz += matrix.nnz();
//     }
//
//     let mut values = Vec::with_capacity(nnz);
//     let mut column_indices = Vec::with_capacity(nnz);
//     let mut row_offsets = Vec::with_capacity(num_rows + 1);
//
//     let mut col_offset = 0;
//     let mut nnz_offset = 0;
//     for matrix in matrices {
//         values.extend_from_slice(matrix.values());
//         column_indices.extend(matrix.column_indices().iter().map(|i| *i + col_offset));
//         row_offsets.extend(
//             matrix
//                 .row_offsets()
//                 .iter()
//                 .take(matrix.nrows())
//                 .map(|offset| *offset + nnz_offset),
//         );
//
//         col_offset += matrix.ncols();
//         nnz_offset += matrix.nnz();
//     }
//
//     row_offsets.push(nnz);
//
//     Self {
//         // TODO: Avoid validation of pattern for performance
//         sparsity_pattern: Arc::new(SparsityPattern::from_offsets_and_indices(
//             num_rows,
//             num_cols,
//             row_offsets,
//             column_indices,
//         )),
//         v: values,
//     }
// }

pub struct ParCsrRow<'a, T> {
    column_indices: &'a [usize],
    values: &'a [T],
}

pub struct ParCsrRowMut<'a, T> {
    column_indices: &'a [usize],
    values: *mut T,
}

impl<'a, T> ParCsrRow<'a, T> {
    /// Number of non-zeros in this row.
    pub fn nnz(&self) -> usize {
        self.column_indices.len()
    }

    pub fn values(&self) -> &[T] {
        self.values
    }

    pub fn col_indices(&self) -> &[usize] {
        self.column_indices
    }
}

impl<'a, T> ParCsrRowMut<'a, T> {
    /// Number of non-zeros in this row.
    pub fn nnz(&self) -> usize {
        self.column_indices.len()
    }

    pub fn values_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.values, self.column_indices.len()) }
    }

    pub fn cols_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        let values_mut = unsafe { slice::from_raw_parts_mut(self.values, self.column_indices.len()) };
        (&self.column_indices, values_mut)
    }
}

/// Wrapper for a CsrMatrix that allows it to be interpreted as a parallel collection of rows.
pub struct ParallelCsrRowCollection<'a, T>(pub &'a mut CsrMatrix<T>);

#[derive(Copy)]
pub struct CsrParallelRowAccess<'a, T> {
    pattern: &'a SparsityPattern,
    values_ptr: *mut T,
}

impl<'a, T> Clone for CsrParallelRowAccess<'a, T> {
    fn clone(&self) -> Self {
        Self {
            pattern: self.pattern,
            values_ptr: self.values_ptr,
        }
    }
}

unsafe impl<'a, T: 'a + Sync> Sync for CsrParallelRowAccess<'a, T> {}
unsafe impl<'a, T: 'a + Send> Send for CsrParallelRowAccess<'a, T> {}

unsafe impl<'a, 'b, T: 'a + Sync + Send> ParallelIndexedAccess<'b> for CsrParallelRowAccess<'a, T>
where
    'a: 'b,
{
    type Record = ParCsrRow<'b, T>;
    type RecordMut = ParCsrRowMut<'b, T>;

    unsafe fn get_unchecked(&self, global_index: usize) -> Self::Record {
        let major_offsets = self.pattern.major_offsets();
        let row_begin = *major_offsets.get_unchecked(global_index);
        let row_end = *major_offsets.get_unchecked(global_index + 1);
        let column_indices = &self.pattern.minor_indices()[row_begin..row_end];
        let values_ptr = self.values_ptr.add(row_begin);
        let values = slice::from_raw_parts(values_ptr, column_indices.len());
        ParCsrRow { column_indices, values }
    }

    unsafe fn get_unchecked_mut(&self, global_index: usize) -> Self::RecordMut {
        let major_offsets = self.pattern.major_offsets();
        let row_begin = *major_offsets.get_unchecked(global_index);
        let row_end = *major_offsets.get_unchecked(global_index + 1);
        let column_indices = &self.pattern.minor_indices()[row_begin..row_end];
        let values_ptr = self.values_ptr.add(row_begin);
        ParCsrRowMut {
            column_indices,
            values: values_ptr,
        }
    }
}

unsafe impl<'a, T: 'a + Sync + Send> ParallelIndexedCollection<'a> for ParallelCsrRowCollection<'a, T> {
    type Access = CsrParallelRowAccess<'a, T>;

    unsafe fn create_access(&'a mut self) -> Self::Access {
        // TODO: Instead of storing a reference to the sparsity pattern we should probably
        // rather store the CSR data directly
        let values_ptr = self.0.values_mut().as_mut_ptr();
        let pattern = self.0.pattern();
        CsrParallelRowAccess { pattern, values_ptr }
    }

    fn len(&self) -> usize {
        self.0.nrows()
    }
}

// impl<T> CsrMatrix<T>
// where
//     T: Real,
// {
//     pub fn scale_rows<'a>(&mut self, diagonal_matrix: impl Into<DVectorSlice<'a, T>>) {
//         let diag = diagonal_matrix.into();
//         assert_eq!(diag.len(), self.nrows());
//         self.transform_values(|i, _, v| *v *= diag[i]);
//     }
//
//     pub fn scale_cols<'a>(&mut self, diagonal_matrix: impl Into<DVectorSlice<'a, T>>) {
//         let diag = diagonal_matrix.into();
//         assert_eq!(diag.len(), self.ncols());
//         self.transform_values(|_, j, v| *v *= diag[j]);
//     }
// }

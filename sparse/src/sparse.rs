//! Functionality for sparse linear algebra.
//!
//! Some of it is intended to be ported to `nalgebra` later.

use alga::general::{ClosedAdd, ClosedMul, ClosedSub};
use nalgebra::{
    DMatrix, DVector, DVectorSlice, DefaultAllocator, Dim, Dynamic, Matrix, RealField, Scalar,
    Vector,
};
use num::{One, Zero};

use itertools::{izip, Itertools};
use nalgebra::allocator::Allocator;
use nalgebra::base::storage::Storage;
use nalgebra::storage::StorageMut;
use paradis::{ParallelAccess, ParallelStorage};
use rayon::prelude::*;
use std::cmp::max;
use std::mem::swap;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::slice;
use std::sync::Arc;

/// A COO representation of a sparse matrix.
///
/// Does not support arithmetic, only used for assembling CSC matrices.
#[derive(Debug, Clone)]
pub struct CooMatrix<T> {
    nrows: usize,
    ncols: usize,
    i: Vec<usize>,
    j: Vec<usize>,
    v: Vec<T>,
}

impl<T> CooMatrix<T>
where
    T: Scalar,
{
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            i: Vec::new(),
            j: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        rows: Vec<usize>,
        cols: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        if rows.iter().any(|i| *i >= nrows) {
            panic!("Row indices contain index out of bounds.");
        }
        if cols.iter().any(|j| *j >= ncols) {
            panic!("Col indices contain index out of bounds.")
        }

        Self {
            nrows,
            ncols,
            i: rows,
            j: cols,
            v: values,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, i: usize, j: usize, v: T) {
        assert!(i < self.nrows);
        assert!(j < self.ncols);
        self.i.push(i);
        self.j.push(j);
        self.v.push(v);
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn nnz(&self) -> usize {
        self.v.len()
    }

    /// TODO: Must take combiner to decide how to combine duplicate elements
    pub fn build_dense(&self) -> DMatrix<T>
    where
        T: ClosedAdd + Zero,
    {
        let mut result = DMatrix::zeros(self.nrows, self.ncols);

        for (i, j, v) in izip!(&self.i, &self.j, &self.v) {
            result[(*i, *j)] += v.clone();
        }

        result
    }

    /// Convert the COO matrix to a CSR matrix.
    ///
    /// The combiner must be associative. TODO: More docs
    pub fn to_csr(&self, combiner: impl Fn(T, T) -> T) -> CsrMatrix<T>
    where
        T: Zero,
    {
        let combiner = &combiner;

        let (unsorted_ia, unsorted_ja, unsorted_v) = {
            let mut ia = vec![0usize; self.nrows() + 1];
            let mut ja = vec![0usize; self.nnz()];
            let mut v = vec![T::zero(); self.nnz()];
            coo_to_unsorted_csr(
                &mut ia,
                &mut ja,
                &mut v,
                self.nrows(),
                &self.i,
                &self.j,
                &self.v,
            );
            (ia, ja, v)
        };

        // At this point, CSR assembly is essentially complete. However, we must ensure
        // that column indices are sorted and without duplicates.
        let mut sorted_ia = Vec::new();
        let mut sorted_ja = Vec::new();
        let mut sorted_v = Vec::new();

        sorted_ia.push(0);

        // We need some temporary storage when working with each row. Since rows often have a
        // very small number of non-zero entries, we try to amortize allocations across
        // rows by reusing workspace vectors
        let mut idx_workspace = Vec::new();
        let mut perm_workspace = Vec::new();
        let mut values_workspace = Vec::new();

        for row in 0..self.nrows() {
            let begin = unsorted_ia[row];
            let end = unsorted_ia[row + 1];
            let count = end - begin;
            let range = begin..end;

            // Ensure that workspaces can hold enough data
            perm_workspace.resize(max(count, perm_workspace.len()), 0);
            idx_workspace.resize(max(count, idx_workspace.len()), 0);
            values_workspace.resize(max(count, values_workspace.len()), T::zero());
            sort_csr_row(
                &mut idx_workspace[..count],
                &mut values_workspace[..count],
                &unsorted_ja[range.clone()],
                &unsorted_v[range.clone()],
                &mut perm_workspace[..count],
            );

            let sorted_ja_current_len = sorted_ja.len();

            combine_duplicates(
                |idx| sorted_ja.push(idx),
                |val| sorted_v.push(val),
                &idx_workspace[..count],
                &values_workspace[..count],
                combiner,
            );

            let new_col_count = sorted_ja.len() - sorted_ja_current_len;
            sorted_ia.push(sorted_ia.last().unwrap() + new_col_count);
        }

        CsrMatrix::from_csr_data(self.nrows(), self.ncols(), sorted_ia, sorted_ja, sorted_v)
    }
}

impl<'a, T: Clone> AddAssign<&'a CooMatrix<T>> for CooMatrix<T> {
    fn add_assign(&mut self, rhs: &'a CooMatrix<T>) {
        assert_eq!(
            self.nrows, rhs.nrows,
            "Addition requires that matrices have the same number of rows."
        );
        assert_eq!(
            self.ncols, rhs.ncols,
            "Addition rquires that matrices have the same number of columns."
        );
        self.i.extend_from_slice(&rhs.i);
        self.j.extend_from_slice(&rhs.j);
        self.v.extend_from_slice(&rhs.v);
    }
}

/// Converts matrix data given in triplet format to unsorted CSR, retaining any duplicated
/// indices.
fn coo_to_unsorted_csr<T: Clone>(
    ia: &mut [usize],
    ja: &mut [usize],
    csr_values: &mut [T],
    num_rows: usize,
    rows: &[usize],
    cols: &[usize],
    coo_values: &[T],
) {
    assert_eq!(ia.len(), num_rows + 1);
    assert_eq!(ja.len(), csr_values.len());
    assert_eq!(csr_values.len(), rows.len());
    assert_eq!(rows.len(), cols.len());
    assert_eq!(cols.len(), coo_values.len());

    // Count the number of occurrences of each row
    for row_index in rows {
        ia[*row_index] += 1;
    }

    // Convert the counts to an offset
    let mut offset = 0;
    for i_offset in ia.iter_mut() {
        let count = *i_offset;
        *i_offset = offset;
        offset += count;
    }

    {
        // TODO: Instead of allocating a whole new vector storing the current counts,
        // I think it's possible to be a bit more clever by storing each count
        // in the last of the column indices for each row
        let mut current_counts = vec![0usize; num_rows + 1];
        for (i, j, value) in izip!(rows, cols, coo_values) {
            let current_offset = ia[*i] + current_counts[*i];
            ja[current_offset] = *j;
            csr_values[current_offset] = value.clone();
            current_counts[*i] += 1;
        }
    }
}

/// Sort the indices of the given CSR row.
///
/// The indices and values in `col_idx` and `col_values` are sorted according to the
/// column indices and stored in `col_idx_result` and `col_values` respectively.
///
/// All input slices are expected to be of the same length. The contents of mutable slices
/// can be arbitrary, as they are anyway overwritten.
fn sort_csr_row<T: Clone>(
    col_idx_result: &mut [usize],
    col_values_result: &mut [T],
    col_idx: &[usize],
    col_values: &[T],
    workspace: &mut [usize],
) {
    assert_eq!(col_idx_result.len(), col_values_result.len());
    assert_eq!(col_values_result.len(), col_idx.len());
    assert_eq!(col_idx.len(), col_values.len());
    assert_eq!(col_values.len(), workspace.len());

    let permutation = workspace;
    // Set permutation to identity
    for (i, p) in permutation.iter_mut().enumerate() {
        *p = i;
    }

    // Compute permutation needed to bring column indices into sorted order
    // Note: Using sort_unstable here avoids internal allocations, which is crucial since
    // each column might have a small number of elements
    permutation.sort_unstable_by_key(|idx| col_idx[*idx]);

    // TODO: Move this into `utils` or something?
    fn apply_permutation<T: Clone>(out_slice: &mut [T], in_slice: &[T], permutation: &[usize]) {
        assert_eq!(out_slice.len(), in_slice.len());
        assert_eq!(out_slice.len(), permutation.len());
        for (out_element, old_pos) in izip!(out_slice, permutation) {
            *out_element = in_slice[*old_pos].clone();
        }
    }

    apply_permutation(col_idx_result, col_idx, permutation);
    apply_permutation(col_values_result, col_values, permutation);
}

/// Given *sorted* indices and corresponding scalar values, combines duplicates with the given
/// associative combiner and calls the provided produce methods with combined indices and values.
fn combine_duplicates<T: Clone>(
    mut produce_idx: impl FnMut(usize),
    mut produce_value: impl FnMut(T),
    idx_array: &[usize],
    values: &[T],
    combiner: impl Fn(T, T) -> T,
) {
    assert_eq!(idx_array.len(), values.len());

    let mut i = 0;
    while i < idx_array.len() {
        let idx = idx_array[i];
        let mut combined_value = values[i].clone();
        let mut j = i + 1;
        while j < idx_array.len() && idx_array[j] == idx {
            let j_val = values[j].clone();
            combined_value = combiner(combined_value, j_val);
            j += 1;
        }
        produce_idx(idx);
        produce_value(combined_value);
        i = j;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsityPattern {
    major_offsets: Vec<usize>,
    minor_indices: Vec<usize>,
    minor_dim: usize,
}

impl SparsityPattern {
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            major_offsets: vec![0; major_dim + 1],
            minor_indices: vec![],
            minor_dim,
        }
    }

    pub fn from_offsets_and_indices(
        major_dim: usize,
        minor_dim: usize,
        major_offsets: Vec<usize>,
        minor_indices: Vec<usize>,
    ) -> Self {
        // TODO: Check validity of data
        assert_eq!(major_offsets.len(), major_dim + 1);
        assert_eq!(*major_offsets.last().unwrap(), minor_indices.len());
        Self {
            major_offsets,
            minor_indices,
            minor_dim,
        }
    }

    // TODO: Do we want to try to remove duplicates? Probably not...
    pub fn from_offsets_and_unsorted_indices(
        major_dim: usize,
        minor_dim: usize,
        major_offsets: Vec<usize>,
        mut minor_indices: Vec<usize>,
    ) -> Self {
        assert_eq!(major_offsets.len(), major_dim + 1);
        assert_eq!(*major_offsets.last().unwrap(), minor_indices.len());
        if major_offsets
            .iter()
            .tuple_windows()
            .any(|(prev, next)| prev > next)
        {
            panic!("Offsets must be non-decreasing.");
        }

        for (major_begin, major_end) in major_offsets.iter().tuple_windows() {
            let minor = &mut minor_indices[*major_begin..*major_end];
            minor.sort_unstable();
            if minor
                .iter()
                .tuple_windows()
                .any(|(prev, next)| prev >= next)
            {
                panic!("Minor indices contain duplicates");
            }
        }

        Self {
            major_offsets,
            minor_indices,
            minor_dim,
        }
    }

    pub fn major_offsets(&self) -> &[usize] {
        &self.major_offsets
    }

    pub fn minor_indices(&self) -> &[usize] {
        &self.minor_indices
    }

    pub fn major_dim(&self) -> usize {
        assert!(self.major_offsets.len() > 0);
        self.major_offsets.len() - 1
    }

    pub fn minor_dim(&self) -> usize {
        self.minor_dim
    }

    pub fn nnz(&self) -> usize {
        self.minor_indices.len()
    }

    /// Get the lane at the given index.
    ///
    /// TODO: Document that lane is a generalization of row/col?
    /// Is there better terminology?
    pub fn lane(&self, major_index: usize) -> Option<&[usize]> {
        let offset_begin = *self.major_offsets().get(major_index)?;
        let offset_end = *self.major_offsets.get(major_index + 1)?;
        Some(&self.minor_indices()[offset_begin..offset_end])
    }

    /// Appends another sparsity pattern to this one, in the sense that it is extended
    /// along its major dimension.
    ///
    /// Panics if `self` and `other` have different minor dimensions.
    pub fn append_pattern(&mut self, other: &SparsityPattern) {
        assert_eq!(self.minor_dim(), other.minor_dim());

        let offset_begin = *self.major_offsets.last().unwrap();
        let new_offsets_iter = other
            .major_offsets()
            .iter()
            .map(|offset| offset + offset_begin);

        self.major_offsets.pop();
        self.major_offsets.extend(new_offsets_iter);
        self.minor_indices.extend_from_slice(&other.minor_indices);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrMatrix<T> {
    // Rows are major, cols are minor in the sparsity pattern
    sparsity_pattern: Arc<SparsityPattern>,
    v: Vec<T>,
}

pub struct CsrRow<'a, T> {
    ncols: usize,
    column_indices: &'a [usize],
    values: &'a [T],
}

pub struct CsrRowMut<'a, T> {
    column_indices: &'a [usize],
    values: *mut T,
}

// TODO: Use macros to avoid code duplication in impls?
impl<'a, T> CsrRow<'a, T> {
    /// Number of non-zeros in this row.
    pub fn nnz(&self) -> usize {
        self.column_indices.len()
    }

    pub fn values(&self) -> &[T] {
        self.values
    }

    pub fn column_indices(&self) -> &[usize] {
        self.column_indices
    }

    pub fn value_at_local_index(&self, local_index: usize) -> &T {
        &self.values[local_index]
    }

    pub fn col_at_local_index(&self, local_index: usize) -> usize {
        self.column_indices[local_index]
    }
}

impl<'a, T: Clone + Zero> CsrRow<'a, T> {
    pub fn get(&self, global_index: usize) -> Option<T> {
        let local_index = self.column_indices.binary_search(&global_index);
        local_index
            .ok()
            .map(|local_index| self.values[local_index].clone())
            .or_else(|| {
                if global_index < self.ncols {
                    Some(T::zero())
                } else {
                    None
                }
            })
    }
}

impl<'a, T> CsrRowMut<'a, T> {
    /// Number of non-zeros in this row.
    pub fn nnz(&self) -> usize {
        self.column_indices.len()
    }

    pub fn values_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.values, self.column_indices.len()) }
    }

    // TODO: This API feels rather inelegant. Is there a better approach?
    pub fn columns_and_values_mut(&mut self) -> (&[usize], &mut [T]) {
        let values_mut =
            unsafe { slice::from_raw_parts_mut(self.values, self.column_indices.len()) };
        (&self.column_indices, values_mut)
    }

    pub fn value_at_local_index(&self, local_index: usize) -> &T {
        assert!(local_index < self.column_indices.len());
        unsafe { &*self.values.add(local_index) }
    }

    pub fn col_at_local_index(&self, local_index: usize) -> usize {
        self.column_indices[local_index]
    }
}

impl<T> CsrMatrix<T> {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            sparsity_pattern: Arc::new(SparsityPattern::new(nrows, ncols)),
            v: vec![],
        }
    }

    /// Computes a linear combination `A = sum(c_i * B_i)` of matrices over an iterator of items `(c_i, &B_i)`.
    pub fn new_linear_combination<'a>(
        mut matrix_iter: impl Iterator<Item = (T, &'a Self)>,
    ) -> Option<Self>
    where
        T: Clone + ClosedAdd + ClosedMul + Scalar + Zero + One + 'a,
    {
        // If the iterator yields at least one coefficient/matrix pair...
        matrix_iter.next().map(|(coeff, first_matrix)| {
            // ...compute the linear combination by...
            matrix_iter.fold(
                // scaling the first matrix
                first_matrix.clone() * coeff,
                // and summing the scaled remaining matrices.
                |mut matrix, (coeff, next_matrix)| {
                    matrix.add_assign_scaled(coeff, next_matrix);
                    matrix
                },
            )
        })
    }

    pub fn row<'a>(&'a self, index: usize) -> CsrRow<'a, T> {
        let row_begin = self.sparsity_pattern.major_offsets()[index];
        let row_end = self.sparsity_pattern.major_offsets()[index + 1];
        let column_indices = &self.sparsity_pattern.minor_indices()[row_begin..row_end];
        let values = &self.v[row_begin..row_end];

        CsrRow {
            ncols: self.ncols(),
            column_indices,
            values,
        }
    }

    pub fn row_mut<'a>(&'a mut self, index: usize) -> CsrRowMut<'a, T> {
        assert!(index < self.nrows());

        // Because of the lifetime of the borrow, we know that the `Arc` holding
        // our sparsity pattern will outlive the returned reference. Thus, its data also cannot
        // be mutated by other holders of the same `Arc`. Therefore we should be able to
        // extend the lifetime of the borrow of the column indices to the given lifetime.

        let row_begin = self.sparsity_pattern.major_offsets()[index];
        let row_end = self.sparsity_pattern.major_offsets()[index + 1];
        let column_indices = &self.sparsity_pattern.minor_indices()[row_begin..row_end];

        // Pointer to the first value
        let values_ptr = unsafe { self.v.as_mut_ptr().add(row_begin) };

        CsrRowMut {
            column_indices,
            values: values_ptr,
        }
    }

    pub fn nrows(&self) -> usize {
        self.sparsity_pattern.major_dim()
    }

    pub fn ncols(&self) -> usize {
        self.sparsity_pattern.minor_dim()
    }

    pub fn nnz(&self) -> usize {
        self.sparsity_pattern.nnz()
    }

    pub fn row_offsets(&self) -> &[usize] {
        self.sparsity_pattern.major_offsets()
    }

    pub fn column_indices(&self) -> &[usize] {
        self.sparsity_pattern.minor_indices()
    }

    pub fn values(&self) -> &[T] {
        &self.v
    }

    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.v
    }

    pub fn sparsity_pattern(&self) -> Arc<SparsityPattern> {
        Arc::clone(&self.sparsity_pattern)
    }

    // TODO: Write tests
    pub fn diag_iter<'a>(&'a self) -> impl 'a + Iterator<Item = T>
    where
        T: Zero + Clone,
    {
        let ia = self.row_offsets();
        let ja = self.column_indices();
        (0..self.nrows()).map(move |i| {
            let row_begin = ia[i];
            let row_end = ia[i + 1];
            let columns_in_row = &ja[row_begin..row_end];
            if let Ok(idx) = columns_in_row.binary_search(&i) {
                self.values()[row_begin + idx].clone()
            } else {
                T::zero()
            }
        })
    }

    pub fn from_csr_data(
        num_rows: usize,
        num_cols: usize,
        ia: Vec<usize>,
        ja: Vec<usize>,
        v: Vec<T>,
    ) -> Self {
        // TODO: Check validity of data
        assert_eq!(
            num_rows + 1,
            ia.len(),
            "length of ia must be equal to num_rows + 1"
        );
        assert_eq!(ja.len(), v.len());
        let pattern = SparsityPattern::from_offsets_and_indices(num_rows, num_cols, ia, ja);
        Self {
            sparsity_pattern: Arc::new(pattern),
            v,
        }
    }

    pub fn from_diagonal<'a>(diagonal: impl Into<DVectorSlice<'a, T>>) -> Self
    where
        T: Scalar,
    {
        let diagonal = diagonal.into();
        let vals = diagonal.iter().cloned().collect();
        let num_rows = diagonal.len();
        let ia = (0..(num_rows + 1)).collect();
        let ja = (0..num_rows).collect();
        Self::from_csr_data(num_rows, num_rows, ia, ja, vals)
    }

    pub fn from_pattern_and_values(pattern: Arc<SparsityPattern>, values: Vec<T>) -> Self {
        assert_eq!(pattern.nnz(), values.len());
        Self {
            sparsity_pattern: pattern,
            v: values,
        }
    }

    /// TODO: Rename to to_dense? (Also for CooMatrix)
    pub fn build_dense(&self) -> DMatrix<T>
    where
        T: Scalar + Zero,
    {
        let mut result = DMatrix::zeros(self.nrows(), self.ncols());

        for (i, j, v) in self.iter() {
            result[(i, j)] = v.clone();
        }

        result
    }

    /// Gives an iterator over non-zero elements in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        let ia = self.row_offsets();
        let ja = self.column_indices();
        (0..self.nrows()).flat_map(move |i| {
            let row_begin = ia[i];
            let row_end = ia[i + 1];
            izip!(&ja[row_begin..row_end], &self.v[row_begin..row_end])
                .map(move |(j, v)| (i, *j, v))
        })
    }

    /// TODO: If we provide an `iter_mut`, then this would be largely redundant. However,
    /// `iter_mut` likely requires a carefully implemented custom iterator to alleviate
    /// lifetime concerns
    pub fn transform_values(&mut self, f: impl Fn(usize, usize, &mut T)) {
        let ia = self.sparsity_pattern.major_offsets();
        let ja = self.sparsity_pattern.minor_indices();
        for i in 0..self.nrows() {
            let row_begin = ia[i];
            let row_end = ia[i + 1];
            for (j, v) in izip!(&ja[row_begin..row_end], &mut self.v[row_begin..row_end]) {
                f(i, *j, v)
            }
        }
    }

    /// Sets all values in the matrix to the specified value.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        for v_i in self.values_mut().iter_mut() {
            *v_i = value.clone();
        }
    }

    /// Scales all values in the matrix by the given factor.
    pub fn scale(&mut self, factor: T)
    where
        T: Clone + ClosedMul,
    {
        for v_i in self.values_mut().iter_mut() {
            *v_i *= factor.clone();
        }
    }

    /// Computes `self += a*x` where `x` is another matrix. Panics if the matrices are of different size.
    pub fn add_assign_scaled(&mut self, a: T, x: &Self)
    where
        T: Clone + ClosedAdd + ClosedMul,
    {
        assert_eq!(self.values_mut().len(), x.values().len());
        for (v_i, x_i) in self.values_mut().iter_mut().zip(x.values().iter()) {
            *v_i += a.clone() * x_i.clone();
        }
    }

    /// Returns a new matrix containing only the elements indicated by the supplied predicate.
    ///
    /// Note that the number of rows and columns in the output is unchanged.
    pub fn filter(&self, predicate: impl Fn(usize, usize, &T) -> bool) -> Self
    where
        T: Clone,
    {
        let ia = self.row_offsets();
        let ja = self.column_indices();

        let mut new_ia = Vec::new();
        let mut new_ja = Vec::new();
        let mut new_v = Vec::new();

        new_ia.push(0);
        for i in 0..self.nrows() {
            let row_begin = ia[i];
            let row_end = ia[i + 1];
            let current_ja_count = new_ja.len();
            for (j, v) in izip!(&ja[row_begin..row_end], &self.v[row_begin..row_end]) {
                if predicate(i, *j, v) {
                    new_ja.push(*j);
                    new_v.push(v.clone());
                }
            }

            let num_row_entries = new_ja.len() - current_ja_count;
            let current_offset = new_ia[i];
            new_ia.push(current_offset + num_row_entries);
        }

        assert_eq!(new_ia.len(), self.nrows() + 1);
        assert_eq!(new_ja.len(), new_v.len());
        assert_eq!(new_ja.len(), *new_ia.last().unwrap());

        // TODO: Circumvent data checks here by calling raw method instead,
        // once checks are in place
        Self::from_csr_data(self.nrows(), self.ncols(), new_ia, new_ja, new_v)
    }

    pub fn upper_triangular_part(&self) -> Self
    where
        T: Clone,
    {
        self.filter(|i, j, _| i <= j)
    }

    pub fn lower_triangular_part(&self) -> Self
    where
        T: Clone,
    {
        self.filter(|i, j, _| i >= j)
    }

    pub fn append_csr_rows(&mut self, other: &CsrMatrix<T>)
    where
        T: Clone,
    {
        Arc::make_mut(&mut self.sparsity_pattern).append_pattern(&other.sparsity_pattern());
        self.v.extend_from_slice(other.values());
    }

    pub fn to_csc(&self) -> CscMatrix<T>
    where
        T: Clone,
    {
        // TODO: Generalize this so that we can reuse the implementation for CSC->CSR conversion

        let mut col_counts = vec![0; self.ncols() + 1];
        for (_, j, _) in self.iter() {
            col_counts[j] += 1;
        }

        let mut col_offsets = col_counts;
        {
            let mut current_offset = 0;
            for offset in col_offsets.iter_mut() {
                let count = *offset;
                *offset = current_offset;
                current_offset += count;
            }
        }

        // Clone values to prevent a T::zero() bound
        let mut csc_values = self.v.clone();
        let mut row_indices = vec![0; self.nnz()];

        // Use the column offsets to keep track of how many values we've stored in each
        // column. This saves us from allocating one more vector
        let mut col_current_offsets = col_offsets;

        let csr_values = self.values();

        let column_indices = self.column_indices();
        for (r, (row_begin, row_end)) in self.row_offsets().iter().tuple_windows().enumerate() {
            let cols = &column_indices[*row_begin..*row_end];
            for (jj, c) in cols.iter().enumerate() {
                let csr_values_index = row_begin + jj;
                let csc_values_index = col_current_offsets[*c];
                csc_values[csc_values_index] = csr_values[csr_values_index].clone();
                row_indices[csc_values_index] = r;
                col_current_offsets[*c] += 1;
            }
        }

        // Restore the original column offsets array
        let mut offset = 0;
        for current_offset in col_current_offsets.iter_mut().take(self.ncols()) {
            swap(&mut offset, current_offset);
        }
        let col_offsets = col_current_offsets;

        let csc_pattern = SparsityPattern::from_offsets_and_indices(
            self.ncols(),
            self.nrows(),
            col_offsets,
            row_indices,
        );
        CscMatrix::from_pattern_and_values(Arc::new(csc_pattern), csc_values)
    }

    pub fn concat_diagonally(matrices: &[CsrMatrix<T>]) -> CsrMatrix<T>
    where
        T: Clone,
    {
        let mut num_rows = 0;
        let mut num_cols = 0;
        let mut nnz = 0;

        // This first pass over the matrices is cheap, since we don't access any of the data.
        // We use this to be able to pre-allocate enough capacity so that no further
        // reallocation will be necessary.
        for matrix in matrices {
            num_rows += matrix.nrows();
            num_cols += matrix.ncols();
            nnz += matrix.nnz();
        }

        let mut values = Vec::with_capacity(nnz);
        let mut column_indices = Vec::with_capacity(nnz);
        let mut row_offsets = Vec::with_capacity(num_rows + 1);

        let mut col_offset = 0;
        let mut nnz_offset = 0;
        for matrix in matrices {
            values.extend_from_slice(matrix.values());
            column_indices.extend(matrix.column_indices().iter().map(|i| *i + col_offset));
            row_offsets.extend(
                matrix
                    .row_offsets()
                    .iter()
                    .take(matrix.nrows())
                    .map(|offset| *offset + nnz_offset),
            );

            col_offset += matrix.ncols();
            nnz_offset += matrix.nnz();
        }

        row_offsets.push(nnz);

        Self {
            // TODO: Avoid validation of pattern for performance
            sparsity_pattern: Arc::new(SparsityPattern::from_offsets_and_indices(
                num_rows,
                num_cols,
                row_offsets,
                column_indices,
            )),
            v: values,
        }
    }
}

impl<T> CsrMatrix<T>
where
    T: Send + Sync,
{
    /// Computes a linear combination `A = sum(c_i * B_i)` of matrices over an iterator of items `(c_i, &B_i)`. Parallel version.
    pub fn new_linear_combination_par<'a>(
        mut matrix_iter: impl Iterator<Item = (T, &'a Self)>,
    ) -> Option<Self>
    where
        T: Clone + ClosedAdd + ClosedMul + Scalar + Zero + One + 'a,
    {
        // If the iterator yields at least one coefficient/matrix pair...
        matrix_iter.next().map(|(coeff, first_matrix)| {
            // ...compute the linear combination by...
            matrix_iter.fold(
                // scaling the first matrix
                first_matrix.clone() * coeff,
                // and summing the scaled remaining matrices.
                |mut matrix, (coeff, next_matrix)| {
                    matrix.add_assign_scaled_par(coeff, next_matrix);
                    matrix
                },
            )
        })
    }

    /// Add assigns a linear combination `self += sum(c_i * A_i)` of matrices over an iterator of items `(c_i, &A_i)`. Parallel version.
    pub fn add_assign_linear_combination_par<'a>(
        &mut self,
        matrix_iter: impl Iterator<Item = (T, &'a Self)>,
    ) where
        T: Clone + ClosedAdd + ClosedMul + Scalar + Zero + One + 'a,
    {
        for (coeff, matrix) in matrix_iter {
            self.add_assign_scaled_par(coeff, matrix);
        }
    }

    /// Sets all values in the matrix to the specified value. Parallel version.
    pub fn fill_par(&mut self, value: T)
    where
        T: Clone,
    {
        self.values_mut().par_iter_mut().for_each(|v_i| {
            *v_i = value.clone();
        });
    }

    /// Scales all values in the matrix by the given factor. Parallel version.
    pub fn scale_par(&mut self, factor: T)
    where
        T: Clone + ClosedMul,
    {
        self.values_mut().par_iter_mut().for_each(|v_i| {
            *v_i *= factor.clone();
        });
    }

    /// Computes `self += a*x` where `x` is another matrix. Panics if the matrices are of different size. Parallel version.
    pub fn add_assign_scaled_par(&mut self, a: T, x: &Self)
    where
        T: Clone + ClosedAdd + ClosedMul,
    {
        assert_eq!(self.values_mut().len(), x.values().len());

        self.values_mut()
            .par_iter_mut()
            .zip(x.values().par_iter())
            .for_each(|(v_i, x_i)| {
                *v_i += a.clone() * x_i.clone();
            });
    }
}

#[derive(Copy)]
pub struct CsrParallelAccess<'a, T> {
    sparsity_pattern: &'a SparsityPattern,
    values_ptr: *mut T,
}

impl<'a, T> Clone for CsrParallelAccess<'a, T> {
    fn clone(&self) -> Self {
        Self {
            sparsity_pattern: self.sparsity_pattern,
            values_ptr: self.values_ptr,
        }
    }
}

unsafe impl<'a, T: 'a + Sync> Sync for CsrParallelAccess<'a, T> {}
unsafe impl<'a, T: 'a + Send> Send for CsrParallelAccess<'a, T> {}

unsafe impl<'a, 'b, T: 'a + Sync + Send> ParallelAccess<'b> for CsrParallelAccess<'a, T>
where
    'a: 'b,
{
    type Record = CsrRow<'a, T>;
    type RecordMut = CsrRowMut<'b, T>;

    unsafe fn get_unchecked(&'b self, global_index: usize) -> Self::Record {
        let major_offsets = self.sparsity_pattern.major_offsets();
        let row_begin = *major_offsets.get_unchecked(global_index);
        let row_end = *major_offsets.get_unchecked(global_index + 1);
        let column_indices = &self.sparsity_pattern.minor_indices()[row_begin..row_end];
        let values_ptr = self.values_ptr.add(row_begin);
        let values = slice::from_raw_parts(values_ptr, column_indices.len());
        CsrRow {
            ncols: self.sparsity_pattern.minor_dim(),
            column_indices,
            values,
        }
    }

    unsafe fn get_unchecked_mut(&'b self, global_index: usize) -> Self::RecordMut {
        let major_offsets = self.sparsity_pattern.major_offsets();
        let row_begin = *major_offsets.get_unchecked(global_index);
        let row_end = *major_offsets.get_unchecked(global_index + 1);
        let column_indices = &self.sparsity_pattern.minor_indices()[row_begin..row_end];
        let values_ptr = self.values_ptr.add(row_begin);
        CsrRowMut {
            column_indices,
            values: values_ptr,
        }
    }
}

unsafe impl<'a, T: 'a + Sync + Send> ParallelStorage<'a> for CsrMatrix<T> {
    type Access = CsrParallelAccess<'a, T>;

    fn create_access(&'a mut self) -> Self::Access {
        let pattern = self.sparsity_pattern.as_ref();
        CsrParallelAccess {
            sparsity_pattern: pattern,
            values_ptr: self.v.as_mut_ptr(),
        }
    }

    fn len(&self) -> usize {
        self.nrows()
    }
}

impl<T> CsrMatrix<T>
where
    T: RealField,
{
    pub fn scale_rows<'a>(&mut self, diagonal_matrix: impl Into<DVectorSlice<'a, T>>) {
        let diag = diagonal_matrix.into();
        assert_eq!(diag.len(), self.nrows());
        self.transform_values(|i, _, v| *v *= diag[i]);
    }

    pub fn scale_cols<'a>(&mut self, diagonal_matrix: impl Into<DVectorSlice<'a, T>>) {
        let diag = diagonal_matrix.into();
        assert_eq!(diag.len(), self.ncols());
        self.transform_values(|_, j, v| *v *= diag[j]);
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CooMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        let mut coo = CooMatrix::new(matrix.nrows(), matrix.ncols());
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let val = matrix[(i, j)].clone();
                if val != T::zero() {
                    coo.push(i, j, matrix[(i, j)].clone());
                }
            }
        }
        coo
    }
}

impl<'a, T, R, C, S> From<&'a Matrix<T, R, C, S>> for CsrMatrix<T>
where
    T: Scalar + Zero,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
{
    fn from(matrix: &'a Matrix<T, R, C, S>) -> Self {
        // TODO: Construct directly as CSR matrix to avoid overhead of conversion from COO
        CooMatrix::from(matrix).to_csr(|_, _| {
            panic!("There cannot be duplicates when constructed from a dense matrix")
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscMatrix<T> {
    // Cols are major, rows are minor in the sparsity pattern
    sparsity_pattern: Arc<SparsityPattern>,
    v: Vec<T>,
}

impl<T> CscMatrix<T> {
    pub fn nrows(&self) -> usize {
        self.sparsity_pattern.minor_dim()
    }

    pub fn ncols(&self) -> usize {
        self.sparsity_pattern.major_dim()
    }

    pub fn nnz(&self) -> usize {
        self.sparsity_pattern.nnz()
    }

    pub fn column_offsets(&self) -> &[usize] {
        self.sparsity_pattern.major_offsets()
    }

    pub fn row_indices(&self) -> &[usize] {
        self.sparsity_pattern.minor_indices()
    }

    pub fn values(&self) -> &[T] {
        &self.v
    }

    pub fn sparsity_pattern(&self) -> Arc<SparsityPattern> {
        Arc::clone(&self.sparsity_pattern)
    }

    pub fn from_pattern_and_values(pattern: Arc<SparsityPattern>, values: Vec<T>) -> Self {
        assert_eq!(pattern.nnz(), values.len());
        Self {
            sparsity_pattern: pattern,
            v: values,
        }
    }

    pub fn from_csc_data(
        num_rows: usize,
        num_cols: usize,
        column_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        // TODO: Check validity of data
        assert_eq!(
            num_cols + 1,
            column_offsets.len(),
            "length of column_offsets must be equal to num_rows + 1"
        );
        let pattern = SparsityPattern::from_offsets_and_indices(
            num_cols,
            num_rows,
            column_offsets,
            row_indices,
        );
        Self {
            sparsity_pattern: Arc::new(pattern),
            v: values,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        let indices = self.row_indices();
        self.column_offsets()
            .iter()
            .cloned()
            .tuple_windows()
            .enumerate()
            .flat_map(move |(col, (col_begin, col_end))| {
                izip!(&indices[col_begin..col_end], &self.v[col_begin..col_end])
                    .map(move |(row, v)| (*row, col, v))
            })
    }

    pub fn to_dense(&self) -> DMatrix<T>
    where
        T: Scalar + Zero,
        DefaultAllocator: Allocator<T, Dynamic, Dynamic>,
    {
        let mut result = DMatrix::zeros(self.nrows(), self.ncols());

        for (i, j, v) in self.iter() {
            result[(i, j)] = v.clone();
        }

        result
    }
}

/// Compute y <- beta * y + alpha * A * x,
/// where A is the CSR matrix
/// TODO: Generalize to any kind of vector, not just DVector
/// Matrix<T, Dynamic, U1, S>
pub fn spmv_csr<T, SA, SB>(
    beta: T,
    y: &mut Vector<T, Dynamic, SA>,
    alpha: T,
    csr: &CsrMatrix<T>,
    x: &Vector<T, Dynamic, SB>,
) where
    T: Scalar + Zero + ClosedMul + ClosedAdd,
    SA: StorageMut<T, Dynamic>,
    SB: Storage<T, Dynamic>,
{
    assert_eq!(y.len(), csr.nrows());
    assert_eq!(csr.ncols(), x.len());
    let y_data = y.data.as_mut_slice();

    for (i, y_i) in y_data.iter_mut().enumerate() {
        let row_begin = csr.row_offsets()[i];
        let row_end = csr.row_offsets()[i + 1];
        let col_indices = &csr.column_indices()[row_begin..row_end];
        let row_values = &csr.values()[row_begin..row_end];

        *y_i = beta.clone() * y_i.clone();

        // Compute axpy between row i in matrix and vector x
        let mut a_i_dot_x = T::zero();
        for (a_ij, j) in izip!(row_values, col_indices) {
            a_i_dot_x += a_ij.clone() * x[*j].clone();
        }

        *y_i += alpha.clone() * a_i_dot_x;
    }
}

/// Compute the CSR sparsity pattern for the product A B.
pub fn spmm_csr_pattern(
    pattern_a: &SparsityPattern,
    pattern_b: &SparsityPattern,
) -> SparsityPattern {
    let mut major_offsets = Vec::with_capacity(pattern_a.major_dim());
    let mut minor_indices = Vec::new();

    assert_eq!(pattern_a.minor_dim(), pattern_b.major_dim());

    let mut current_offset = 0;
    major_offsets.push(current_offset);

    // Build one lane at a time
    for i in 0..pattern_a.major_dim() {
        let a_lane = pattern_a.lane(i).unwrap();
        // Find non-zeros common to both a_i (lane i in A) and b_k (lane k in B)
        for k in a_lane {
            let b_lane = pattern_b.lane(*k).unwrap();

            let mut local_offset = 0;
            for b_kj in b_lane {
                let c_i_col_indices = &minor_indices[(current_offset + local_offset)..];
                // TODO: An exponential search would probably be faster than the binary search here
                let c_i_local_index = c_i_col_indices.binary_search(b_kj);

                // Upset local offset so that we binary search on a progressively smaller range
                local_offset = match c_i_local_index {
                    // If the index was found, then we need not do anything
                    Ok(c_i_local_index) => local_offset + c_i_local_index,
                    // On the other hand, if it was not, insert into the appropriate position
                    Err(c_i_local_index) => {
                        let global_index = current_offset + local_offset + c_i_local_index;
                        // TODO: This might be expensive
                        minor_indices.insert(global_index, *b_kj);
                        local_offset + c_i_local_index
                    }
                }
            }
        }

        let num_row_entries = minor_indices.len() - current_offset;
        current_offset += num_row_entries;
        major_offsets.push(current_offset);
    }

    // TODO: Avoid potential consistency checks
    SparsityPattern::from_offsets_and_indices(
        pattern_a.major_dim(),
        pattern_b.minor_dim(),
        major_offsets,
        minor_indices,
    )
}

#[derive(Debug, Clone)]
pub struct IncompatibleSparsityPattern;

/// Compute C <- beta * C + alpha * A * B.
///
/// TODO: At the moment, this expects the output matrix c to have the same sparsity pattern as A * B.
/// It is not clear at present what kind of API is best suited to deal with the various situations
/// one might have
pub fn spmm_csr<T>(
    beta: T,
    c: &mut CsrMatrix<T>,
    alpha: T,
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> Result<(), IncompatibleSparsityPattern>
where
    T: Scalar + ClosedAdd + ClosedMul,
{
    assert_eq!(c.nrows(), a.nrows());
    assert_eq!(c.ncols(), b.ncols());

    for i in 0..c.nrows() {
        let mut c_row = c.row_mut(i);
        for c_ik in c_row.values_mut() {
            *c_ik *= beta.inlined_clone();
        }

        let (c_columns, c_values) = c_row.columns_and_values_mut();

        let a_row = a.row(i);
        for (k, a_ik) in a_row.column_indices().iter().zip(a_row.values()) {
            let b_row = b.row(*k);

            let mut local_offset = 0;
            for (j, b_kj) in b_row.column_indices().iter().zip(b_row.values()) {
                // Make sure to reduce the space we binary search in as
                // we move through indices in b
                let c_columns = &c_columns[local_offset..];
                // TODO: An exponential search would presumably be (much) faster
                let index = c_columns
                    .binary_search(j)
                    .map_err(|_| IncompatibleSparsityPattern)?;
                local_offset += index;
                c_values[local_offset] +=
                    alpha.inlined_clone() * a_ik.inlined_clone() * b_kj.inlined_clone();
            }
        }
    }

    Ok(())
}

/// Compute C <- alpha * A * B
pub fn spmm_csr_owned<T>(alpha: T, a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero,
{
    let pattern = spmm_csr_pattern(&a.sparsity_pattern(), &b.sparsity_pattern());
    let nnz = pattern.nnz();
    let mut result = CsrMatrix::from_pattern_and_values(Arc::new(pattern), vec![T::zero(); nnz]);
    spmm_csr(T::zero(), &mut result, alpha, a, b).expect("Sparsity pattern is always compatible");
    result
}

/// Compute C <- op(beta * C, alpha * A), where A and C are CSR matrices with *the same sparsity
/// pattern*.
pub fn csr_comp_bin_op_in_place_same_pattern<T, Op: Fn(T, T) -> T>(
    op: Op,
    beta: T,
    c: &mut CsrMatrix<T>,
    alpha: T,
    a: &CsrMatrix<T>,
) where
    T: Scalar + Zero + ClosedMul,
{
    assert_eq!(c.nrows(), a.nrows());
    assert_eq!(c.ncols(), a.ncols());
    assert_eq!(c.sparsity_pattern(), a.sparsity_pattern());
    assert_eq!(c.values().len(), a.values().len());

    for (c_val, a_val) in c.values_mut().iter_mut().zip(a.values()) {
        let new_value = op(beta.clone() * c_val.clone(), alpha.clone() * a_val.clone());
        *c_val = new_value;
    }
}

// TODO: Generalize to vectors of any dimension
impl<'a, T> Mul<&'a DVector<T>> for &'a CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    type Output = DVector<T>;

    fn mul(self, rhs: &'a DVector<T>) -> Self::Output {
        let mut y = DVector::zeros(self.nrows());
        spmv_csr(T::zero(), &mut y, T::one(), self, rhs);
        y
    }
}

impl<'a, T> Mul<&'a CsrMatrix<T>> for &'a CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    type Output = CsrMatrix<T>;

    fn mul(self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        spmm_csr_owned(T::one(), self, rhs)
    }
}

#[inline(always)]
fn csr_comp_bin_op_in_place<T, Op: Fn(T, T) -> T>(
    op: Op,
    beta: T,
    lhs: &mut CsrMatrix<T>,
    alpha: T,
    rhs: &CsrMatrix<T>,
) where
    T: Scalar + ClosedMul + Zero + One,
{
    if lhs.sparsity_pattern() == rhs.sparsity_pattern() {
        csr_comp_bin_op_in_place_same_pattern(op, beta, lhs, alpha, rhs);
    } else {
        // TODO: Implement this!
        unimplemented!()
    }
}

impl<'a, T> Add<&'a CsrMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    type Output = CsrMatrix<T>;

    fn add(mut self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        csr_comp_bin_op_in_place(Add::add, T::one(), &mut self, T::one(), rhs);
        self
    }
}

impl<'a, T> Add<&'a CsrMatrix<T>> for &'a CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    type Output = CsrMatrix<T>;

    fn add(self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        let mut result = self.clone();
        csr_comp_bin_op_in_place(Add::add, T::one(), &mut result, T::one(), rhs);
        result
    }
}

impl<T> Add<CsrMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + ClosedAdd + ClosedMul + Zero + One,
{
    type Output = CsrMatrix<T>;

    fn add(mut self, rhs: CsrMatrix<T>) -> Self::Output {
        csr_comp_bin_op_in_place(Add::add, T::one(), &mut self, T::one(), &rhs);
        self
    }
}

impl<'a, T> Sub<&'a CsrMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + ClosedSub + ClosedMul + Zero + One,
{
    type Output = CsrMatrix<T>;

    fn sub(mut self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        csr_comp_bin_op_in_place(Sub::sub, T::one(), &mut self, T::one(), rhs);
        self
    }
}

impl<'a, T> Sub<&'a CsrMatrix<T>> for &'a CsrMatrix<T>
where
    T: Scalar + ClosedSub + ClosedMul + Zero + One,
{
    type Output = CsrMatrix<T>;

    fn sub(self, rhs: &'a CsrMatrix<T>) -> Self::Output {
        let mut result = self.clone();
        csr_comp_bin_op_in_place(Sub::sub, T::one(), &mut result, T::one(), rhs);
        result
    }
}

impl<T> Sub<CsrMatrix<T>> for CsrMatrix<T>
where
    T: Scalar + ClosedSub + ClosedMul + Zero + One,
{
    type Output = CsrMatrix<T>;

    fn sub(mut self, rhs: CsrMatrix<T>) -> Self::Output {
        csr_comp_bin_op_in_place(Sub::sub, T::one(), &mut self, T::one(), &rhs);
        self
    }
}

impl<T> Mul<T> for CsrMatrix<T>
where
    T: Scalar + ClosedMul + Zero + One,
{
    type Output = Self;

    fn mul(mut self, scalar: T) -> Self::Output {
        for val in self.values_mut() {
            *val *= scalar.clone();
        }
        self
    }
}

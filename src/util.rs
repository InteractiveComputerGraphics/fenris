use crate::Real;
use itertools::izip;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::constraint::{DimEq, ShapeConstraint};
use nalgebra::storage::{Storage, StorageMut};
use nalgebra::{
    DMatrixView, DVector, DVectorView, DefaultAllocator, Dim, DimDiff, DimMin, DimMul, DimName, DimProd, DimSub,
    Matrix, Matrix3, MatrixView, MatrixViewMut, OMatrix, OPoint, OVector, Quaternion, Scalar, ViewStorage,
    ViewStorageMut, SquareMatrix, UnitQuaternion, Vector, Vector3, U1,
};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use num::Zero;
use numeric_literals::replace_float_literals;
use std::any::TypeId;
use std::error::Error;
use std::fmt::Display;
use std::fmt::LowerExp;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::transmute;
use std::path::Path;

pub use fenris_geometry::util::compute_orthonormal_vectors_3d;
pub use fenris_nested_vec::*;

use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::assembly::global::CsrParAssembler;
use crate::connectivity::Connectivity;
use crate::mesh::Mesh;
use crate::nalgebra::Dyn;
use crate::SmallDim;

/// Clones the upper triangle entries into the lower triangle entries.
///
/// The primary use case for this is to construct a full symmetric matrix from a symmetric
/// matrix represented only by its upper triangular entries.
pub(crate) fn clone_upper_to_lower<T, R, C, S>(matrix: &mut Matrix<T, R, C, S>)
where
    T: Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<T, R, C>,
{
    for j in 0..matrix.ncols() {
        for i in (j + 1)..matrix.nrows() {
            matrix[(i, j)] = matrix[(j, i)].clone();
        }
    }
}

/// Given a matrix, returns a matrix slice reshaped to the requested shape.
// TODO: Implement ReshapeableStorage for slices in `nalgebra`
pub(crate) fn reshape_to_slice<T, R, C, S, R2, C2>(
    matrix: &Matrix<T, R, C, S>,
    shape: (R2, C2),
) -> MatrixView<T, R2, C2, U1, R2>
where
    T: Scalar,
    R: DimMul<C>,
    C: Dim,
    R2: DimMul<C2>,
    C2: Dim,
    S: Storage<T, R, C>,
    ShapeConstraint: DimEq<DimProd<R, C>, DimProd<R2, C2>>,
{
    let (r2, c2) = shape;
    assert_eq!(
        matrix.nrows() * matrix.ncols(),
        r2.value() * c2.value(),
        "Cannot reshape with different number of elements"
    );
    let strides = (U1::name(), r2);
    let storage = unsafe { ViewStorage::from_raw_parts(matrix.data.ptr(), shape, strides) };
    Matrix::from_data(storage)
}

/// Creates a column-major slice from the given matrix.
///
/// Panics if the matrix does not have column-major storage.
pub fn coerce_col_major_slice<T, R, C, S, RSlice, CSlice>(
    matrix: &Matrix<T, R, C, S>,
    slice_rows: RSlice,
    slice_cols: CSlice,
) -> MatrixView<T, RSlice, CSlice, U1, RSlice>
where
    T: Scalar,
    R: Dim,
    RSlice: Dim,
    C: Dim,
    CSlice: Dim,
    S: Storage<T, R, C>,
    ShapeConstraint: DimEq<R, RSlice> + DimEq<C, CSlice>,
{
    assert_eq!(slice_rows.value(), matrix.nrows());
    assert_eq!(slice_cols.value(), matrix.ncols());
    let (rstride, cstride) = matrix.strides();
    assert!(
        rstride == 1 && cstride == matrix.nrows(),
        "Matrix must have column-major storage."
    );

    unsafe {
        let data = ViewStorage::new_with_strides_unchecked(
            &matrix.data,
            (0, 0),
            (slice_rows, slice_cols),
            (U1::name(), slice_rows),
        );
        Matrix::from_data_statically_unchecked(data)
    }
}

/// An SVD-like decomposition in which the orthogonal matrices `U` and `V` are rotation matrices.
///
/// Given a matrix `A`, this method returns factors `U`, `S` and `V` such that
/// `A = U S V^T`, with `U, V` orthogonal and `det(U) = det(V) = 1` and `S` a diagonal matrix
/// whose entries are represented by a vector.
///
/// Note that unlike the standard SVD, `S` may contain negative entries, and so they do not
/// generally coincide with singular values. However, it holds that `S(i)^2 == sigma_i^2`, where
/// `sigma_i` is the `i`th singular value of `A`.
///
/// Returns a tuple `(U, S, V^T)`.
pub fn rotation_svd<T, D>(matrix: &OMatrix<T, D, D>) -> (OMatrix<T, D, D>, OVector<T, D>, OMatrix<T, D, D>)
where
    T: Real,
    D: DimName + DimMin<D, Output = D> + DimSub<U1>,
    DefaultAllocator: Allocator<T, D>
        + Allocator<T, D, D>
        + Allocator<T, <D as DimSub<U1>>::Output>
        + Allocator<(usize, usize), D>
        + Allocator<(T, usize), D>,
{
    let minus_one = T::from_f64(-1.0).unwrap();
    let mut svd = matrix.clone().svd(true, true);
    let min_val_idx = svd.singular_values.imin();

    let mut u = svd.u.unwrap();
    if u.determinant() < T::zero() {
        let mut u_col = u.column_mut(min_val_idx);
        u_col *= minus_one;
        svd.singular_values[min_val_idx] *= minus_one;
    }

    let mut v_t = svd.v_t.unwrap();
    if v_t.determinant() < T::zero() {
        let mut v_t_row = v_t.row_mut(min_val_idx);
        v_t_row *= minus_one;
        svd.singular_values[min_val_idx] *= minus_one;
    }

    (u, svd.singular_values, v_t)
}

/// "Analytic polar decomposition"
///
/// Translated to Rust from <https://github.com/InteractiveComputerGraphics/FastCorotatedFEM/blob/351b007b6bb6e8d97f457766e9ecf9b2bced7079/FastCorotFEM.cpp#L413>
///
/// ```
/// use fenris::util::apd;
/// use nalgebra::{Matrix3, UnitQuaternion, Quaternion, Vector3};
///
/// let eps: f64 = 1e-12;
/// let guess = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.2);
/// assert!((apd::<f64>(&Matrix3::identity(), &guess, 100, eps).as_ref() - &Quaternion::identity()).norm() < 1.0e1 * eps);
/// assert!((apd::<f64>(&Matrix3::identity(), &guess, 100, eps).as_ref() - guess.as_ref()).norm() > 1.0e2 * eps);
/// ```
///
#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn apd<T: Real>(
    deformation_grad: &Matrix3<T>,
    initial_guess: &UnitQuaternion<T>,
    max_iter: usize,
    tol: T,
) -> UnitQuaternion<T> {
    let F = deformation_grad;
    let mut q: UnitQuaternion<T> = initial_guess.clone();

    let tol_squared = tol * tol;
    // TODO: Fix unwrap
    let mut res = T::max_value().unwrap();
    let mut iter = 0;
    while res > tol_squared && iter < max_iter {
        let R = q.to_rotation_matrix();
        let B = R.transpose() * F;

        let B0 = B.column(0);
        let B1 = B.column(1);
        let B2 = B.column(2);

        let gradient = Vector3::new(B2[1] - B1[2], B0[2] - B2[0], B1[0] - B0[1]);

        // compute Hessian, use the fact that it is symmetric
        let h00 = B1[1] + B2[2];
        let h11 = B0[0] + B2[2];
        let h22 = B0[0] + B1[1];
        let h01 = (B1[0] + B0[1]) * 0.5;
        let h02 = (B2[0] + B0[2]) * 0.5;
        let h12 = (B2[1] + B1[2]) * 0.5;

        let detH =
            -(h02 * h02 * h11) + (h01 * h02 * h12) * 2.0 - (h00 * h12 * h12) - (h01 * h01 * h22) + (h00 * h11 * h22);
        let factor = detH.recip() * (-0.25);

        let mut omega = Vector3::zeros();

        // compute symmetric inverse
        omega[0] = (h11 * h22 - h12 * h12) * gradient[0]
            + (h02 * h12 - h01 * h22) * gradient[1]
            + (h01 * h12 - h02 * h11) * gradient[2];
        omega[1] = (h02 * h12 - h01 * h22) * gradient[0]
            + (h00 * h22 - h02 * h02) * gradient[1]
            + (h01 * h02 - h00 * h12) * gradient[2];
        omega[2] = (h01 * h12 - h02 * h11) * gradient[0]
            + (h01 * h02 - h00 * h12) * gradient[1]
            + (h00 * h11 - h01 * h01) * gradient[2];
        omega *= factor;

        // if det(H) = 0 use gradient descent, never happened in our tests, could also be removed
        if detH.abs() < 1.0e-9 {
            omega = -gradient;
        }

        // instead of clamping just use gradient descent. also works fine and does not require the norm
        let useGD = omega.dot(&gradient) > T::zero();
        if useGD {
            omega = &gradient * (-0.125);
        }

        let l_omega2 = omega.norm_squared();

        let w = (1.0 - l_omega2) / (1.0 + l_omega2);
        let vec = omega * (2.0 / (1.0 + l_omega2));

        // no normalization needed because the Cayley map returs a unit quaternion
        q = q * UnitQuaternion::new_unchecked(Quaternion::from_parts(w, vec));

        iter += 1;
        res = l_omega2;
    }

    q
}

pub fn diag_left_mul<T, D1, D2, S>(diag: &Vector<T, D1, S>, matrix: &OMatrix<T, D1, D2>) -> OMatrix<T, D1, D2>
where
    T: Real,
    D1: DimName,
    D2: DimName,
    S: Storage<T, D1>,
    DefaultAllocator: Allocator<T, D1, D2>,
{
    // TODO: This is inefficient
    let mut result = matrix.clone();
    for (i, mut row) in result.row_iter_mut().enumerate() {
        row *= diag[i];
    }
    result
}

/// Creates a mutable column-major slice from the given matrix.
///
/// Panics if the matrix does not have column-major storage.
pub fn coerce_col_major_slice_mut<T, R, C, S, RSlice, CSlice>(
    matrix: &mut Matrix<T, R, C, S>,
    slice_rows: RSlice,
    slice_cols: CSlice,
) -> MatrixViewMut<T, RSlice, CSlice, U1, RSlice>
where
    T: Scalar,
    R: Dim,
    RSlice: Dim,
    C: Dim,
    CSlice: Dim,
    S: StorageMut<T, R, C>,
    ShapeConstraint: DimEq<R, RSlice> + DimEq<C, CSlice>,
{
    assert_eq!(slice_rows.value(), matrix.nrows());
    assert_eq!(slice_cols.value(), matrix.ncols());
    let (rstride, cstride) = matrix.strides();
    assert!(
        rstride == 1 && cstride == matrix.nrows(),
        "Matrix must have column-major storage."
    );

    unsafe {
        let data = ViewStorageMut::new_with_strides_unchecked(
            &mut matrix.data,
            (0, 0),
            (slice_rows, slice_cols),
            (U1::name(), slice_rows),
        );
        Matrix::from_data_statically_unchecked(data)
    }
}

pub fn try_transmute_slice<T: 'static, U: 'static>(e: &[T]) -> Option<&[U]> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        Some(unsafe { transmute(e) })
    } else {
        None
    }
}

pub fn try_transmute_ref<T: 'static, U: 'static>(e: &T) -> Option<&U> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        Some(unsafe { transmute(e) })
    } else {
        None
    }
}

pub fn try_transmute_ref_mut<T: 'static, U: 'static>(e: &mut T) -> Option<&mut U> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        Some(unsafe { transmute(e) })
    } else {
        None
    }
}

pub fn cross_product_matrix<T: Real>(x: &Vector3<T>) -> Matrix3<T> {
    Matrix3::new(T::zero(), -x[2], x[1], x[2], T::zero(), -x[0], -x[1], x[0], T::zero())
}

pub fn dump_matrix_to_file<'a, T: Scalar + Display>(
    path: impl AsRef<Path>,
    matrix: impl Into<DMatrixView<'a, T>>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);

    let matrix = matrix.into();
    for i in 0..matrix.nrows() {
        write!(writer, "{}", matrix[(i, 0)])?;
        for j in 1..matrix.ncols() {
            write!(writer, " {}", matrix[(i, j)])?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;

    Ok(())
}

/// Dumps matrices corresponding to node-node connectivity and element-node connectivity
/// to the Matrix Market sparse storage format.
pub fn dump_mesh_connectivity_matrices<T, D, C>(
    node_path: impl AsRef<Path>,
    element_path: impl AsRef<Path>,
    mesh: &Mesh<T, D, C>,
) -> Result<(), Box<dyn Error>>
where
    T: Scalar + LowerExp,
    D: DimName,
    C: Sync + Connectivity,
    DefaultAllocator: Allocator<T, D>,
    Mesh<T, D, C>: Sync,
{
    let pattern = CsrParAssembler::<usize>::default().assemble_pattern(mesh);
    let nnz = pattern.nnz();
    let node_matrix = CsrMatrix::try_from_pattern_and_values(pattern, vec![1.0f64; nnz])
        .expect("CSR data must be valid by definition");

    dump_csr_matrix_to_mm_file(node_path.as_ref(), &node_matrix).map_err(|err| err as Box<dyn Error>)?;

    // Create a rectangular matrix with element index on the rows and
    // node indices as columns
    let mut element_node_matrix = CooMatrix::new(mesh.connectivity().len(), mesh.vertices().len());
    for (i, conn) in mesh.connectivity().iter().enumerate() {
        for &j in conn.vertex_indices() {
            element_node_matrix.push(i, j, 1.0f64);
        }
    }

    dump_csr_matrix_to_mm_file(element_path.as_ref(), &CsrMatrix::from(&element_node_matrix))
        .map_err(|err| err as Box<dyn Error>)?;
    Ok(())
}

/// Dumps a CSR matrix to a matrix market file.
///
/// TODO: Support writing integers etc. Probably need a custom trait for this
/// for writing the correct header, as well as for formatting numbers correctly
/// (scientific notation for floating point, integer for integers)
pub fn dump_csr_matrix_to_mm_file<T: Scalar + LowerExp>(
    path: impl AsRef<Path>,
    matrix: &CsrMatrix<T>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "%%MatrixMarket matrix coordinate real general")?;

    // Write dimensions
    writeln!(writer, "{} {} {}", matrix.nrows(), matrix.ncols(), matrix.nnz())?;

    for (i, j, v) in matrix.triplet_iter() {
        // Indices have to be stored as 1-based
        writeln!(writer, "{} {} {:.e}", i + 1, j + 1, v)?;
    }
    writer.flush()?;

    Ok(())
}

pub fn min_eigenvalue_symmetric<T, D, S>(matrix: &SquareMatrix<T, D, S>) -> T
where
    T: Real,
    D: Dim + DimSub<U1>,
    S: Storage<T, D, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>> + Allocator<T, D> + Allocator<T, DimDiff<D, U1>>,
{
    use std::cmp::Ordering;
    matrix
        .symmetric_eigenvalues()
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .unwrap()
        .to_owned()
}

/// Extracts D-dimensional nodal values from a global vector using a node index list
pub fn extract_by_node_index<T, D>(u: &[T], node_indices: &[usize]) -> DVector<T>
where
    T: Scalar + Copy + Zero,
    D: DimName,
{
    let u = DVectorView::from(u);
    let mut extracted = DVector::zeros(D::dim() * node_indices.len());
    for (i_local, &i_global) in node_indices.iter().enumerate() {
        let ui = u.rows_generic(D::dim() * i_global, D::name());
        extracted
            .rows_generic_mut(D::dim() * i_local, D::name())
            .copy_from(&ui);
    }
    extracted
}

pub fn min_max_symmetric_eigenvalues<T, D, S>(matrix: &SquareMatrix<T, D, S>) -> (T, T)
where
    T: Real,
    D: Dim + DimSub<U1>,
    S: Storage<T, D, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>> + Allocator<T, D> + Allocator<T, DimDiff<D, U1>>,
{
    use std::cmp::Ordering;
    matrix
        .symmetric_eigenvalues()
        .iter()
        .minmax_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .into_option()
        .map(|(a, b)| (*a, *b))
        .unwrap()
}

pub fn condition_number_symmetric<T, D, S>(matrix: &SquareMatrix<T, D, S>) -> T
where
    T: Real,
    D: Dim + DimSub<U1>,
    S: Storage<T, D, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, DimDiff<D, U1>> + Allocator<T, D> + Allocator<T, DimDiff<D, U1>>,
{
    use std::cmp::Ordering;
    let (min, max) = matrix
        .symmetric_eigenvalues()
        .into_iter()
        .cloned()
        .minmax_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
        .into_option()
        .expect("Currently don't support empty matrices");

    max.abs() / min.abs()
}

/*
pub fn condition_number_csr<T>(matrix: &CsrMatrix<T>) -> T
where
    T: Real + mkl_corrode::SupportedScalar,
{
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "Matrix must be square for condition number computation."
    );
    assert!(
        matrix.nrows() > 0,
        "Cannot compute condition number for empty matrix."
    );
    use mkl_corrode::mkl_sys::MKL_INT;
    use mkl_corrode::sparse::{CsrMatrixHandle, MatrixDescription};
    use std::convert::TryFrom;

    let row_offsets: Vec<_> = matrix
        .row_offsets()
        .iter()
        .cloned()
        .map(|idx| MKL_INT::try_from(idx).unwrap())
        .collect();
    let columns: Vec<_> = matrix
        .column_indices()
        .iter()
        .cloned()
        .map(|idx| MKL_INT::try_from(idx).unwrap())
        .collect();

    // TODO: This isn't 100% safe at the moment, because we don't properly enforce
    // the necessary invariants in `Csr` (but we should, it's just a lack of time)
    // TODO: Error handling
    let mkl_csr = unsafe {
        CsrMatrixHandle::from_raw_csr_data(
            matrix.nrows(),
            matrix.ncols(),
            &row_offsets[..matrix.nrows()],
            &row_offsets[1..],
            &columns,
            matrix.values(),
        )
    }
    .unwrap();

    let description = MatrixDescription::default();

    // TODO: Error handling
    let eigenresult_largest = k_largest_eigenvalues(&mkl_csr, &description, 1).unwrap();
    let eigenresult_smallest = k_smallest_eigenvalues(&mkl_csr, &description, 1).unwrap();

    let eig_max = eigenresult_largest.eigenvalues().first().unwrap();
    let eig_min = eigenresult_smallest.eigenvalues().first().unwrap();

    eig_max.abs() / eig_min.abs()
}
*/

#[cfg(feature = "proptest")]
pub mod proptest {
    use nalgebra::{DMatrix, Point2, Scalar, Vector2};
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{Reason, TestRunner};

    pub fn point2_f64_strategy() -> impl Strategy<Value = Point2<f64>> {
        vector2_f64_strategy().prop_map(|vector| Point2::from(vector))
    }

    pub fn vector2_f64_strategy() -> impl Strategy<Value = Vector2<f64>> {
        let xrange = prop_oneof![-3.0..3.0, -100.0..100.0];
        let yrange = xrange.clone();
        (xrange, yrange).prop_map(|(x, y)| Vector2::new(x, y))
    }

    /// Simple helper function to produce square shapes for use with matrix strategies.
    pub fn square_shape<S>(dim: S) -> impl Strategy<Value = (usize, usize)>
    where
        S: Strategy<Value = usize>,
    {
        dim.prop_map(|dim| (dim, dim))
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct DMatrixStrategy<ElementStrategy, ShapeStrategy> {
        element_strategy: ElementStrategy,
        shape_strategy: ShapeStrategy,
    }

    impl DMatrixStrategy<(), ()> {
        pub fn new() -> Self {
            Self {
                element_strategy: (),
                shape_strategy: (),
            }
        }
    }

    impl<ElementStrategy, ShapeStrategy> DMatrixStrategy<ElementStrategy, ShapeStrategy> {
        pub fn with_elements<E>(self, element_strategy: E) -> DMatrixStrategy<E, ShapeStrategy>
        where
            E: Strategy,
        {
            DMatrixStrategy {
                element_strategy,
                shape_strategy: self.shape_strategy,
            }
        }

        pub fn with_shapes<S>(self, shape_strategy: S) -> DMatrixStrategy<ElementStrategy, S>
        where
            S: Strategy<Value = (usize, usize)>,
        {
            DMatrixStrategy {
                element_strategy: self.element_strategy,
                shape_strategy,
            }
        }
    }

    impl<ElementStrategy, ShapeStrategy> Strategy for DMatrixStrategy<ElementStrategy, ShapeStrategy>
    where
        ElementStrategy: Clone + 'static + Strategy,
        ElementStrategy::Value: Scalar,
        ShapeStrategy: Clone + 'static + Strategy<Value = (usize, usize)>,
    {
        type Tree = Box<dyn ValueTree<Value = Self::Value>>;
        type Value = DMatrix<ElementStrategy::Value>;

        fn new_tree(&self, runner: &mut TestRunner) -> Result<Self::Tree, Reason> {
            let element_strategy = self.element_strategy.clone();
            self.shape_strategy
                .clone()
                .prop_flat_map(move |(nrows, ncols)| {
                    let num_elements = nrows * ncols;
                    vec(element_strategy.clone(), num_elements)
                        .prop_map(move |elements| DMatrix::from_row_slice(nrows, ncols, &elements))
                })
                .boxed()
                .new_tree(runner)
        }
    }

    #[cfg(test)]
    mod tests {
        use proptest::prelude::*;

        use super::DMatrixStrategy;

        proptest! {
            #[test]
            fn dmatrix_strategy_respects_strategies(
                matrix in DMatrixStrategy::new()
                            .with_shapes((Just(5), 2usize..=3))
                            .with_elements(0i32 ..= 5))
            {
                prop_assert_eq!(matrix.nrows(), 5);
                prop_assert!(matrix.ncols() >= 2);
                prop_assert!(matrix.ncols() <= 3);
                prop_assert!(matrix.iter().cloned().all(|x| x >= 0 && x <= 5));
            }
        }
    }
}

/// Computes the interpolation $u_h$ given basis function values and interpolation weights.
///
/// More precisely, computes
/// <div>$$
/// u_h = \sum_I u_I \, N_I \quad \in \mathbb{R}^s
/// $$</div>
/// given interpolation weights $u_I \in \mathbb{R}^s$ and nodal basis function values $N_I \in \mathbb{R}$.
/// The interpolation weights and basis function values are stored in the block vectors
/// <div>$$
/// \dvec u = \begin{pmatrix}
///  u_1 \\
///  u_2 \\
///  \vdots \\
/// \end{pmatrix}
/// \qquad
/// \dvec N = \begin{pmatrix}
///  N_1 \\
///  N_2 \\
///  \vdots \\
/// \end{pmatrix}.
/// $$</div>
///
/// # Panics
///
/// Panics if `u` does not have `SolutionDim` entries for every entry in `basis`.
///
/// TODO: This is not directly tested at the moment
// TODO: Move elsewhere
pub fn compute_interpolation<'a, 'b, T, SolutionDim>(
    u: impl Into<DVectorView<'a, T>>,
    basis: impl Into<DVectorView<'b, T>>,
) -> OVector<T, SolutionDim>
where
    T: Real,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, SolutionDim>,
{
    compute_interpolation_(u.into(), basis.into())
}

fn compute_interpolation_<T, SolutionDim>(u: DVectorView<T>, basis: DVectorView<T>) -> OVector<T, SolutionDim>
where
    T: Real,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, SolutionDim>,
{
    let s = SolutionDim::dim();
    let n = basis.len();
    assert_eq!(u.len(), s * n);
    assert_eq!(s, SolutionDim::dim());

    // Reshape u as the matrix
    //  [u1 u2 .. un ]
    // for vectors u1, u2, ... associated with each basis function value
    let u = reshape_to_slice(&u, (SolutionDim::name(), Dyn(n)));
    u * basis
}

/// Computes the gradient $\nabla u_h$ of the interpolation $u_h$ given basis function gradients
/// and interpolation weights.
///
/// More precisely, computes
/// <div>$$
/// \nabla u_h = \sum_I \nabla N_I \otimes u_I \quad \in \mathbb{R}^{d \times s}
/// $$</div>
/// given interpolation weights $u_I \in \mathbb{R}^s$ and nodal basis gradients $\nabla N_I \in \mathbb{R}^d$.
/// The interpolation weights and basis gradients are stored in the block vectors
/// <div>$$
/// \dvec u = \begin{pmatrix}
///  u_1 \\
///  u_2 \\
///  \vdots \\
/// \end{pmatrix}
/// \qquad
/// \dvec G = \begin{pmatrix}
///  \nabla N_1 \\
///  \nabla N_2 \\
///  \vdots \\
/// \end{pmatrix}.
/// $$</div>
///
/// This function can be used to compute either coordinates with respect to physical coordinates or reference
/// coordinates, depending on what is provided as basis gradients.
///
/// # Panics
///
/// Panics if the dimensions of `u` and/or the basis gradients are inconsistent.
///
/// TODO: This is not directly tested at the moment
pub fn compute_interpolation_gradient<'a, T, SolutionDim, GeometryDim>(
    u: impl Into<DVectorView<'a, T>>,
    basis_gradients: impl Into<DVectorView<'a, T>>,
) -> OMatrix<T, GeometryDim, SolutionDim>
where
    T: Real,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    compute_interpolation_gradient_(u.into(), basis_gradients.into())
}

fn compute_interpolation_gradient_<T, SolutionDim, GeometryDim>(
    u: DVectorView<T>,
    basis_gradients: DVectorView<T>,
) -> OMatrix<T, GeometryDim, SolutionDim>
where
    T: Real,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    let d = GeometryDim::dim();
    let s = SolutionDim::dim();
    let n = basis_gradients.len() / d;
    assert_eq!(basis_gradients.len(), d * n);
    assert_eq!(u.len(), s * n);

    // Reshape u as the matrix
    //  [u1 u2 .. un ]
    // for vectors u1, u2, ... associated with each basis function value
    let u = reshape_to_slice(&u, (SolutionDim::name(), Dyn(n)));

    // Reshape gradients g as the matrix
    //  [g1 g2 ... gn]
    let g = reshape_to_slice(&basis_gradients, (GeometryDim::name(), Dyn(n)));

    let mut u_grad = OMatrix::<T, GeometryDim, SolutionDim>::zeros();
    for (u_i, g_i) in izip!(u.column_iter(), g.column_iter()) {
        // Outer product addition
        //  u_grad += g_I * u_I^T
        u_grad.ger(T::one(), &g_i, &u_i, T::one());
    }
    u_grad
}

/// Evaluate a function at a set of points and concatenate the results into a single global
/// vector.
///
/// Specifically, given $N$ points and the map $f : \mathbb{R}^d \rightarrow \mathbb{R}^s$,
/// returns the vector
///
/// <div>
/// $$
/// \begin{pmatrix}
/// f(x_1) \\
/// f(x_2) \\
/// \vdots \\
/// f(x_N)
/// \end{pmatrix}.
/// $$
/// </div>
///
/// # Example
/// ```rust
/// use fenris::util::global_vector_from_point_fn;
/// use nalgebra::{point, Point2, vector};
/// use matrixcompare::assert_matrix_eq;
///
/// let x1 = point![0.0, 0.0];
/// let x2 = point![1.0, 0.0];
/// let f = |p: &Point2<f64>| vector![2.0 * p.y, 3.0 * p.x];
/// let u = global_vector_from_point_fn(&[x1, x2], f);
///
/// assert_matrix_eq!(u.fixed_rows::<2>(0), f(&x1));
/// assert_matrix_eq!(u.fixed_rows::<2>(2), f(&x2));
/// assert_eq!(u.len(), 4);
/// ```
pub fn global_vector_from_point_fn<T, D, S, F>(points: &[OPoint<T, D>], mut f: F) -> DVector<T>
where
    T: Scalar,
    D: SmallDim,
    S: SmallDim,
    F: FnMut(&OPoint<T, D>) -> OVector<T, S>,
    DefaultAllocator: DimAllocator<T, D> + DimAllocator<T, S>,
{
    let mut result = Vec::with_capacity(points.len() * S::dim());
    for point in points {
        result.extend_from_slice(f(point).as_slice());
    }
    DVector::from_vec(result)
}

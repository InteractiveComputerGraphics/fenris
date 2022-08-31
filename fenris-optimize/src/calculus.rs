use fenris_traits::Real;
use nalgebra::{
    DMatrix, DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut, Dim, DimName, Dynamic, Scalar, Vector, U1,
};

use nalgebra::base::storage::{Storage, StorageMut};
use numeric_literals::replace_float_literals;
use std::error::Error;

pub trait VectorFunction<T>
where
    T: Scalar,
{
    fn dimension(&self) -> usize;
    fn eval_into(&mut self, f: &mut DVectorSliceMut<T>, x: &DVectorSlice<T>);
}

impl<T, X> VectorFunction<T> for &mut X
where
    T: Scalar,
    X: VectorFunction<T>,
{
    fn dimension(&self) -> usize {
        X::dimension(self)
    }

    fn eval_into(&mut self, f: &mut DVectorSliceMut<T>, x: &DVectorSlice<T>) {
        X::eval_into(self, f, x)
    }
}

pub trait DifferentiableVectorFunction<T>: VectorFunction<T>
where
    T: Scalar,
{
    fn solve_jacobian_system(
        &mut self,
        sol: &mut DVectorSliceMut<T>,
        x: &DVectorSlice<T>,
        rhs: &DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>>;
}

impl<T, X> DifferentiableVectorFunction<T> for &mut X
where
    T: Scalar,
    X: DifferentiableVectorFunction<T>,
{
    fn solve_jacobian_system(
        &mut self,
        sol: &mut DVectorSliceMut<T>,
        x: &DVectorSlice<T>,
        rhs: &DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        X::solve_jacobian_system(self, sol, x, rhs)
    }
}

#[derive(Debug, Clone)]
pub struct VectorFunctionBuilder {
    dimension: usize,
}

#[derive(Debug, Clone)]
pub struct ConcreteVectorFunction<F, J> {
    dimension: usize,
    function: F,
    jacobian_solver: J,
}

impl VectorFunctionBuilder {
    pub fn with_dimension(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn with_function<F, T>(self, function: F) -> ConcreteVectorFunction<F, ()>
    where
        T: Scalar,
        F: FnMut(&mut DVectorSliceMut<T>, &DVectorSlice<T>),
    {
        ConcreteVectorFunction {
            dimension: self.dimension,
            function,
            jacobian_solver: (),
        }
    }
}

impl<F> ConcreteVectorFunction<F, ()> {
    pub fn with_jacobian_solver<J, T>(self, jacobian_solver: J) -> ConcreteVectorFunction<F, J>
    where
        T: Scalar,
        J: FnMut(&mut DVectorSliceMut<T>, &DVectorSlice<T>, &DVectorSlice<T>) -> Result<(), Box<dyn Error>>,
    {
        ConcreteVectorFunction {
            dimension: self.dimension,
            function: self.function,
            jacobian_solver,
        }
    }
}

impl<F, J, T> VectorFunction<T> for ConcreteVectorFunction<F, J>
where
    T: Scalar,
    F: FnMut(&mut DVectorSliceMut<T>, &DVectorSlice<T>),
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn eval_into(&mut self, f: &mut DVectorSliceMut<T>, x: &DVectorSlice<T>) {
        let func = &mut self.function;
        func(f, x)
    }
}

impl<F, J, T> DifferentiableVectorFunction<T> for ConcreteVectorFunction<F, J>
where
    T: Scalar,
    F: FnMut(&mut DVectorSliceMut<T>, &DVectorSlice<T>),
    J: FnMut(&mut DVectorSliceMut<T>, &DVectorSlice<T>, &DVectorSlice<T>) -> Result<(), Box<dyn Error>>,
{
    fn solve_jacobian_system(
        &mut self,
        sol: &mut DVectorSliceMut<T>,
        x: &DVectorSlice<T>,
        rhs: &DVectorSlice<T>,
    ) -> Result<(), Box<dyn Error>> {
        let j = &mut self.jacobian_solver;
        j(sol, x, rhs)
    }
}

// TODO: Move somewhere else? Ideally contribute as From<_> for DVectorSlice<T> in `nalgebra`
fn as_vector_slice<T, R, S>(vector: &Vector<T, R, S>) -> DVectorSlice<T>
where
    T: Scalar,
    S: Storage<T, R, U1, RStride = U1, CStride = Dynamic>,
    R: Dim,
{
    vector.generic_slice((0, 0), (Dynamic::new(vector.nrows()), U1::name()))
}

// TODO: Move somewhere else? Ideally contribute as From<_> for DVectorSliceMut<T> in `nalgebra`
fn as_vector_slice_mut<T, R, S>(vector: &mut Vector<T, R, S>) -> DVectorSliceMut<T>
where
    T: Scalar,
    S: StorageMut<T, R, U1, RStride = U1, CStride = Dynamic>,
    R: Dim,
{
    vector.generic_slice_mut((0, 0), (Dynamic::new(vector.nrows()), U1::name()))
}

/// Approximates the Jacobian of a vector function evaluated at `x`, using
/// central finite differences with resolution `h`.
#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
pub fn approximate_jacobian<T>(mut f: impl VectorFunction<T>, x: &DVector<T>, h: &T) -> DMatrix<T>
where
    T: Real,
{
    let out_dim = f.dimension();
    let in_dim = x.len();

    let mut result = DMatrix::zeros(out_dim, in_dim);

    // Define quantities x+ and x- as follows:
    //  x+ := x + h e_j
    //  x- := x - h e_j
    // where e_j is the jth basis vector consisting of all zeros except for the j-th element,
    // which is 1.
    let mut x_plus = x.clone();
    let mut x_minus = x.clone();

    // f+ := f(x+)
    // f- := f(x-)
    let mut f_plus = DVector::zeros(out_dim);
    let mut f_minus = DVector::zeros(out_dim);

    // Use finite differences to compute a numerical approximation of the Jacobian
    for j in 0..in_dim {
        // TODO: Can optimize this a little by simple resetting the element at the end of the iteration
        x_plus.copy_from(x);
        x_plus[j] += *h;
        x_minus.copy_from(x);
        x_minus[j] -= *h;

        f.eval_into(&mut as_vector_slice_mut(&mut f_plus), &as_vector_slice(&x_plus));
        f.eval_into(&mut as_vector_slice_mut(&mut f_minus), &as_vector_slice(&x_minus));

        // result[.., j] := (f+ - f-) / 2h
        let mut column_j = result.column_mut(j);
        column_j += &f_plus;
        column_j -= &f_minus;
        column_j /= 2.0 * *h;
    }

    result
}

/// Approximates the derivative of the function `f: R^n -> R` with finite differences.
///
/// The parameter `h` determines the step size of the finite difference approximation.
///
/// The vector `x` is mutable in order to contain intermediate computations, but upon returning,
/// its content remains unchanged.
pub fn approximate_gradient_fd<'a, T>(
    f: impl FnMut(DVectorSlice<T>) -> T,
    x: impl Into<DVectorSliceMut<'a, T>>,
    h: T,
) -> DVector<T>
where
    T: Real,
{
    let x = x.into();
    let mut df = DVector::zeros(x.len());
    approximate_gradient_fd_into_(DVectorSliceMut::from(&mut df), f, x, h);
    df
}

/// Approximates the derivative of the function `f: R^n -> R` with finite differences.
///
/// Analogous to [`approximate_gradient_fd`], but stores the result in the provided
/// output vector.
///
/// The vector `x` is mutable in order to contain intermediate results, but upon returning,
/// its content remains unchanged.
pub fn approximate_gradient_fd_into<'a, T>(
    mut df: DVectorSliceMut<T>,
    f: impl FnMut(DVectorSlice<T>) -> T,
    x: impl Into<DVectorSliceMut<'a, T>>,
    h: T,
) where
    T: Real,
{
    approximate_gradient_fd_into_(DVectorSliceMut::from(&mut df), f, x.into(), h);
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn approximate_gradient_fd_into_<T>(
    mut df: DVectorSliceMut<T>,
    mut f: impl FnMut(DVectorSlice<T>) -> T,
    mut x: DVectorSliceMut<T>,
    h: T,
) where
    T: Real,
{
    let n = x.len();
    for i in 0..n {
        let x_i = x[i];
        x[i] = x_i + h;
        let f_plus = f(DVectorSlice::from(&x));
        x[i] = x_i - h;
        let f_minus = f(DVectorSlice::from(&x));
        let df_i = (f_plus - f_minus) / (2.0 * h);
        df[i] = df_i;
        x[i] = x_i;
    }
}

/// Approximates the Jacobian of the function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$
/// with finite differences.
///
/// The Jacobian matrix is the $m \times n$ matrix whose entries are given by
/// $$ J_{ij} := \pd{f_i}{x_j}.$$
///
/// The parameter `h` determines the step size of the finite difference approximation.
pub fn approximate_jacobian_fd<'a, T>(
    m: usize,
    f: impl FnMut(DVectorSlice<T>, DVectorSliceMut<T>),
    x: impl Into<DVectorSliceMut<'a, T>>,
    h: T,
) -> DMatrix<T>
where
    T: Real,
{
    let x = x.into();
    let n = x.len();
    let mut jacobian = DMatrix::zeros(m, n);
    approximate_jacobian_fd_into_(DMatrixSliceMut::from(&mut jacobian), f, x, h);
    jacobian
}

/// Approximates the Jacobian of the function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$
/// with finite differences.
///
/// Same as [`approximate_jacobian_fd`], but stores the result in the provided output matrix.
pub fn approximate_jacobian_fd_into<'a, T>(
    jacobian: impl Into<DMatrixSliceMut<'a, T>>,
    f: impl FnMut(DVectorSlice<T>, DVectorSliceMut<T>),
    x: impl Into<DVectorSliceMut<'a, T>>,
    h: T,
) where
    T: Real,
{
    approximate_jacobian_fd_into_(jacobian.into(), f, x.into(), h);
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn approximate_jacobian_fd_into_<T>(
    mut j: DMatrixSliceMut<T>,
    mut f: impl FnMut(DVectorSlice<T>, DVectorSliceMut<T>),
    mut x: DVectorSliceMut<T>,
    h: T,
) where
    T: Real,
{
    let m = j.nrows();
    let n = x.len();
    assert_eq!(n, j.ncols());

    // Buffers to hold f(x + e_i h) and f(x - e_i h)
    let mut f_plus = DVector::zeros(m);
    let mut f_minus = DVector::zeros(m);

    // Jacobian has m x n rows
    // Build column by column
    for i in 0..n {
        // df_dxi ~ (f(x + h e_i) - f(x - h e_i)) / (2 h)
        let xi = x[i];
        x[i] = xi + h;
        f(DVectorSlice::from(&x), DVectorSliceMut::from(&mut f_plus));
        x[i] = xi - h;
        f(DVectorSlice::from(&x), DVectorSliceMut::from(&mut f_minus));
        x[i] = xi;

        let mut df_dxi = j.column_mut(i);
        df_dxi.copy_from(&f_plus);
        df_dxi -= &f_minus;
        df_dxi /= 2.0 * h;
    }
}

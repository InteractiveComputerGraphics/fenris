use crate::calculus::{DifferentiableVectorFunction, VectorFunction};
use fenris_traits::Real;
use itertools::iterate;
use log::debug;
use nalgebra::{DVector, DVectorView, DVectorViewMut, Scalar};
use numeric_literals::replace_float_literals;
use std::error::Error;
use std::fmt;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct NewtonResult<T>
where
    T: Scalar,
{
    pub solution: DVector<T>,
    pub iterations: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NewtonSettings<T> {
    pub max_iterations: Option<usize>,
    pub tolerance: T,
}

#[derive(Debug)]
pub enum NewtonError {
    /// The procedure failed because the maximum number of iterations was reached.
    MaximumIterationsReached(usize),
    /// The procedure failed because solving the Jacobian system failed.
    JacobianError(Box<dyn Error>),
    // The line search failed to produce a valid step direction.
    LineSearchError(Box<dyn Error>),
}

impl Display for NewtonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            &NewtonError::MaximumIterationsReached(maxit) => {
                write!(f, "Failed to converge within maximum number of iterations ({}).", maxit)
            }
            &NewtonError::JacobianError(ref err) => {
                write!(f, "Failed to solve Jacobian system. Error: {}", err)
            }
            &NewtonError::LineSearchError(ref err) => {
                write!(f, "Line search failed to produce valid step direction. Error: {}", err)
            }
        }
    }
}

impl Error for NewtonError {}

/// Attempts to solve the non-linear equation F(u) = 0.
///
/// No heap allocation is performed. The solution is said to have converged if
/// ```|F(u)|_2 <= tolerance```.
///
/// If successful, returns the number of iterations performed.
#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn newton<'a, T, F>(
    function: F,
    x: impl Into<DVectorViewMut<'a, T>>,
    f: impl Into<DVectorViewMut<'a, T>>,
    dx: impl Into<DVectorViewMut<'a, T>>,
    settings: NewtonSettings<T>,
) -> Result<usize, NewtonError>
where
    T: Real,
    F: DifferentiableVectorFunction<T>,
{
    newton_line_search(function, x, f, dx, settings, &mut NoLineSearch {})
}

/// Same as `newton`, but allows specifying a line search.
#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn newton_line_search<'a, T, F>(
    mut function: F,
    x: impl Into<DVectorViewMut<'a, T>>,
    f: impl Into<DVectorViewMut<'a, T>>,
    dx: impl Into<DVectorViewMut<'a, T>>,
    settings: NewtonSettings<T>,
    line_search: &mut impl LineSearch<T, F>,
) -> Result<usize, NewtonError>
where
    T: Real,
    F: DifferentiableVectorFunction<T>,
{
    let mut x = x.into();
    let mut f = f.into();
    let mut minus_dx = dx.into();

    assert_eq!(x.nrows(), f.nrows());
    assert_eq!(minus_dx.nrows(), f.nrows());

    function.eval_into(&mut f, &DVectorView::from(&x));

    let mut iter = 0;

    while f.norm() > settings.tolerance {
        if settings
            .max_iterations
            .map(|max_iter| iter == max_iter)
            .unwrap_or(false)
        {
            return Err(NewtonError::MaximumIterationsReached(iter));
        }

        // Solve the system J dx = -f   <=>   J (-dx) = f
        let j_result = function.solve_jacobian_system(&mut minus_dx, &DVectorView::from(&x), &DVectorView::from(&f));
        if let Err(err) = j_result {
            return Err(NewtonError::JacobianError(err));
        }

        // Flip sign to make it consistent with line search
        minus_dx *= -1.0;
        let dx = &minus_dx;

        let step_length = line_search
            .step(
                &mut function,
                DVectorViewMut::from(&mut f),
                DVectorViewMut::from(&mut x),
                DVectorView::from(dx),
            )
            .map_err(|err| NewtonError::LineSearchError(err))?;
        debug!("Newton step length at iter {}: {}", iter, step_length);
        iter += 1;
    }

    Ok(iter)
}

pub trait LineSearch<T: Scalar, F: VectorFunction<T>> {
    fn step(
        &mut self,
        function: &mut F,
        f: DVectorViewMut<T>,
        x: DVectorViewMut<T>,
        direction: DVectorView<T>,
    ) -> Result<T, Box<dyn Error>>;
}

/// Trivial implementation of line search. Equivalent to a single, full Newton step.
#[derive(Clone, Debug)]
pub struct NoLineSearch;

impl<T, F> LineSearch<T, F> for NoLineSearch
where
    T: Real,
    F: VectorFunction<T>,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn step(
        &mut self,
        function: &mut F,
        mut f: DVectorViewMut<T>,
        mut x: DVectorViewMut<T>,
        direction: DVectorView<T>,
    ) -> Result<T, Box<dyn Error>> {
        let p = direction;
        x.axpy(T::one(), &p, T::one());
        function.eval_into(&mut f, &DVectorView::from(&x));
        Ok(T::one())
    }
}

/// Standard backtracking line search using the Armijo condition.
///
/// See Jorge & Nocedal (2006), Numerical Optimization, Chapter 3.1.
/// #[derive(Clone, Debug)]
pub struct BacktrackingLineSearch;

impl<T, F> LineSearch<T, F> for BacktrackingLineSearch
where
    T: Real,
    F: VectorFunction<T>,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn step(
        &mut self,
        function: &mut F,
        mut f: DVectorViewMut<T>,
        mut x: DVectorViewMut<T>,
        direction: DVectorView<T>,
    ) -> Result<T, Box<dyn Error>> {
        // We seek to solve
        //  F(x) = 0
        // by minimizing
        //  g(x) = (1/2) || F(x) ||^2
        // We have that
        //  grad g = (grad F) * F,
        // and the sufficient decrease condition becomes
        //  g(x_k + alpha * p_k) <= g(x_k) + c * alpha * (grad g)^T * p_k
        //                       ~= g(x_k) - c * alpha * g(x_k)
        //                        = (1 - c * alpha) * g(x_k)
        // where p_k is the step direction, c is a parameter in (0, 1)
        // and we have assumed that
        //  grad F^T p_k ~= - F(x_k)
        // (which it would satisfy if p_k is the exact solution of the Newton step equation).

        // TODO: Is this an OK parameter? Should anyway make it configurable
        let c = 1e-4;
        let alpha_min = 1e-6;

        let p = direction;
        let g_initial = 0.5 * f.magnitude_squared();

        // Start out with some alphas that don't decrease too quickly, then
        // start decreasing them much faster if the first few iterations don't let us
        // take a step.
        let initial_alphas = [0.0, 1.0, 0.75, 0.5];
        let mut alpha_iter = initial_alphas
            .iter()
            .copied()
            .chain(iterate(0.25, |alpha_i| 0.25 * *alpha_i));

        let mut alpha_prev = alpha_iter.next().unwrap();
        let mut alpha = alpha_iter.next().unwrap();

        loop {
            let delta_alpha = alpha - alpha_prev;

            // We have that x^{k + 1} = x^0 + alpha^k * p,
            // where x^{k+1} is the value of x after taking the step based on the current alpha
            // parameter. It is straightforward to show that this implies that
            //  x^{k + 1} = x^k + (alpha^k - alpha^{k - 1}) * p,
            // which is far more amenable to computation
            x.axpy(delta_alpha, &p, T::one());
            function.eval_into(&mut f, &DVectorView::from(&x));

            let g = 0.5 * f.magnitude_squared();
            if g <= (1.0 - c * alpha) * g_initial {
                break;
            } else if alpha < alpha_min {
                return Err(Box::from(format!(
                    "Failed to produce valid step direction.\
                    Alpha {} is smaller than minimum allowed alpha {}.",
                    alpha, alpha_min
                )));
            } else {
                alpha_prev = alpha;
                alpha = alpha_iter.next().unwrap();
            }
        }

        Ok(alpha)
    }
}

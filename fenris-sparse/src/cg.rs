use core::fmt;
use nalgebra::base::constraint::AreMultipliable;
use nalgebra::constraint::{DimEq, ShapeConstraint};
use nalgebra::storage::Storage;
use nalgebra::{
    ClosedAdd, ClosedMul, DVector, DVectorSlice, DVectorSliceMut, Dim, Dynamic, Matrix, RealField, Scalar, U1,
};
use nalgebra_sparse::ops::serial::spmm_csr_dense;
use nalgebra_sparse::ops::Op;
use nalgebra_sparse::CsrMatrix;
use num::{One, Zero};
use std::error::Error;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub trait LinearOperator<T: Scalar> {
    fn apply(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>>;
}

impl<'a, T, A> LinearOperator<T> for &'a A
where
    T: Scalar,
    A: ?Sized + LinearOperator<T>,
{
    fn apply(&self, y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        <A as LinearOperator<T>>::apply(self, y, x)
    }
}

impl<T, R, C, S> LinearOperator<T> for Matrix<T, R, C, S>
where
    T: Scalar + One + Zero + ClosedMul + ClosedAdd,
    R: Dim,
    C: Dim,
    S: Storage<T, R, C>,
    ShapeConstraint: DimEq<Dynamic, R> + DimEq<C, Dynamic> + AreMultipliable<R, C, Dynamic, U1>,
{
    fn apply(&self, mut y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        y.gemv(T::one(), self, &x, T::zero());
        Ok(())
    }
}

impl<T> LinearOperator<T> for CsrMatrix<T>
where
    T: Scalar + Zero + One + ClosedMul + ClosedAdd,
{
    fn apply(&self, mut y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        spmm_csr_dense(T::zero(), &mut y, T::one(), Op::NoOp(self), Op::NoOp(&x));
        Ok(())
    }
}

pub struct IdentityOperator;

impl<T: Scalar> LinearOperator<T> for IdentityOperator {
    fn apply(&self, mut y: DVectorSliceMut<T>, x: DVectorSlice<T>) -> Result<(), Box<dyn Error>> {
        y.copy_from(&x);
        Ok(())
    }
}

pub trait CgStoppingCriterion<T: Scalar> {
    /// Called by CG at the start of a new solve.
    fn reset(&self, _a: &dyn LinearOperator<T>, _x: DVectorSlice<T>, _b: DVectorSlice<T>) {}

    fn has_converged(
        &self,
        a: &dyn LinearOperator<T>,
        x: DVectorSlice<T>,
        b: DVectorSlice<T>,
        b_norm: T,
        iteration: usize,
        approx_residual: DVectorSlice<T>,
    ) -> Result<bool, SolveErrorKind>;
}

/// Relative residual tolerance ||r|| <= tol * ||b||.
///
/// Note that we use the *approximate* residual given by Conjugate-Gradient. For ill-conditioned
/// problems, it is possible that CG's residual converges, but the real residual does not.
/// However, in these cases, it is often the case that CG in any case is unable to obtain
/// a more accurate solution, and a better preconditioner would be required if a high-resolution
/// solution is desired.
#[derive(Debug)]
pub struct RelativeResidualCriterion<T: Scalar> {
    tol: T,
}

impl<T: Scalar + Zero> RelativeResidualCriterion<T> {
    pub fn new(tol: T) -> Self {
        Self { tol }
    }
}

impl Default for RelativeResidualCriterion<f64> {
    fn default() -> Self {
        Self::new(1e-8)
    }
}

impl Default for RelativeResidualCriterion<f32> {
    fn default() -> Self {
        Self::new(1e-4)
    }
}

impl<T> CgStoppingCriterion<T> for RelativeResidualCriterion<T>
where
    T: RealField,
{
    fn has_converged(
        &self,
        _a: &dyn LinearOperator<T>,
        _x: DVectorSlice<T>,
        _b: DVectorSlice<T>,
        b_norm: T,
        _iteration: usize,
        approx_residual: DVectorSlice<T>,
    ) -> Result<bool, SolveErrorKind> {
        let r_approx_norm = approx_residual.norm();
        let converged = r_approx_norm <= self.tol * b_norm;
        Ok(converged)
    }
}

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct CgWorkspace<T: Scalar> {
    r: DVector<T>,
    z: DVector<T>,
    p: DVector<T>,
    Ap: DVector<T>,
}

#[allow(non_snake_case)]
struct Buffers<'a, T: Scalar> {
    r: &'a mut DVector<T>,
    z: &'a mut DVector<T>,
    p: &'a mut DVector<T>,
    Ap: &'a mut DVector<T>,
}

impl<T: Scalar + Zero> Default for CgWorkspace<T> {
    fn default() -> Self {
        Self {
            r: DVector::zeros(0),
            z: DVector::zeros(0),
            p: DVector::zeros(0),
            Ap: DVector::zeros(0),
        }
    }
}

impl<T: Scalar + Zero> CgWorkspace<T> {
    fn prepare_buffers(&mut self, dim: usize) -> Buffers<T> {
        self.r.resize_vertically_mut(dim, T::zero());
        self.z.resize_vertically_mut(dim, T::zero());
        self.p.resize_vertically_mut(dim, T::zero());
        self.Ap.resize_vertically_mut(dim, T::zero());
        Buffers {
            r: &mut self.r,
            z: &mut self.z,
            p: &mut self.p,
            Ap: &mut self.Ap,
        }
    }
}

#[derive(Debug)]
enum OwnedOrMutRef<'a, T> {
    Owned(T),
    MutRef(&'a mut T),
}

impl<'a, T> Deref for OwnedOrMutRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(owned) => &owned,
            Self::MutRef(mutref) => &*mutref,
        }
    }
}

impl<'a, T> DerefMut for OwnedOrMutRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(owned) => owned,
            Self::MutRef(mutref) => mutref,
        }
    }
}

#[derive(Debug)]
pub struct ConjugateGradient<'a, T, A, P, Criterion>
where
    T: Scalar,
{
    workspace: OwnedOrMutRef<'a, CgWorkspace<T>>,
    operator: A,
    preconditioner: P,
    stopping_criterion: Criterion,
    max_iter: Option<usize>,
}

impl<'a, T: Scalar + Zero> ConjugateGradient<'a, T, (), IdentityOperator, ()> {
    pub fn new() -> Self {
        Self {
            workspace: OwnedOrMutRef::Owned(CgWorkspace::default()),
            operator: (),
            preconditioner: IdentityOperator,
            stopping_criterion: (),
            max_iter: None,
        }
    }
}

impl<'a, T: Scalar> ConjugateGradient<'a, T, (), IdentityOperator, ()> {
    pub fn with_workspace(workspace: &'a mut CgWorkspace<T>) -> Self {
        Self {
            workspace: OwnedOrMutRef::MutRef(workspace),
            operator: (),
            preconditioner: IdentityOperator,
            stopping_criterion: (),
            max_iter: None,
        }
    }
}

impl<'a, T: Scalar, P, Criterion> ConjugateGradient<'a, T, (), P, Criterion> {
    pub fn with_operator<A>(self, operator: A) -> ConjugateGradient<'a, T, A, P, Criterion> {
        ConjugateGradient {
            workspace: self.workspace,
            operator,
            preconditioner: self.preconditioner,
            stopping_criterion: self.stopping_criterion,
            max_iter: self.max_iter,
        }
    }
}

impl<'a, T: Scalar, A, P, Criterion> ConjugateGradient<'a, T, A, P, Criterion> {
    pub fn with_preconditioner<P2>(self, preconditioner: P2) -> ConjugateGradient<'a, T, A, P2, Criterion> {
        ConjugateGradient {
            workspace: self.workspace,
            operator: self.operator,
            preconditioner,
            stopping_criterion: self.stopping_criterion,
            max_iter: self.max_iter,
        }
    }

    pub fn with_max_iter(self, max_iter: usize) -> Self {
        Self {
            max_iter: Some(max_iter),
            ..self
        }
    }
}

impl<'a, T: Scalar, A, P> ConjugateGradient<'a, T, A, P, ()> {
    pub fn with_stopping_criterion<Criterion>(
        self,
        stopping_criterion: Criterion,
    ) -> ConjugateGradient<'a, T, A, P, Criterion> {
        ConjugateGradient {
            workspace: self.workspace,
            operator: self.operator,
            preconditioner: self.preconditioner,
            stopping_criterion,
            max_iter: self.max_iter,
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum SolveErrorKind {
    OperatorError(Box<dyn Error>),
    PreconditionerError(Box<dyn Error>),
    StoppingCriterionError(Box<dyn Error>),
    IndefiniteOperator,
    IndefinitePreconditioner,
    MaxIterationsReached { max_iter: usize },
}

impl fmt::Display for SolveErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperatorError(err) => {
                write!(f, "Error applying operator: ")?;
                err.fmt(f)
            }
            Self::PreconditionerError(err) => {
                write!(f, "Error applying preconditioner: ")?;
                err.fmt(f)
            }
            Self::StoppingCriterionError(err) => {
                write!(f, "Error evaluating stopping criterion: ")?;
                err.fmt(f)
            }
            Self::IndefiniteOperator => write!(f, "Operator appears to be indefinite: "),
            Self::IndefinitePreconditioner => write!(f, "Indefinite preconditioner: "),
            Self::MaxIterationsReached { max_iter } => {
                write!(f, "Max iterations ({}) reached.", max_iter)
            }
        }
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub struct SolveError<T> {
    pub output: CgOutput<T>,
    pub kind: SolveErrorKind,
}

impl<T> SolveError<T> {
    fn new(output: CgOutput<T>, kind: SolveErrorKind) -> Self {
        Self { output, kind }
    }
}

impl<T> fmt::Display for SolveError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CG solve failed after {}", self.output.num_iterations)?;
        write!(f, "Error: {}", self.kind)
    }
}

impl<T: fmt::Debug> std::error::Error for SolveError<T> {}

/// y = Ax
fn apply_operator<'a, T, A>(
    y: impl Into<DVectorSliceMut<'a, T>>,
    a: &'a A,
    x: impl Into<DVectorSlice<'a, T>>,
) -> Result<(), Box<dyn Error>>
where
    T: Scalar,
    A: LinearOperator<T>,
{
    a.apply(y.into(), x.into())
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CgOutput<T> {
    /// Number of iterations of the solver.
    ///
    /// Corresponds to the number of updates made to the (initial) solution vector,
    pub num_iterations: usize,
    marker: PhantomData<T>,
}

impl<'a, T, A, P, Criterion> ConjugateGradient<'a, T, A, P, Criterion>
where
    T: RealField,
    A: LinearOperator<T>,
    P: LinearOperator<T>,
    Criterion: CgStoppingCriterion<T>,
{
    pub fn solve_with_guess<'b>(
        &mut self,
        b: impl Into<DVectorSlice<'b, T>>,
        x: impl Into<DVectorSliceMut<'b, T>>,
    ) -> Result<CgOutput<T>, SolveError<T>> {
        self.solve_with_guess_(b.into(), x.into())
    }

    #[allow(non_snake_case)]
    fn solve_with_guess_(
        &mut self,
        b: DVectorSlice<T>,
        mut x: DVectorSliceMut<T>,
    ) -> Result<CgOutput<T>, SolveError<T>> {
        use SolveErrorKind::*;
        assert_eq!(b.len(), x.len());

        let mut output = CgOutput {
            num_iterations: 0,
            marker: PhantomData,
        };

        let Buffers { r, z, p, Ap } = self.workspace.prepare_buffers(x.len());

        // r = b - Ax
        if let Err(err) = apply_operator(&mut *r, &self.operator, &x) {
            return Err(SolveError::new(output, OperatorError(err)));
        }
        r.zip_apply(&b, |Ax_i, b_i| b_i - Ax_i);

        // z = Pr
        if let Err(err) = apply_operator(&mut *z, &self.preconditioner, &*r) {
            return Err(SolveError::new(output, PreconditionerError(err)));
        }

        // p = z
        p.copy_from(&z);

        let mut zTr = z.dot(r);
        let mut pAp;

        let b_norm = b.norm();

        if b_norm == T::zero() {
            x.fill(T::zero());
            return Ok(output);
        }

        loop {
            // TODO: Can we simplify this monstronsity?
            let convergence = self.stopping_criterion.has_converged(
                &self.operator,
                (&x).into(),
                (&b).into(),
                b_norm,
                output.num_iterations,
                (&*r).into(),
            );

            let has_converged = match convergence {
                Ok(converged) => converged,
                Err(error_kind) => return Err(SolveError::new(output, error_kind)),
            };

            if has_converged {
                break;
            } else if let Some(max_iter) = self.max_iter {
                if output.num_iterations >= max_iter {
                    return Err(SolveError::new(output, MaxIterationsReached { max_iter }));
                }
            }

            // Ap = A * p
            if let Err(err) = apply_operator(&mut *Ap, &self.operator, &*p) {
                return Err(SolveError::new(output, OperatorError(err)));
            }
            pAp = p.dot(&Ap);

            if pAp <= T::zero() {
                return Err(SolveError {
                    output,
                    kind: SolveErrorKind::IndefiniteOperator,
                });
            }
            if zTr <= T::zero() {
                return Err(SolveError {
                    output,
                    kind: SolveErrorKind::IndefinitePreconditioner,
                });
            }

            let alpha = zTr / pAp;
            // x <- x + alpha * p
            x.zip_apply(&*p, |x_i, p_i| x_i + alpha * p_i);
            // r <- r - alpha * Ap
            r.zip_apply(&*Ap, |r_i, Ap_i| r_i - alpha * Ap_i);

            // Number of iterations corresponds to number of updates to the x vector
            output.num_iterations += 1;

            // z <- P r
            if let Err(err) = apply_operator(&mut *z, &self.preconditioner, &*r) {
                return Err(SolveError::new(output, PreconditionerError(err)));
            }
            let zTr_next = z.dot(&*r);
            let beta = zTr_next / zTr;

            // p <- z + beta * p
            p.zip_apply(&*z, |p_i, z_i| z_i + beta * p_i);

            zTr = zTr_next;
        }

        Ok(output)
    }
}

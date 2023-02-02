//! Tools for integrating functions on finite element spaces.
use crate::allocators::{BiDimAllocator, DimAllocator, TriDimAllocator};
use crate::assembly::buffers::{BasisFunctionBuffer, QuadratureBuffer};
use crate::assembly::global::gather_global_to_local;
use crate::assembly::local::{ElementConnectivityAssembler, ElementScalarAssembler, QuadratureTable};
use crate::element::{FiniteElement, VolumetricFiniteElement};
use crate::nalgebra::{DVector, DefaultAllocator, DimName, OMatrix, OPoint, Scalar, U1};
use crate::quadrature::Quadrature;
use crate::space::{ElementInSpace, FiniteElementSpace, VolumetricFiniteElementSpace};
use crate::util::{reshape_to_slice, try_transmute_ref};
use crate::{Real, SmallDim};
use davenport::{define_thread_local_workspace, with_thread_local_workspace};
use eyre::eyre;
use nalgebra::{DVectorSlice, Dynamic, MatrixSliceMut, OVector};
use std::marker::PhantomData;

/// Computes the Riemannian volume form for the given dimensions.
///
/// TODO: This is not actively tested at the moment, need to do this.
pub fn volume_form<T, GeometryDim, ReferenceDim>(jacobian: &OMatrix<T, GeometryDim, ReferenceDim>) -> T
where
    T: Real,
    GeometryDim: SmallDim,
    ReferenceDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, ReferenceDim>,
{
    if GeometryDim::is::<ReferenceDim>() {
        let jacobian: &OMatrix<T, GeometryDim, GeometryDim> =
            try_transmute_ref(jacobian).expect("This cannot fail since we know that GeometryDim == ReferenceDim");
        jacobian.determinant().abs()
    } else {
        // TODO: Specialize other dimension combinations
        (jacobian.transpose() * jacobian).determinant().sqrt()
    }
}

/// A wrapper for turning an [`Fn`] into a [`Function`].
///
/// This wrapper works around some limitations of the type system, and provides facilities
/// to disambiguate otherwise ambiguous situations. See the below examples for usage.
///
/// # Examples
///
/// ```rust
/// use nalgebra::{Matrix2, Point2, U2, vector, Vector1, Vector2};
/// use fenris::element::{Tet4Element, Tri3d2Element};
/// use fenris::integrate::{FnFunction, integrate_over_element, integrate_over_volume_element,
///     IntegrationWorkspace};
/// use fenris::integrate::dependency::{DependsOnGrad, DependsOnU, NoDeps};
/// use fenris::mesh::procedural::create_unit_square_uniform_tri_mesh_2d;
/// use fenris::quadrature::CanonicalStiffnessQuadrature;
/// use fenris::util::global_vector_from_point_fn;
///
/// // Set up some arbitrary test data: a 2D linear triangle element,
/// // a quadrature rule and arbitrary interpolation weights
/// let element = Tri3d2Element::<f64>::reference();
/// let quadrature = element.canonical_stiffness_quadrature();
/// let u = vector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mut workspace = IntegrationWorkspace::default();
///
/// // Integrate a function f(x) with integrate_over_element
/// let f1 = FnFunction::new(|x: &Point2<_>| vector![x.x + x.y, 3.0])
///     // Since the function has no dependency on u, we need to indicate which
///     // dimension the solution variables have if we want to use it
///     // in a context where UFunction/UGradFunction is required
///     .with_dependencies::<NoDeps<U2>>();
/// let i1 = integrate_over_element(&f1, &element, &quadrature, &u, &mut workspace);
///
/// // Integrate a function f(x, u) -> R^2
/// let f2 = FnFunction::new(|x: &Point2<_>, u: &Vector2<_>| vector![x.x * u[1], u[0] + u[1]]);
/// let i2 = integrate_over_element(&f2, &element, &quadrature, &u, &mut workspace);
///
/// // Integrate a function f(x, u, grad u) -> R^2
/// // In this case we need to use integrate_over_volume_element, since grad u is not
/// // well-defined for surface elements
/// let f3 = FnFunction::new(|x: &Point2<_>, u: &Vector2<_>, u_grad: &Matrix2<_>|
///     vector![x.x + x.y + u.x + u.y, u_grad.determinant()]);
/// let i3 = integrate_over_volume_element(&f3, &element, &quadrature, &u, &mut workspace)
///     .expect("Element is non-degenerate");
///
/// // Integrate a function f(x, grad u) -> R^2
/// let f4 = FnFunction::new(|x: &Point2<_>, u_grad: &Matrix2<_>|
///     vector![x.x + x.y + u_grad.determinant(), u_grad.norm()])
///     // In order to resolve ambiguities due to the two-parameter closure,
///     // we need to declare that the function depends only on grad u and not u
///     .with_dependencies::<DependsOnGrad>();
/// let i4 = integrate_over_volume_element(&f4, &element, &quadrature, &u, &mut workspace)
///     .expect("Element is non-degenerate");
///
/// // Similarly, if we want to ingrate f2, which is of the form f(x, u),
/// // with integrate_over_volume_element, we must declare that it depends only on u
/// let f5 = f2.with_dependencies::<DependsOnU>();
/// let i5 = integrate_over_volume_element(&f5, &element, &quadrature, &u, &mut workspace)
///     .expect("Element is non-degenerate");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FnFunction<F, Dependencies = dependency::AutoDeps> {
    f: F,
    marker: PhantomData<Dependencies>,
}

impl<F> FnFunction<F> {
    pub fn new(f: F) -> Self {
        Self { f, marker: PhantomData }
    }

    pub fn with_dependencies<Dependencies>(self) -> FnFunction<F, Dependencies> {
        FnFunction {
            f: self.f,
            marker: PhantomData,
        }
    }
}

/// Dependency declarations for [`FnFunction`].
pub mod dependency {
    use std::marker::PhantomData;

    /// Indicates that dependencies are "automatic", i.e. the natural dependencies for the
    /// given situation.
    ///
    /// This is generally used if the wrapped `Fn` has the same number of parameters as the
    /// function trait it is used with. For example, if passed to a function taking an instance
    /// of [`UFunction`](crate::integrate::UFunction)
    /// ($f(x, u)$, the "automatic" dependencies are $x$ and $u$.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct AutoDeps;

    /// The function has no dependencies on $u$, i.e. $f = f(x)$.
    ///
    /// Since the function does not depend on $u$, it is necessary to specify the solution
    /// dimension, since it cannot be deduced from the function parameters.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct NoDeps<SolutionDim> {
        marker: PhantomData<SolutionDim>,
    }

    /// The function has the form $f(x, u)$.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct DependsOnU;

    /// The function has the form $f(x, \nabla u)$.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct DependsOnGrad;
}

impl<T, F, GeometryDim, OutputDim> Function<T, GeometryDim> for FnFunction<F>
where
    T: Scalar,
    F: Fn(&OPoint<T, GeometryDim>) -> OVector<T, OutputDim>,
    GeometryDim: SmallDim,
    OutputDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + DimAllocator<T, GeometryDim>,
{
    type OutputDim = OutputDim;

    fn evaluate(&self, x: &OPoint<T, GeometryDim>) -> OVector<T, Self::OutputDim> {
        (self.f)(x)
    }
}

impl<T, F, GeometryDim, OutputDim, SolutionDim> UFunction<T, GeometryDim, SolutionDim> for FnFunction<F>
where
    T: Scalar,
    F: Fn(&OPoint<T, GeometryDim>, &OVector<T, SolutionDim>) -> OVector<T, OutputDim>,
    GeometryDim: SmallDim,
    OutputDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + DimAllocator<T, GeometryDim> + DimAllocator<T, SolutionDim>,
{
    type OutputDim = OutputDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        u: impl FnOnce() -> OVector<T, SolutionDim>,
    ) -> OVector<T, Self::OutputDim> {
        (self.f)(x, &u())
    }
}

impl<T, F, GeometryDim, OutputDim, SolutionDim> UFunction<T, GeometryDim, SolutionDim>
    for FnFunction<F, dependency::NoDeps<SolutionDim>>
where
    T: Scalar,
    F: Fn(&OPoint<T, GeometryDim>) -> OVector<T, OutputDim>,
    GeometryDim: SmallDim,
    OutputDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + DimAllocator<T, GeometryDim> + DimAllocator<T, SolutionDim>,
{
    type OutputDim = OutputDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        _: impl FnOnce() -> OVector<T, SolutionDim>,
    ) -> OVector<T, Self::OutputDim> {
        (self.f)(x)
    }
}

impl<T, F, GeometryDim, OutputDim, SolutionDim> UGradFunction<T, GeometryDim, SolutionDim> for FnFunction<F>
where
    T: Scalar,
    F: Fn(
        &OPoint<T, GeometryDim>,
        &OVector<T, SolutionDim>,
        &OMatrix<T, GeometryDim, SolutionDim>,
    ) -> OVector<T, OutputDim>,
    GeometryDim: SmallDim,
    OutputDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    type OutputDim = OutputDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        u: impl FnOnce() -> OVector<T, SolutionDim>,
        u_grad: impl FnOnce() -> OMatrix<T, GeometryDim, SolutionDim>,
    ) -> OVector<T, Self::OutputDim> {
        (self.f)(x, &u(), &u_grad())
    }
}

impl<T, F, GeometryDim, OutputDim, SolutionDim> UGradFunction<T, GeometryDim, SolutionDim>
    for FnFunction<F, dependency::DependsOnGrad>
where
    T: Scalar,
    F: Fn(&OPoint<T, GeometryDim>, &OMatrix<T, GeometryDim, SolutionDim>) -> OVector<T, OutputDim>,
    GeometryDim: SmallDim,
    OutputDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    type OutputDim = OutputDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        _: impl FnOnce() -> OVector<T, SolutionDim>,
        u_grad: impl FnOnce() -> OMatrix<T, GeometryDim, SolutionDim>,
    ) -> OVector<T, Self::OutputDim> {
        (self.f)(x, &u_grad())
    }
}

impl<T, F, GeometryDim, OutputDim, SolutionDim> UGradFunction<T, GeometryDim, SolutionDim>
    for FnFunction<F, dependency::DependsOnU>
where
    T: Scalar,
    F: Fn(&OPoint<T, GeometryDim>, &OVector<T, SolutionDim>) -> OVector<T, OutputDim>,
    GeometryDim: SmallDim,
    OutputDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    type OutputDim = OutputDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        u: impl FnOnce() -> OVector<T, SolutionDim>,
        _: impl FnOnce() -> OMatrix<T, GeometryDim, SolutionDim>,
    ) -> OVector<T, Self::OutputDim> {
        (self.f)(x, &u())
    }
}

// pub fn function_from_fn<T, GeometryDim, OutputDim>(f: impl Fn(&OPoint<T, GeometryDim>) -> OVector<T, OutputDim>)
//     -> impl Function<T, GeometryDim>
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     OutputDim: SmallDim,
//     DefaultAllocator: DimAllocator<T, OutputDim> + DimAllocator<T, GeometryDim>
// {
//     struct Wrapper<F>(F);
//
//     impl<T, F, GeometryDim, OutputDim> Function<T, GeometryDim> for Wrapper<F>
//     where
//         T: Scalar,
//         F: Fn(&OPoint<T, GeometryDim>) -> OVector<T, OutputDim>,
//         GeometryDim: SmallDim,
//         OutputDim: SmallDim,
//         DefaultAllocator: DimAllocator<T, OutputDim> + DimAllocator<T, GeometryDim>
//     {
//         type OutputDim = OutputDim;
//
//         fn evaluate(&self, x: &OPoint<T, GeometryDim>) -> OVector<T, Self::OutputDim> {
//             self.0(x)
//         }
//     }
//
//     Wrapper(f)
// }

/// A vector-valued function $f(x)$.
///
/// Functions of this type can be integrated over both volumes and surfaces.
pub trait Function<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: DimAllocator<T, GeometryDim> + DimAllocator<T, Self::OutputDim>,
{
    type OutputDim: SmallDim;

    fn evaluate(&self, x: &OPoint<T, GeometryDim>) -> OVector<T, Self::OutputDim>;
}

// impl<T, F, GeometryDim, OutputDim> Function<T, GeometryDim> for F
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     OutputDim: SmallDim,
//     F: Fn(&OPoint<T, GeometryDim>) -> OVector<T, OutputDim>,
//     DefaultAllocator: DimAllocator<T, GeometryDim> + DimAllocator<T, OutputDim>
// {
//     type OutputDim = OutputDim;
//
//     fn evaluate(&self, x: &OPoint<T, GeometryDim>) -> OVector<T, OutputDim> {
//         self(x)
//     }
// }

/// A vector-valued function $f(x, u)$.
///
/// Here $u = u(x)$ is a function $u: \Omega \rightarrow \mathbb{R}^s$.
///
/// Functions of this type can be integrated over both volumes and surfaces.
pub trait UFunction<T, GeometryDim, SolutionDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, SolutionDim> + DimAllocator<T, Self::OutputDim> + DimAllocator<T, GeometryDim>,
{
    type OutputDim: SmallDim;

    fn evaluate<'a>(
        &'a self,
        x: &OPoint<T, GeometryDim>,
        u: impl FnOnce() -> OVector<T, SolutionDim>,
    ) -> OVector<T, Self::OutputDim>;
}

// impl<T, GeometryDim, SolutionDim, F> UFunction<T, GeometryDim, SolutionDim> for F
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     SolutionDim: SmallDim,
//     F: Function<T, GeometryDim>,
//     DefaultAllocator: DimAllocator<T, F::OutputDim> + DimAllocator<T, GeometryDim> + DimAllocator<T, SolutionDim>
// {
//     type OutputDim = F::OutputDim;
//
//     fn evaluate(&self,
//                 x: &OPoint<T, GeometryDim>,
//                 _: impl FnOnce() -> OVector<T, SolutionDim>
//     ) -> OVector<T, Self::OutputDim> {
//         self.evaluate(x)
//     }
// }

// /// A vector-valued function $f(x, u)$.
// ///
// /// Here $u = u(x)$ is a function $u: \Omega \rightarrow \mathbb{R}^s$.
// ///
// /// Functions of this type can be integrated over both volumes and surfaces.
// #[deprecated]
// pub trait OldUFunction<T, GeometryDim>
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     DefaultAllocator:
//         DimAllocator<T, Self::SolutionDim> + DimAllocator<T, Self::OutputDim> + DimAllocator<T, GeometryDim>,
// {
//     type OutputDim: SmallDim;
//     type SolutionDim: SmallDim;
//
//     fn evaluate(&self, x: &OPoint<T, GeometryDim>, u: &OVector<T, Self::SolutionDim>) -> OVector<T, Self::OutputDim>;
// }

// impl<'a, T, GeometryDim, F> OldUFunction<T, GeometryDim> for &'a F
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     F: OldUFunction<T, GeometryDim>,
//     DefaultAllocator: DimAllocator<T, F::SolutionDim> + DimAllocator<T, F::OutputDim> + DimAllocator<T, GeometryDim>,
// {
//     type OutputDim = F::OutputDim;
//     type SolutionDim = F::SolutionDim;
//
//     fn evaluate(&self, x: &OPoint<T, GeometryDim>, u: &OVector<T, Self::SolutionDim>) -> OVector<T, Self::OutputDim> {
//         F::evaluate(self, x, u)
//     }
// }

// #[derive(Debug)]
// pub struct Integrand<SolutionDim = (), F = ()> {
//     marker: PhantomData<SolutionDim>,
//     function: F,
// }

// impl Integrand {
//     pub fn new_with_solution_dim<SolutionDim>() -> Integrand<SolutionDim> {
//         Integrand {
//             marker: Default::default(),
//             function: (),
//         }
//     }
// }

// impl<SolutionDim> Integrand<SolutionDim, ()> {
//     pub fn with_function<F>(self, f: F) -> Integrand<SolutionDim, F> {
//         Integrand {
//             marker: Default::default(),
//             function: f,
//         }
//     }
//
//     pub fn with_volume_function<F>(self, f: F) -> Integrand<SolutionDim, VolumeIntegrand<F>> {
//         Integrand {
//             marker: Default::default(),
//             function: VolumeIntegrand(f),
//         }
//     }
// }

// impl<F, T, OutputDim, SolutionDim, GeometryDim> OldUFunction<T, GeometryDim> for Integrand<SolutionDim, F>
// where
//     F: Fn(&OPoint<T, GeometryDim>, &OVector<T, SolutionDim>) -> OVector<T, OutputDim>,
//     T: Scalar,
//     OutputDim: SmallDim,
//     SolutionDim: SmallDim,
//     GeometryDim: SmallDim,
//     DefaultAllocator: DimAllocator<T, OutputDim> + DimAllocator<T, SolutionDim> + DimAllocator<T, GeometryDim>,
// {
//     type OutputDim = OutputDim;
//     type SolutionDim = SolutionDim;
//
//     fn evaluate(&self, x: &OPoint<T, GeometryDim>, u: &OVector<T, SolutionDim>) -> OVector<T, OutputDim> {
//         (self.function)(x, u)
//     }
// }

// /// A vector-valued function $f(x, u, \nabla u)$.
// ///
// /// Here $u = u(x)$ is a function $u: \Omega \rightarrow \mathbb{R}^s$.
// ///
// /// Functions of this type can be integrated only over volumes, since the gradient $\nabla u$
// /// is otherwise not well-defined.
// #[deprecated]
// pub trait OldUGradUFunction<T, GeometryDim>
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     DefaultAllocator: DimAllocator<T, Self::OutputDim> + BiDimAllocator<T, Self::SolutionDim, GeometryDim>,
// {
//     type OutputDim: SmallDim;
//     type SolutionDim: SmallDim;
//
//     fn evaluate(
//         &self,
//         x: &OPoint<T, GeometryDim>,
//         u: &OVector<T, Self::SolutionDim>,
//         u_grad: &OMatrix<T, GeometryDim, Self::SolutionDim>,
//     ) -> OVector<T, Self::OutputDim>;
// }

/// A vector-valued function $f(x, u, \nabla u)$.
///
/// Here $u = u(x)$ is a function $u: \Omega \rightarrow \mathbb{R}^s$.
///
/// Functions of this type can be integrated only over volumes, since the gradient $\nabla u$
/// is otherwise not well-defined.
pub trait UGradFunction<T, GeometryDim, SolutionDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: DimAllocator<T, Self::OutputDim> + BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    type OutputDim: SmallDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        u: impl FnOnce() -> OVector<T, SolutionDim>,
        u_grad: impl FnOnce() -> OMatrix<T, GeometryDim, SolutionDim>,
    ) -> OVector<T, Self::OutputDim>;
}

// impl<T, GeometryDim, SolutionDim, F> UGradUFunction<T, GeometryDim, SolutionDim> for F
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     SolutionDim: SmallDim,
//     F: UFunction<T, GeometryDim, SolutionDim>,
//     DefaultAllocator: DimAllocator<T, F::OutputDim> + BiDimAllocator<T, GeometryDim, SolutionDim>,
// {
//     type OutputDim = F::OutputDim;
//
//     fn evaluate(&self,
//                 x: &OPoint<T, GeometryDim>,
//                 u: impl FnOnce() -> OVector<T, SolutionDim>,
//                 _: impl FnOnce() -> OMatrix<T, GeometryDim, SolutionDim>
//     ) -> OVector<T, Self::OutputDim> {
//         self.evaluate(x, u)
//     }
// }

// impl<'a, T, F, GeometryDim, SolutionDim> UGradUFunction<T, GeometryDim, SolutionDim> for &'a F
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     SolutionDim: SmallDim,
//     F: UGradUFunction<T, GeometryDim, SolutionDim>,
//     DefaultAllocator: BiDimAllocator<T, SolutionDim, GeometryDim> + DimAllocator<T, F::OutputDim>,
// {
//     type OutputDim = F::OutputDim;
//
//     fn evaluate(
//         &self,
//         x: &OPoint<T, GeometryDim>,
//         u: &OVector<T, SolutionDim>,
//         u_grad: &OMatrix<T, GeometryDim, SolutionDim>,
//     ) -> OVector<T, Self::OutputDim> {
//         F::evaluate(self, x, u, u_grad)
//     }
// }

// impl<'a, T, GeometryDim, F> OldUGradUFunction<T, GeometryDim> for &'a F
// where
//     T: Scalar,
//     GeometryDim: SmallDim,
//     F: OldUGradUFunction<T, GeometryDim>,
//     DefaultAllocator: BiDimAllocator<T, F::SolutionDim, GeometryDim> + DimAllocator<T, F::OutputDim>,
// {
//     type OutputDim = F::OutputDim;
//     type SolutionDim = F::SolutionDim;
//
//     fn evaluate(
//         &self,
//         x: &OPoint<T, GeometryDim>,
//         u: &OVector<T, Self::SolutionDim>,
//         u_grad: &OMatrix<T, GeometryDim, Self::SolutionDim>,
//     ) -> OVector<T, Self::OutputDim> {
//         F::evaluate(self, x, u, u_grad)
//     }
// }

// impl<F, T, OutputDim, SolutionDim, GeometryDim> OldUGradUFunction<T, GeometryDim>
//     for Integrand<SolutionDim, VolumeIntegrand<F>>
// where
//     F: Fn(
//         &OPoint<T, GeometryDim>,
//         &OVector<T, SolutionDim>,
//         &OMatrix<T, GeometryDim, SolutionDim>,
//     ) -> OVector<T, OutputDim>,
//     T: Scalar,
//     OutputDim: SmallDim,
//     SolutionDim: SmallDim,
//     GeometryDim: SmallDim,
//     DefaultAllocator: DimAllocator<T, OutputDim> + BiDimAllocator<T, SolutionDim, GeometryDim>,
// {
//     type OutputDim = OutputDim;
//     type SolutionDim = SolutionDim;
//
//     fn evaluate(
//         &self,
//         x: &OPoint<T, GeometryDim>,
//         u: &OVector<T, SolutionDim>,
//         u_grad: &OMatrix<T, GeometryDim, SolutionDim>,
//     ) -> OVector<T, OutputDim> {
//         (self.function.0)(x, u, u_grad)
//     }
// }

pub struct IntegrationWorkspace<T: Scalar> {
    basis_buffer: BasisFunctionBuffer<T>,
}

impl<T: Real> Default for IntegrationWorkspace<T> {
    fn default() -> Self {
        Self {
            basis_buffer: BasisFunctionBuffer::default(),
        }
    }
}

/// Integrates the given function on the given element with the provided quadrature and interpolation weights.
pub fn integrate_over_element<'a, T, F, Element, SolutionDim>(
    integrand: &F,
    element: &Element,
    quadrature: impl Quadrature<T, Element::ReferenceDim>,
    interpolation_weights: impl Into<DVectorSlice<'a, T>>,
    workspace: &mut IntegrationWorkspace<T>,
) -> OVector<T, F::OutputDim>
where
    T: Real,
    F: UFunction<T, Element::GeometryDim, SolutionDim>,
    SolutionDim: SmallDim,
    Element: FiniteElement<T>,
    DefaultAllocator: TriDimAllocator<T, Element::GeometryDim, Element::ReferenceDim, SolutionDim>
        // This is a separate bound because we generally don't need to mix the output dimension
        // with the other dimensions, so this way the bounds necessary for downstream consumers
        // are somewhat relaxed (the output dimension is often *fixed*, so maybe no bounds at all are necessary)
        + DimAllocator<T, F::OutputDim>,
{
    let interpolation_weights = interpolation_weights.into();

    let n = element.num_nodes();
    let (weights, points) = (quadrature.weights(), quadrature.points());
    let basis_buffer = &mut workspace.basis_buffer;
    basis_buffer.resize(element.num_nodes(), Element::ReferenceDim::dim());

    let mut result = OVector::<T, F::OutputDim>::zeros();
    for (w, p_ref) in weights.iter().zip(points) {
        let u_h = || {
            element.populate_basis(basis_buffer.element_basis_values_mut(), p_ref);
            crate::util::compute_interpolation(
                interpolation_weights,
                DVectorSlice::from_slice(basis_buffer.element_basis_values(), n),
            )
        };
        let x = element.map_reference_coords(p_ref);
        let jacobian = element.reference_jacobian(p_ref);
        let f = integrand.evaluate(&x, u_h);
        let volume_form = volume_form(&jacobian);

        result += f * (w.clone() * volume_form);
    }

    result
}

#[derive(Debug)]
pub enum IntegrationFailure {
    SingularJacobian,
}

/// Integrates the given volume function on the given element with the provided quadrature and interpolation weights.
pub fn integrate_over_volume_element<'a, T, Element, F, SolutionDim>(
    function: &F,
    element: &Element,
    quadrature: impl Quadrature<T, Element::ReferenceDim>,
    interpolation_weights: impl Into<DVectorSlice<'a, T>>,
    workspace: &mut IntegrationWorkspace<T>,
) -> Result<OVector<T, F::OutputDim>, IntegrationFailure>
where
    T: Real,
    F: UGradFunction<T, Element::GeometryDim, SolutionDim>,
    SolutionDim: SmallDim,
    Element: VolumetricFiniteElement<T>,
    DefaultAllocator:
        TriDimAllocator<T, Element::GeometryDim, Element::ReferenceDim, SolutionDim> + DimAllocator<T, F::OutputDim>,
{
    let interpolation_weights = interpolation_weights.into();
    let n = element.num_nodes();
    let r = Element::ReferenceDim::dim();
    let basis_buffer = &mut workspace.basis_buffer;
    basis_buffer.resize(element.num_nodes(), Element::ReferenceDim::dim());

    let mut result = OVector::<T, F::OutputDim>::zeros();
    for (w, p_ref) in quadrature.weights().iter().zip(quadrature.points()) {
        let x = element.map_reference_coords(p_ref);
        let jacobian = element.reference_jacobian(p_ref);
        let jacobian_inv_t = jacobian
            .transpose()
            .try_inverse()
            .ok_or_else(|| IntegrationFailure::SingularJacobian)?;

        let (values_buffer, mut gradients_buffer): (_, MatrixSliceMut<_, Element::ReferenceDim, _>) =
            basis_buffer.element_values_gradients_mut();

        // First we compute u_h
        let u_h = || {
            element.populate_basis(values_buffer, p_ref);
            crate::util::compute_interpolation(interpolation_weights, DVectorSlice::from_slice(values_buffer, n))
        };

        // Then we compute u_h_grad. To do so we first compute the gradient with respect to *reference element coords*,
        // then we transform this to physical coordinates by the inverse transposed Jacobian
        let u_h_grad = || {
            element.populate_basis_gradients(MatrixSliceMut::from(&mut gradients_buffer), p_ref);
            let reference_gradients = gradients_buffer;
            // let reference_gradients = basis_buffer.element_gradients::<Element::ReferenceDim>();
            let reference_gradients = reshape_to_slice(&reference_gradients, (Dynamic::new(r * n), U1::name()));
            let u_h_ref_grad: OMatrix<T, Element::ReferenceDim, SolutionDim> =
                crate::util::compute_interpolation_gradient(interpolation_weights, &reference_gradients);
            let u_h_grad = jacobian_inv_t * u_h_ref_grad;
            u_h_grad
        };

        let f = function.evaluate(&x, u_h, u_h_grad);
        let volume_form = volume_form(&jacobian);

        result += f * (w.clone() * volume_form);
    }

    Ok(result)
}

pub struct ElementIntegralAssembler<'a, T, F, SolutionDim, Space, QTable>
where
    T: Scalar,
{
    space: &'a Space,
    u: DVectorSlice<'a, T>,
    integrand: F,
    qtable: &'a QTable,
    marker: PhantomData<SolutionDim>,
}

pub struct ElementIntegralVolumeAssembler<'a, T, F, SolutionDim, Space, QTable>
where
    T: Scalar,
{
    space: &'a Space,
    u: DVectorSlice<'a, T>,
    integrand: F,
    qtable: &'a QTable,
    marker: PhantomData<SolutionDim>,
}

pub struct ElementIntegralAssemblerBuilder<'a, T, F, SolutionDim, Space, QTable>
where
    T: Scalar,
{
    space: Option<&'a Space>,
    u: Option<DVectorSlice<'a, T>>,
    integrand: Option<F>,
    qtable: Option<&'a QTable>,
    marker: PhantomData<SolutionDim>,
}

impl<'a, T, F, SolutionDim, Space, QTable> ElementIntegralAssemblerBuilder<'a, T, F, SolutionDim, Space, QTable>
where
    T: Scalar,
    SolutionDim: SmallDim,
{
    pub fn new() -> Self {
        Self {
            space: None,
            u: None,
            integrand: None,
            qtable: None,
            marker: PhantomData,
        }
    }

    pub fn with_space(self, space: &'a Space) -> Self {
        Self {
            space: Some(space),
            ..self
        }
    }

    pub fn with_quadrature_table(self, qtable: &'a QTable) -> Self {
        Self {
            qtable: Some(qtable),
            ..self
        }
    }

    pub fn with_interpolation_weights(self, u: impl Into<DVectorSlice<'a, T>>) -> Self {
        Self {
            u: Some(u.into()),
            ..self
        }
    }

    pub fn with_integrand(self, integrand: F) -> Self {
        Self {
            integrand: Some(integrand),
            ..self
        }
    }

    pub fn build_integrator(self) -> ElementIntegralAssembler<'a, T, F, SolutionDim, Space, QTable>
    where
        Space: FiniteElementSpace<T>,
        F: UFunction<T, Space::GeometryDim, SolutionDim>,
        DefaultAllocator:
            TriDimAllocator<T, SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
    {
        // We take all the trait bounds here so that we can do some sanity checking.
        // This makes it much easier for the user to debug where something went wrong,
        // such as mismatch between interpolation length vector size and number of nodes in space etc.
        let assembler = ElementIntegralAssembler {
            space: self.space.expect("Must provide space"),
            u: self.u.expect("Must provide interpolation weights"),
            integrand: self.integrand.expect("Must provide integrand"),
            qtable: self.qtable.expect("Must provide quadrature table"),
            marker: PhantomData,
        };

        let ndof = assembler.space.num_nodes() * SolutionDim::dim();
        assert_eq!(
            assembler.u.len(),
            ndof,
            "Size of interpolation weight vector does not match expected number of DOFs ( {} x {} )",
            SolutionDim::dim(),
            assembler.space.num_nodes()
        );

        assembler
    }

    pub fn build_volume_integrator(self) -> ElementIntegralVolumeAssembler<'a, T, F, SolutionDim, Space, QTable>
    where
        Space: VolumetricFiniteElementSpace<T>,
        F: UGradFunction<T, Space::ReferenceDim, SolutionDim>,
        DefaultAllocator:
            TriDimAllocator<T, SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
    {
        let assembler = ElementIntegralVolumeAssembler {
            space: self.space.expect("Must provide space"),
            u: self.u.expect("Must provide interpolation weights"),
            integrand: self.integrand.expect("Must provide integrand"),
            qtable: self.qtable.expect("Must provide quadrature table"),
            marker: PhantomData,
        };

        let ndof = assembler.space.num_nodes() * SolutionDim::dim();
        assert_eq!(
            assembler.u.len(),
            ndof,
            "Size of interpolation weight vector does not match expected number of DOFs ( {} x {} )",
            SolutionDim::dim(),
            assembler.space.num_nodes()
        );

        assembler
    }
}

impl<'a, T, F, SolutionDim, Space, QTable> ElementConnectivityAssembler
    for ElementIntegralAssembler<'a, T, F, SolutionDim, Space, QTable>
where
    T: Scalar,
    SolutionDim: SmallDim,
    Space: FiniteElementSpace<T>,
    F: UFunction<T, Space::GeometryDim, SolutionDim>,
    DefaultAllocator:
        TriDimAllocator<T, SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
{
    fn solution_dim(&self) -> usize {
        SolutionDim::dim()
    }

    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(output, element_index)
    }
}

define_thread_local_workspace!(WORKSPACE);

struct ElementIntegralAssemblerWorkspace<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: DimAllocator<T, D>,
{
    integration_workspace: IntegrationWorkspace<T>,
    quadrature_buffer: QuadratureBuffer<T, D>,
    local_interpolation_weights: DVector<T>,
    nodes: Vec<usize>,
}

impl<T, D> Default for ElementIntegralAssemblerWorkspace<T, D>
where
    T: Real,
    D: DimName,
    DefaultAllocator: DimAllocator<T, D>,
{
    fn default() -> Self {
        Self {
            integration_workspace: Default::default(),
            quadrature_buffer: Default::default(),
            local_interpolation_weights: DVector::zeros(0),
            nodes: Default::default(),
        }
    }
}

impl<'a, T, F, SolutionDim, Space, QTable> ElementScalarAssembler<T>
    for ElementIntegralAssembler<'a, T, F, SolutionDim, Space, QTable>
where
    T: Real,
    F: UFunction<T, Space::GeometryDim, SolutionDim>,
    SolutionDim: SmallDim,
    Space: FiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator:
        TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim> + DimAllocator<T, F::OutputDim>,
{
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T> {
        let n = self.element_node_count(element_index);
        let s = self.solution_dim();
        let element_ndof = n * s;
        let integral = with_thread_local_workspace(
            &WORKSPACE,
            |workspace: &mut ElementIntegralAssemblerWorkspace<T, Space::ReferenceDim>| {
                workspace
                    .quadrature_buffer
                    .populate_element_weights_and_points_from_table(element_index, self.qtable);
                workspace
                    .local_interpolation_weights
                    .resize_vertically_mut(element_ndof, T::zero());
                workspace.nodes.resize(n, usize::MAX);
                self.populate_element_nodes(&mut workspace.nodes, element_index);
                let u_local = &mut workspace.local_interpolation_weights;
                let quadrature = workspace.quadrature_buffer.weights_and_points();
                gather_global_to_local(&self.u, &mut *u_local, &workspace.nodes, s);
                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                integrate_over_element(
                    &self.integrand,
                    &element,
                    quadrature,
                    u_local,
                    &mut workspace.integration_workspace,
                )
            },
        );
        Ok(integral[0])
    }
}

impl<'a, T, F, SolutionDim, Space, QTable> ElementConnectivityAssembler
    for ElementIntegralVolumeAssembler<'a, T, F, SolutionDim, Space, QTable>
where
    T: Real,
    // TODO: For some reason this only works if we specify Space::ReferenceDim. However, Space::GeometryDim would be
    // more appropriate, and we anyway have Space::GeometryDim == Space::ReferenceDim by definition of
    // a volumetric finite element space... But unsure if it may cause downstream issues
    F: UGradFunction<T, Space::ReferenceDim, SolutionDim>,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    DefaultAllocator:
        TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim> + DimAllocator<T, F::OutputDim>,
{
    fn solution_dim(&self) -> usize {
        SolutionDim::dim()
    }

    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(output, element_index)
    }
}

impl<'a, T, F, SolutionDim, Space, QTable> ElementScalarAssembler<T>
    for ElementIntegralVolumeAssembler<'a, T, F, SolutionDim, Space, QTable>
where
    T: Real,
    // TODO: See comment in impl for ElementConnectivityAssembler. Here we should ideally have Space::GeometryDim
    F: UGradFunction<T, Space::ReferenceDim, SolutionDim>,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator:
        TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim> + DimAllocator<T, F::OutputDim>,
{
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T> {
        let n = self.element_node_count(element_index);
        let s = self.solution_dim();
        let element_ndof = n * s;
        let integral = with_thread_local_workspace(
            &WORKSPACE,
            |workspace: &mut ElementIntegralAssemblerWorkspace<T, Space::ReferenceDim>| {
                workspace
                    .quadrature_buffer
                    .populate_element_weights_and_points_from_table(element_index, self.qtable);
                workspace
                    .local_interpolation_weights
                    .resize_vertically_mut(element_ndof, T::zero());
                workspace.nodes.resize(n, usize::MAX);
                self.populate_element_nodes(&mut workspace.nodes, element_index);
                let u_local = &mut workspace.local_interpolation_weights;
                let quadrature = workspace.quadrature_buffer.weights_and_points();
                gather_global_to_local(&self.u, &mut *u_local, &workspace.nodes, s);
                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                integrate_over_volume_element(
                    &self.integrand,
                    &element,
                    quadrature,
                    u_local,
                    &mut workspace.integration_workspace,
                )
            },
        )
        .map_err(|err| match err {
            // TODO: Handle this better? Alternatively we could make the integral "work"
            // since a singular Jacobian also means that the volume form is 0,
            // so the integral vanishes in some sense
            IntegrationFailure::SingularJacobian => {
                eyre!("Failed to compute integral due to singular Jacobian")
            }
        })?;
        Ok(integral[0])
    }
}

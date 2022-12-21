//! Functionality for error estimation.
use crate::allocators::{BiDimAllocator, TriDimAllocator};
use crate::assembly::global::assemble_scalar;
use crate::assembly::local::QuadratureTable;
use crate::element::VolumetricFiniteElement;
use crate::integrate::{integrate_over_element, integrate_over_volume_element, ElementIntegralAssemblerBuilder, IntegrationWorkspace, UFunction, FnFunction, UGradFunction};
use crate::integrate::dependency::DependsOnGrad;
use crate::nalgebra::DVectorSlice;
use crate::nalgebra::{DefaultAllocator, OPoint, OVector};
use crate::space::{InterpolateGradientInSpace, InterpolateInSpace, VolumetricFiniteElementSpace};
use crate::{Real, SmallDim};
use nalgebra::{OMatrix, Vector1, U1, Scalar};

/// A function $u: \mathbb{R}^d \rightarrow \mathbb{R}^s$ of the form $u(x)$ used to represent a reference solution.
///
/// The trait is implemented by closures with the appropriate signature. Finite element
/// spaces can be used to construct solution functions through the [`SpaceInterpolationFn`]
/// helper.
pub trait SolutionFunction<T, GeometryDim, SolutionDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>
{
    fn evaluate(&self, x: &OPoint<T, GeometryDim>) -> OVector<T, SolutionDim>;
}

impl<T, F, GeometryDim, SolutionDim> SolutionFunction<T, GeometryDim, SolutionDim> for F
where
    T: Scalar,
    F: Fn(&OPoint<T, GeometryDim>) -> OVector<T, SolutionDim>,
    GeometryDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>
{
    fn evaluate(&self, x: &OPoint<T, GeometryDim>) -> OVector<T, SolutionDim> {
        self(x)
    }
}

/// The gradient $\nabla u$ of a function $u: \mathbb{R}^d \rightarrow \mathbb{R}^s$ of the form $u(x)$ used to represent a reference solution.
///
/// The trait is implemented by closures with the appropriate signature. Finite element
/// spaces can be used to construct solution functions through the [`SpaceInterpolationFn`]
/// helper.
pub trait SolutionGradient<T, GeometryDim, SolutionDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>
{
    fn evaluate_grad(&self, x: &OPoint<T, GeometryDim>) -> OMatrix<T, GeometryDim, SolutionDim>;
}

impl<T, F, GeometryDim, SolutionDim> SolutionGradient<T, GeometryDim, SolutionDim> for F
where
    T: Scalar,
    F: Fn(&OPoint<T, GeometryDim>) -> OMatrix<T, GeometryDim, SolutionDim>,
    GeometryDim: SmallDim,
    SolutionDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>
{
    fn evaluate_grad(&self, x: &OPoint<T, GeometryDim>) -> OMatrix<T, GeometryDim, SolutionDim> {
        self(x)
    }
}

/// Interpret an [interpolating finite element space](crate::space::InterpolateInSpace)
/// and associated interpolation weights as a [`SolutionFunction`].
///
/// If the space also implements
/// [`InterpolateGradientsInSpace`](crate::space::InterpolateGradientInSpace),
/// then an implementation of [`SolutionGradient`] is also provided.
pub struct SpaceInterpolationFn<'a, Space, Weights>(pub &'a Space, pub Weights);

impl<'a, T, Space, Weights, SolutionDim> SolutionFunction<T, Space::GeometryDim, SolutionDim> for SpaceInterpolationFn<'a, Space, Weights>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: InterpolateInSpace<T, SolutionDim>,
    Weights: Copy + Into<DVectorSlice<'a, T>>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>
{
    fn evaluate(&self, x: &OPoint<T, Space::GeometryDim>) -> OVector<T, SolutionDim> {
        self.0.interpolate_at_point(x, self.1.into())
    }
}

impl<'a, T, Space, Weights, SolutionDim> SolutionGradient<T, Space::GeometryDim, SolutionDim> for SpaceInterpolationFn<'a, Space, Weights>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: InterpolateGradientInSpace<T, SolutionDim>,
    Weights: Copy + Into<DVectorSlice<'a, T>>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>
{
    fn evaluate_grad(&self, x: &OPoint<T, Space::GeometryDim>) -> OMatrix<T, Space::GeometryDim, SolutionDim> {
        self.0.interpolate_gradient_at_point(x, self.1.into())
    }
}

/// Estimate the squared $L^2$ error $\norm{u_h - u}^2_{L^2}$ on the given element with the given basis
/// weights and quadrature points.
///
/// # Panics
///
/// Panics if the basis buffer does not have the length $n$, where $n$ is the number of nodes
/// in the element.
#[allow(non_snake_case)]
pub fn estimate_element_L2_error_squared<T, Element, SolutionDim>(
    element: &Element,
    u: &impl SolutionFunction<T, Element::GeometryDim, SolutionDim>,
    u_h_element: DVectorSlice<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    workspace: &mut IntegrationWorkspace<T>,
) -> T
where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    SolutionDim: SmallDim,
    DefaultAllocator: TriDimAllocator<T, Element::GeometryDim, Element::ReferenceDim, SolutionDim>,
{
    let n = element.num_nodes();
    assert_eq!(u_h_element.len(), n * SolutionDim::dim());
    let result_as_vector = integrate_over_element(
        &make_L2_error_squared_integrand(u),
        element,
        (quadrature_weights, quadrature_points),
        u_h_element,
        workspace,
    );

    // Result is a 1-vector, we want to return a scalar
    result_as_vector[0]
}

/// Estimate the squared $H^1$ *seminorm* error $\seminorm{u_h - u}^2_{H^1}$ on the given element with the given basis
/// weights and quadrature points.
///
/// # Panics
///
/// Panics if the basis buffer does not have the length $n$, where $n$ is the number of nodes
/// in the element.
#[allow(non_snake_case)]
pub fn estimate_element_H1_seminorm_error_squared<T, Element, SolutionDim>(
    element: &Element,
    u_grad: &impl SolutionGradient<T, Element::GeometryDim, SolutionDim>,
    u_h_element: DVectorSlice<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    workspace: &mut IntegrationWorkspace<T>,
) -> T
where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    SolutionDim: SmallDim,
    DefaultAllocator: TriDimAllocator<T, Element::GeometryDim, Element::ReferenceDim, SolutionDim>,
{
    let n = element.num_nodes();
    assert_eq!(u_h_element.len(), n * SolutionDim::dim());
    let result_as_vector: Vector1<T> = integrate_over_volume_element(
        &make_H1_seminorm_error_squared_integrand(u_grad),
        element,
        (quadrature_weights, quadrature_points),
        u_h_element,
        workspace,
    )
    .expect("TODO: Handle the case where this might fail (due to e.g. singular Jacobian)");

    // Result is a 1-vector, we want to return a scalar
    result_as_vector[0]
}

/// Estimate the $H^1$ *seminorm* error $\seminorm{u_h - u}_{H^1}$ on the given element with the given basis
/// weights and quadrature points.
///
/// # Panics
///
/// Panics if the basis buffer does not have the length $n$, where $n$ is the number of nodes
/// in the element.
#[allow(non_snake_case)]
pub fn estimate_element_H1_seminorm_error<T, Element, SolutionDim>(
    element: &Element,
    u_grad: &impl SolutionGradient<T, Element::GeometryDim, SolutionDim>,
    u_h_element: DVectorSlice<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    workspace: &mut IntegrationWorkspace<T>,
) -> T
where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    SolutionDim: SmallDim,
    DefaultAllocator: TriDimAllocator<T, Element::GeometryDim, Element::ReferenceDim, SolutionDim>,
{
    estimate_element_H1_seminorm_error_squared(
        element,
        u_grad,
        u_h_element,
        quadrature_weights,
        quadrature_points,
        workspace,
    )
    .sqrt()
}

/// Estimate the $L^2$ error $\norm{u_h - u}_{L^2}$ on the given element with the given basis
/// weights and quadrature points.
///
/// # Panics
///
/// Panics if the basis buffer does not have the length $n$, where $n$ is the number of nodes
/// in the element.
#[allow(non_snake_case)]
pub fn estimate_element_L2_error<T, Element, SolutionDim>(
    element: &Element,
    u: &impl SolutionFunction<T, Element::GeometryDim, SolutionDim>,
    u_h_element: DVectorSlice<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    workspace: &mut IntegrationWorkspace<T>,
) -> T
where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    SolutionDim: SmallDim,
    DefaultAllocator: TriDimAllocator<T, Element::GeometryDim, Element::ReferenceDim, SolutionDim>,
{
    estimate_element_L2_error_squared(
        element,
        u,
        u_h_element,
        quadrature_weights,
        quadrature_points,
        workspace,
    )
    .sqrt()
}

#[allow(non_snake_case)]
fn make_L2_error_squared_integrand<'a, T, SolutionDim, GeometryDim>(
    u: &'a (impl SolutionFunction<T, GeometryDim, SolutionDim> + ?Sized)
) -> impl 'a + UFunction<T, GeometryDim, SolutionDim, OutputDim=U1>
where
    T: Real,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    let function = move |x: &OPoint<T, GeometryDim>, u_h: &OVector<T, SolutionDim>| {
        let u_at_x = u.evaluate(&x);
        let error = u_h - u_at_x;
        Vector1::new(error.norm_squared())
    };
    FnFunction::new(function)
}

#[allow(non_snake_case)]
fn make_H1_seminorm_error_squared_integrand<'a, T, SolutionDim, GeometryDim>(
    u_grad: &'a impl SolutionGradient<T, GeometryDim, SolutionDim>
) -> impl 'a + UGradFunction<T, GeometryDim, SolutionDim, OutputDim = U1>
where
    T: Real,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    let function = move |x: &OPoint<T, GeometryDim>,
                         u_h_grad: &OMatrix<T, GeometryDim, SolutionDim>| {
        let u_grad_at_x = u_grad.evaluate_grad(&x);
        let error = u_h_grad - u_grad_at_x;
        Vector1::new(error.norm_squared())
    };
    FnFunction::new(function).with_dependencies::<DependsOnGrad>()
}

/// Estimate the squared $L^2$ error $\norm{u_h - u}^2_{L^2}$ on the given finite element space
/// with the given solution weights and quadrature table.
#[allow(non_snake_case)]
pub fn estimate_L2_error_squared<'a, T, SolutionDim, Space, QTable>(
    space: &Space,
    u: &(impl SolutionFunction<T, Space::GeometryDim, SolutionDim> + ?Sized),
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>,
{
    let assembler = ElementIntegralAssemblerBuilder::new()
        .with_space(space)
        .with_quadrature_table(qtable)
        .with_interpolation_weights(u_h.into())
        .with_integrand(make_L2_error_squared_integrand(u))
        .build_integrator();

    assemble_scalar(&assembler)
}

/// Estimate the $L^2$ error $\norm{u_h - u}_{L^2}$ on the given finite element space
/// with the given solution weights and quadrature table.
#[allow(non_snake_case)]
pub fn estimate_L2_error<'a, T, SolutionDim, Space, QTable>(
    space: &Space,
    u: &(impl SolutionFunction<T, Space::GeometryDim, SolutionDim> + ?Sized),
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>,
{
    Ok(estimate_L2_error_squared(space, u, u_h, qtable)?.sqrt())
}

/// Estimate the squared $H^1$ *seminorm* error $\| u_h - u \|^2_{H^1}$ on the given finite element space
/// with the given solution weights and quadrature table.
#[allow(non_snake_case)]
pub fn estimate_H1_seminorm_error_squared<'a, T, SolutionDim, Space, QTable>(
    space: &Space,
    u_grad: &impl SolutionGradient<T, Space::GeometryDim, SolutionDim>,
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>,
{
    let assembler = ElementIntegralAssemblerBuilder::new()
        .with_space(space)
        .with_quadrature_table(qtable)
        .with_interpolation_weights(u_h.into())
        .with_integrand(make_H1_seminorm_error_squared_integrand(u_grad))
        .build_volume_integrator();

    assemble_scalar(&assembler)
}

/// Estimate the squared $H^1$ *seminorm* error $\|u_h - u \|^2_{H^1}$ on the given finite element space
/// with the given solution weights and quadrature table.
#[allow(non_snake_case)]
pub fn estimate_H1_seminorm_error<'a, T, SolutionDim, Space, QTable>(
    space: &Space,
    u_grad: &impl SolutionGradient<T, Space::GeometryDim, SolutionDim>,
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>,
{
    estimate_H1_seminorm_error_squared(space, u_grad, u_h, qtable).map(|err2| err2.sqrt())
}

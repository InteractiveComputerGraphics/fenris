//! Functionality for error estimation.
use crate::allocators::{BiDimAllocator, TriDimAllocator};
use crate::assembly::global::assemble_scalar;
use crate::assembly::local::QuadratureTable;
use crate::element::VolumetricFiniteElement;
use crate::integrate::{integrate_over_element, integrate_over_volume_element, ElementIntegralAssemblerBuilder, IntegrationWorkspace, UFunction, FnFunction, UGradFunction};
use crate::integrate::dependency::DependsOnGrad;
use crate::nalgebra::DVectorSlice;
use crate::nalgebra::{DefaultAllocator, OPoint, OVector};
use crate::space::VolumetricFiniteElementSpace;
use crate::{Real, SmallDim};
use nalgebra::{OMatrix, Vector1, U1};

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
    u: impl Fn(&OPoint<T, Element::GeometryDim>) -> OVector<T, SolutionDim>,
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
    u_grad: impl Fn(&OPoint<T, Element::GeometryDim>) -> OMatrix<T, Element::GeometryDim, SolutionDim>,
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
    u_grad: impl Fn(&OPoint<T, Element::GeometryDim>) -> OMatrix<T, Element::GeometryDim, SolutionDim>,
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
    u: impl Fn(&OPoint<T, Element::GeometryDim>) -> OVector<T, SolutionDim>,
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
    u: impl 'a + Fn(&OPoint<T, GeometryDim>) -> OVector<T, SolutionDim>,
) -> impl 'a + UFunction<T, GeometryDim, SolutionDim, OutputDim=U1>
where
    T: Real,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
{
    let function = move |x: &OPoint<T, GeometryDim>, u_h: &OVector<T, SolutionDim>| {
        let u_at_x = u(&x);
        let error = u_h - u_at_x;
        Vector1::new(error.norm_squared())
    };
    FnFunction::new(function)
}

#[allow(non_snake_case)]
fn make_H1_seminorm_error_squared_integrand<'a, T, SolutionDim, GeometryDim>(
    u_grad: impl 'a + Fn(&OPoint<T, GeometryDim>) -> OMatrix<T, GeometryDim, SolutionDim>,
) -> impl UGradFunction<T, GeometryDim, SolutionDim, OutputDim = U1>
where
    T: Real,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, SolutionDim, GeometryDim>,
{
    let function = move |x: &OPoint<T, GeometryDim>,
                         // _u_h: &OVector<T, SolutionDim>,
                         u_h_grad: &OMatrix<T, GeometryDim, SolutionDim>| {
        let u_grad_at_x = u_grad(&x);
        let error = u_h_grad - u_grad_at_x;
        Vector1::new(error.norm_squared())
    };
    FnFunction::new(function).with_dependencies::<DependsOnGrad>()
}

/// Estimate the squared $L^2$ error $\norm{u_h - u}^2_{L^2}$ on the given finite element space
/// with the given solution weights and quadrature table.
#[allow(non_snake_case)]
pub fn estimate_L2_error_squared<'a, T, Space, SolutionDim, QTable>(
    space: &Space,
    u: impl Fn(&OPoint<T, Space::GeometryDim>) -> OVector<T, SolutionDim>,
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, SolutionDim, Space::GeometryDim, Space::ReferenceDim>,
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
pub fn estimate_L2_error<'a, T, Space, SolutionDim, QTable>(
    space: &Space,
    u: impl Fn(&OPoint<T, Space::GeometryDim>) -> OVector<T, SolutionDim>,
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, SolutionDim, Space::GeometryDim, Space::ReferenceDim>,
{
    Ok(estimate_L2_error_squared(space, u, u_h, qtable)?.sqrt())
}

/// Estimate the squared $H^1$ *seminorm* error $\| u_h - u \|^2_{H^1}$ on the given finite element space
/// with the given solution weights and quadrature table.
#[allow(non_snake_case)]
pub fn estimate_H1_seminorm_error_squared<'a, T, Space, SolutionDim, QTable>(
    space: &Space,
    u_grad: impl Fn(&OPoint<T, Space::GeometryDim>) -> OMatrix<T, Space::GeometryDim, SolutionDim>,
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, SolutionDim, Space::GeometryDim, Space::ReferenceDim>,
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
pub fn estimate_H1_seminorm_error<'a, T, Space, SolutionDim, QTable>(
    space: &Space,
    u_grad: impl Fn(&OPoint<T, Space::GeometryDim>) -> OMatrix<T, Space::GeometryDim, SolutionDim>,
    u_h: impl Into<DVectorSlice<'a, T>>,
    qtable: &QTable,
) -> eyre::Result<T>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator: TriDimAllocator<T, SolutionDim, Space::GeometryDim, Space::ReferenceDim>,
{
    estimate_H1_seminorm_error_squared(space, u_grad, u_h, qtable).map(|err2| err2.sqrt())
}

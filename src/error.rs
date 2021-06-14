//! Functionality for error estimation.

use crate::allocators::VolumeFiniteElementAllocator;
use crate::element::{FiniteElement, MatrixSlice};
use crate::quadrature::Quadrature;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimMin, DimName, Dynamic, Point, RealField, Scalar, VectorN, U1, VectorSliceN};

#[derive(Debug, Clone)]
pub struct ErrorWorkspace<T: Scalar> {
    // Intermediate buffers used in computation
    basis: Vec<T>,
    basis_tr: Vec<T>
}

impl<T: RealField> Default for ErrorWorkspace<T> {
    fn default() -> Self {
        Self {
            basis: Vec::new(),
            basis_tr: Vec::new(),
        }
    }
}

/// Estimate the squared L^2 error of `u_h - u` on the given element with the given basis
/// weights and quadrature points.
///
/// `u(x, i)` represents the value of `u` at physical coordinate `x`. `i` is the index of the
/// quadrature point.
///
/// More precisely, estimate the integral of `dot(u_h - u, u_h - u)`, where `u_h = u_i N_i`,
/// with `u_i` the `i`-th column in `u` denoting the `m`-dimensional weight associated with node `i`,
/// and `N_i` is the basis function associated with node `i`.
#[allow(non_snake_case)]
pub fn estimate_element_L2_error_squared<T, SolutionDim, GeometryDim, Element>(
    workspace: &mut ErrorWorkspace<T>,
    element: &Element,
    u: impl Fn(&Point<T, GeometryDim>, usize) -> VectorN<T, SolutionDim>,
    u_weights: MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: impl Quadrature<T, GeometryDim>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator:
        VolumeFiniteElementAllocator<T, Element::GeometryDim> + Allocator<T, SolutionDim, U1>,
{
    let weights = quadrature.weights();
    let points = quadrature.points();

    use itertools::izip;
    let n = element.num_nodes();

    workspace
        .basis
        .resize(n, T::zero());
    workspace
        .basis_tr
        .resize(n, T::zero());

    // TODO: There is no reason to store the transposed here
    let mut result = T::zero();
    for (i, (w, xi)) in izip!(weights, points).enumerate() {
        let x = element.map_reference_coords(xi);
        let j = element.reference_jacobian(xi);
        element.populate_basis(&mut workspace.basis, xi);
        let basis_transposed = VectorSliceN::from_slice_generic(&workspace.basis, Dynamic::new(n), U1);

        let u_h = &u_weights * &basis_transposed;
        let u_at_x = u(&Point::from(x), i);
        let error = u_h - u_at_x;
        let error2 = error.dot(&error);
        result += error2 * *w * j.determinant().abs();
    }
    result
}

#[allow(non_snake_case)]
pub fn estimate_element_L2_error<T, SolutionDim, GeometryDim, Element>(
    workspace: &mut ErrorWorkspace<T>,
    element: &Element,
    u: impl Fn(&Point<T, GeometryDim>, usize) -> VectorN<T, SolutionDim>,
    u_weights: MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: impl Quadrature<T, GeometryDim>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    SolutionDim: DimName,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    DefaultAllocator:
        VolumeFiniteElementAllocator<T, Element::GeometryDim> + Allocator<T, SolutionDim, U1>,
{
    estimate_element_L2_error_squared(workspace, element, u, u_weights, quadrature).sqrt()
}

/// Estimate the squared L^2 norm on the given element with the given basis weights and quadrature
/// points.
///
/// More precisely, compute the integral of `dot(u_h, u_h)`, where `u_h = u_i N_i`, with `u_i`,
/// the `i`-th column in `u`, denoting the `m`-dimensional weight associated with node `i`,
/// and `N_i` is the basis function associated with node `i`.
#[allow(non_snake_case)]
pub fn estimate_element_L2_norm_squared<T, SolutionDim, GeometryDim, Element>(
    workspace: &mut ErrorWorkspace<T>,
    element: &Element,
    u_weights: MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: impl Quadrature<T, GeometryDim>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator:
        VolumeFiniteElementAllocator<T, Element::GeometryDim> + Allocator<T, SolutionDim, U1>,
{
    estimate_element_L2_error_squared(
        workspace,
        element,
        |_, _| VectorN::<T, SolutionDim>::repeat(T::zero()),
        u_weights,
        quadrature,
    )
}

#[allow(non_snake_case)]
pub fn estimate_element_L2_norm<T, SolutionDim, GeometryDim, Element>(
    workspace: &mut ErrorWorkspace<T>,
    element: &Element,
    u: MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: impl Quadrature<T, GeometryDim>,
) -> T
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: VolumeFiniteElementAllocator<T, GeometryDim> + Allocator<T, SolutionDim, U1>,
{
    estimate_element_L2_norm_squared(workspace, element, u, quadrature).sqrt()
}

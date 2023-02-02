use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::connectivity::Connectivity;
use crate::nalgebra::MatrixViewMut;
use crate::{Real, SmallDim};
use fenris_geometry::AxisAlignedBoundingBox;
use fenris_optimize::newton::NewtonSettings;
use nalgebra::allocator::Allocator;
use nalgebra::OPoint;
use nalgebra::{DVectorView, DVectorViewMut, DimName, Dyn};
use nalgebra::{DefaultAllocator, DimMin, OMatrix, OVector, Scalar, U1};
use num::Zero;
use numeric_literals::replace_float_literals;
use std::error::Error;
use std::fmt::Debug;

mod hexahedron;
mod quadrilateral;
mod segment;
mod tetrahedron;
mod triangle;
pub use hexahedron::*;
pub use quadrilateral::*;
pub use segment::*;
pub use tetrahedron::*;
pub use triangle::*;

pub trait ReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: DimAllocator<T, Self::ReferenceDim>,
{
    type ReferenceDim: SmallDim;

    /// Returns the number of nodes in the element.
    fn num_nodes(&self) -> usize;

    /// Evaluates each basis function at the given reference coordinates. The result is given
    /// in a row vector where each entry is the value of the corresponding basis function.
    ///
    /// TODO: Document that it should panic if the result does not have exactly the correct
    /// number of columns (==nodes)
    fn populate_basis(&self, basis_values: &mut [T], reference_coords: &OPoint<T, Self::ReferenceDim>);

    /// Given nodal weights, construct a matrix whose columns are the
    /// gradients of each shape function in the element.
    fn populate_basis_gradients(
        &self,
        basis_gradients: MatrixViewMut<T, Self::ReferenceDim, Dyn>,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    );
}

/// Reference finite elements with a number of nodes fixed at compile-time.
///
/// This is essentially equivalent to the old ReferenceFiniteElement trait, and exists only
/// to make existing code written with the old API in mind still work without having to
/// immediately re-write everything. It will be removed once everything has been ported over
/// to the new API (`ReferenceFiniteElement`).
///
/// TODO: Remove this trait once all elements and tests have been ported over to
///       `ReferenceFiniteElement`
pub trait FixedNodesReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: DimAllocator<T, Self::ReferenceDim>
        + Allocator<T, U1, Self::NodalDim>
        + Allocator<T, Self::ReferenceDim, Self::NodalDim>,
{
    type ReferenceDim: SmallDim;
    type NodalDim: SmallDim;

    /// Evaluates each basis function at the given reference coordinates. The result is given
    /// in a row vector where each entry is the value of the corresponding basis function.
    fn evaluate_basis(&self, reference_coords: &OPoint<T, Self::ReferenceDim>) -> OMatrix<T, U1, Self::NodalDim>;

    /// Given nodal weights, construct a matrix whose columns are the
    /// gradients of each shape function in the element.
    fn gradients(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::ReferenceDim, Self::NodalDim>;
}

/// Implements `ReferenceFiniteElement` for any element that implements
/// `FixedNodesReferenceFiniteElement`.
///
/// This could be done with a blanket impl, but this prevents
/// us from implementing `ReferenceFiniteElement` in some generic contexts, so we instead
/// invoke this macro for each element that this applies to. This is a temporary solution
/// that we use because it would take some work reworking the tests in order to remove the
/// `FixedNodesReferenceFiniteElement` trait altogether.
macro_rules! impl_reference_finite_element_for_fixed {
    ($element:ty) => {
        impl<T> ReferenceFiniteElement<T> for $element
        where
            T: Scalar,
            $element: FixedNodesReferenceFiniteElement<T>,
            DefaultAllocator: BiDimAllocator<
                T,
                <$element as FixedNodesReferenceFiniteElement<T>>::NodalDim,
                <$element as FixedNodesReferenceFiniteElement<T>>::ReferenceDim,
            >,
        {
            type ReferenceDim = <Self as FixedNodesReferenceFiniteElement<T>>::ReferenceDim;

            fn num_nodes(&self) -> usize {
                use nalgebra::DimName;
                <Self as FixedNodesReferenceFiniteElement<T>>::NodalDim::dim()
            }

            fn populate_basis(&self, result: &mut [T], reference_coords: &OPoint<T, Self::ReferenceDim>) {
                let basis_values = <Self as crate::element::FixedNodesReferenceFiniteElement<T>>::evaluate_basis(
                    self,
                    reference_coords,
                );
                result.clone_from_slice(&basis_values.as_slice());
            }

            fn populate_basis_gradients(
                &self,
                mut result: nalgebra::MatrixViewMut<T, Self::ReferenceDim, nalgebra::Dyn>,
                reference_coords: &OPoint<T, Self::ReferenceDim>,
            ) {
                let gradients =
                    <Self as crate::element::FixedNodesReferenceFiniteElement<T>>::gradients(self, reference_coords);
                result.copy_from(&gradients);
            }
        }
    };
}

impl_reference_finite_element_for_fixed!(Tri3d2Element<T>);
impl_reference_finite_element_for_fixed!(Tri6d2Element<T>);
impl_reference_finite_element_for_fixed!(Quad4d2Element<T>);
impl_reference_finite_element_for_fixed!(Quad9d2Element<T>);
impl_reference_finite_element_for_fixed!(Segment2d1Element<T>);
impl_reference_finite_element_for_fixed!(Segment2d2Element<T>);
impl_reference_finite_element_for_fixed!(Tet4Element<T>);
impl_reference_finite_element_for_fixed!(Hex8Element<T>);
impl_reference_finite_element_for_fixed!(Hex27Element<T>);
impl_reference_finite_element_for_fixed!(Hex20Element<T>);
impl_reference_finite_element_for_fixed!(Tri3d3Element<T>);
impl_reference_finite_element_for_fixed!(Tet10Element<T>);
impl_reference_finite_element_for_fixed!(Tet20Element<T>);

pub trait FiniteElement<T>: ReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    type GeometryDim: SmallDim;

    /// Compute the Jacobian of the transformation from the reference element to the given
    /// element at the given reference coordinates.
    fn reference_jacobian(
        &self,
        reference_coords: &OPoint<T, Self::ReferenceDim>,
    ) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim>;

    /// Maps reference coordinates to physical coordinates in the element.
    fn map_reference_coords(&self, reference_coords: &OPoint<T, Self::ReferenceDim>) -> OPoint<T, Self::GeometryDim>;

    /// The diameter of the finite element.
    ///
    /// The diameter of a finite element is defined as the largest distance between any two
    /// points in the element, i.e.
    ///  h = min |x - y| for x, y in K
    /// where K is the element and h is the diameter.
    fn diameter(&self) -> T;
}

/// TODO: Do we *really* need the Debug bound?
pub trait ElementConnectivity<T>: Debug + Connectivity
where
    T: Scalar,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    type Element: FiniteElement<T, GeometryDim = Self::GeometryDim, ReferenceDim = Self::ReferenceDim>;
    type GeometryDim: SmallDim;
    type ReferenceDim: SmallDim;

    /// Returns the finite element associated with this connectivity.
    ///
    /// The vertices passed in should be the collection of *all* vertices in the mesh.
    fn element(&self, all_vertices: &[OPoint<T, Self::GeometryDim>]) -> Option<Self::Element>;

    /// TODO: Move this out of the trait itself?
    fn populate_element_variables<'a, SolutionDim>(
        &self,
        mut u_local: MatrixViewMut<T, SolutionDim, Dyn>,
        u_global: impl Into<DVectorView<'a, T>>,
    ) where
        T: Zero,
        SolutionDim: DimName,
    {
        let u_global = u_global.into();
        let indices = self.vertex_indices();
        let sol_dim = SolutionDim::dim();
        for (i_local, i_global) in indices.iter().enumerate() {
            u_local
                .index_mut((.., i_local))
                .copy_from(&u_global.index((sol_dim * i_global..sol_dim * i_global + sol_dim, ..)));
        }
    }
}

/// A finite element whose geometry dimension and reference dimension coincide.
pub trait VolumetricFiniteElement<T>: FiniteElement<T, ReferenceDim = <Self as FiniteElement<T>>::GeometryDim>
where
    T: Scalar,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

impl<T, E> VolumetricFiniteElement<T> for E
where
    T: Scalar,
    E: FiniteElement<T, ReferenceDim = <Self as FiniteElement<T>>::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

pub trait SurfaceFiniteElement<T>: FiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    /// Compute the normal at the point associated with the provided reference coordinate.
    fn normal(&self, xi: &OPoint<T, Self::ReferenceDim>) -> OVector<T, Self::GeometryDim>;
}

// TODO: Move these?
pub type ElementForConnectivity<T, Connectivity> = <Connectivity as ElementConnectivity<T>>::Element;

pub type ConnectivityGeometryDim<T, Conn> = <Conn as ElementConnectivity<T>>::GeometryDim;
pub type ConnectivityReferenceDim<T, Conn> = <Conn as ElementConnectivity<T>>::ReferenceDim;

pub type ElementGeometryDim<T, Element> = <Element as FiniteElement<T>>::GeometryDim;

/// Linear basis function on the interval [-1, 1].
///
///`alpha == -1` denotes the basis function associated with the node at `x == -1`,
/// and `alpha == 1` for `x == 1`.
#[replace_float_literals(T::from_f64(literal).unwrap())]
#[inline(always)]
fn phi_linear_1d<T>(alpha: T, xi: T) -> T
where
    T: Real,
{
    (1.0 + alpha * xi) / 2.0
}

/// Gradient for the linear basis function on the interval [-1, 1].
///
/// See `phi_linear_1d` for the meaning of `alpha`.
#[replace_float_literals(T::from_f64(literal).unwrap())]
#[inline(always)]
fn phi_linear_1d_grad<T>(alpha: T) -> T
where
    T: Real,
{
    alpha / 2.0
}

/// Quadratic basis function on the interval [-1, 1].
///
/// `alpha == -1` denotes the basis function associated with the node at `x == -1`,
/// `alpha == 0` denotes the basis function associated with the node at `x == 0`,
/// and `alpha == 1` for `x == 1`.
#[replace_float_literals(T::from_f64(literal).unwrap())]
#[inline(always)]
fn phi_quadratic_1d<T>(alpha: T, xi: T) -> T
where
    T: Real,
{
    // The compiler should hopefully be able to use constant propagation to
    // precompute all expressions involving constants and alpha
    let alpha2 = alpha * alpha;
    let xi2 = xi * xi;
    (3.0 / 2.0 * alpha2 - 1.0) * xi2 + 0.5 * alpha * xi + 1.0 - alpha2
}

/// Derivative of quadratic basis function on the interval [-1, 1].
///
/// `alpha == -1` denotes the basis function associated with the node at `x == -1`,
/// `alpha == 0` denotes the basis function associated with the node at `x == 0`,
/// and `alpha == 1` for `x == 1`.
#[replace_float_literals(T::from_f64(literal).unwrap())]
#[inline(always)]
fn phi_quadratic_1d_grad<T>(alpha: T, xi: T) -> T
where
    T: Real,
{
    // The compiler should hopefully be able to use constant propagation to
    // precompute all expressions involving constants and alpha
    let alpha2 = alpha * alpha;
    2.0 * (3.0 / 2.0 * alpha2 - 1.0) * xi + 0.5 * alpha
}

/// Maps physical coordinates `x` to reference coordinates `xi` by solving the equation
///  x - T(xi) = 0 using Newton's method.
///
pub fn map_physical_coordinates<T, Element, GeometryDim>(
    element: &Element,
    x: &OPoint<T, GeometryDim>,
) -> Result<OPoint<T, GeometryDim>, Box<dyn Error>>
where
    T: Real,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    use fenris_optimize::calculus::VectorFunctionBuilder;
    use fenris_optimize::newton::newton;

    let f = VectorFunctionBuilder::with_dimension(GeometryDim::dim())
        .with_function(|f, xi| {
            // Need to create stack-allocated xi
            let xi = OPoint::from(
                xi.generic_view((0, 0), (GeometryDim::name(), U1::name()))
                    .clone_owned(),
            );
            f.copy_from(&(element.map_reference_coords(&xi).coords - &x.coords));
        })
        .with_jacobian_solver(
            |sol: &mut DVectorViewMut<T>, xi: &DVectorView<T>, rhs: &DVectorView<T>| {
                let xi = OPoint::from(
                    xi.generic_view((0, 0), (GeometryDim::name(), U1::name()))
                        .clone_owned(),
                );
                let j = element.reference_jacobian(&xi);
                let lu = j.full_piv_lu();
                sol.copy_from(rhs);
                if lu.solve_mut(sol) {
                    Ok(())
                } else {
                    Err(Box::<dyn Error>::from(
                        "LU decomposition failed. Jacobian not invertible?",
                    ))
                }
            },
        );

    // We solve the equation T(xi) = x, i.e. we seek reference coords xi such that when
    // transformed to physical coordinates yield x. We note here that what Newton's method solves
    // is the system T(xi) - x = 0, which can be re-interpreted as finding xi such that
    //   T_trans(xi) = 0, with T_trans(xi) = T(xi) - x.
    // This means that we seek xi such that the translated transformation transforms xi to
    // the zero vector. Since x should be a point in the element, it follows that we can expect
    // the diameter of the element to give us a representative scale of the "size" of x,
    // so we can construct our convergence criterion as follows:
    //   ||T(x_i) - x|| <= eps * diameter
    // with eps some small constant.

    let settings = NewtonSettings {
        // Note: Max iterations is entirely random at this point. Should of course
        // be made configurable. TODO
        max_iterations: Some(20),
        // TODO: eps is here hard-coded without respect to the type T, so it will not be appropriate
        // across e.g. different floating point types. Fix this!
        tolerance: T::from_f64(1e-12).unwrap() * element.diameter(),
    };

    let mut xi = OVector::<T, GeometryDim>::zeros();
    let mut f_val = OVector::<T, GeometryDim>::zeros();
    let mut dx = OVector::<T, GeometryDim>::zeros();

    // Because we cannot prove to the compiler that the strides of `OVector<T, GeometryDim>`
    // are compatible (in a `DimEq` sense) without nasty additional trait bounds,
    // we first take slices of the vectors so that the stride is dynamic. At this point,
    // it is known that `DimEq<Dyn, U1>` works, so we can use it with `newton`,
    // `which expects `Into<DMatrixViewMut<T>>`.
    macro_rules! slice {
        ($e:expr) => {
            $e.generic_view_with_steps_mut((0, 0), (GeometryDim::name(), U1::name()), (0, 0))
        };
    }

    newton(f, &mut slice!(xi), &mut slice!(f_val), &mut slice!(dx), settings)?;

    Ok(OPoint::from(xi))
}

/// Projects physical coordinates `x` to reference coordinates `xi` by solving the equation
///  x - T(xi) = 0 using a generalized form of Newton's method.
///
/// Unlike `map_physical_coordinates`, this method is also applicable to e.g. surface finite
/// elements, in which the reference dimension and geometry dimension differ.
///
/// The method panics if `ReferenceDim` is greater than `GeometryDim`.
///
/// TODO: This method is totally misleading as is, because it does not take into account
/// the geometry of the reference element in reference coordinates, so it will happily
/// return points outside of the reference geometry.
#[allow(non_snake_case)]
pub fn project_physical_coordinates<T, Element>(
    element: &Element,
    x: &OPoint<T, Element::GeometryDim>,
) -> Result<OPoint<T, Element::ReferenceDim>, Box<dyn Error>>
where
    T: Real,
    Element: FiniteElement<T>,
    Element::ReferenceDim: DimName + DimMin<Element::ReferenceDim, Output = Element::ReferenceDim>,
    DefaultAllocator: BiDimAllocator<T, Element::GeometryDim, Element::ReferenceDim>,
{
    assert!(
        Element::ReferenceDim::dim() <= Element::GeometryDim::dim(),
        "ReferenceDim must be smaller or equal to GeometryDim."
    );

    // See comments in `map_physical_coordinates` for why this is a reasonable tolerance.
    let tolerance = T::from_f64(1e-12).unwrap() * element.diameter();

    // We wish to solve the system
    //  f(xi) - x = 0,
    // but xi and x have different dimensions. To overcome this difficulty, we use a modified
    // version of Newton's method in which we solve the normal equations for the Jacobian equation
    // instead of solving the Jacobian system directly (which is underdetermined in this case).
    //
    // Our stopping condition is based on the optimality condition for the
    // least-squares problem min || x - f(xi) ||, whose geometrical interpretation at the
    // minimum is exactly that of a projection onto the surface.

    let x = &x.coords;
    let mut xi = OPoint::<T, Element::ReferenceDim>::origin();
    let mut f = element.map_reference_coords(&xi).coords;
    let mut j = element.reference_jacobian(&xi);
    let mut jT = j.transpose();

    let mut iter = 0;
    // TODO: Do we need to alter the tolerance due to the jT term?
    while (&jT * (&f - x)).norm() > tolerance {
        let jTj = &jT * j;
        let lu = jTj.full_piv_lu();
        let rhs = -jT * (&f - x);

        if let Some(sol) = lu.solve(&rhs) {
            xi += sol;
        } else {
            return Err(Box::from(
                "LU decomposition failed. Normal equation for Jacobian not invertible?",
            ));
        }

        f = element.map_reference_coords(&xi).coords;
        j = element.reference_jacobian(&xi);
        jT = j.transpose();
        iter += 1;

        // TODO: Should better handle degenerate/problematic cases. For now we just want to
        // avoid infinite loops
        if iter > 1000 {
            eprintln!("Exceeded 1000 iterations for project_physical_coordinates");
        }
    }

    Ok(OPoint::from(xi))
}

/// The result of a [`ClosestPointInElement`] query.
#[derive(Debug, Clone, PartialEq)]
pub enum ClosestPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// The point is inside the element.
    InElement(OPoint<T, D>),
    /// The closest point in the element to the query point.
    ClosestPoint(OPoint<T, D>),
}

impl<T, D> ClosestPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn point(&self) -> &OPoint<T, D> {
        match self {
            ClosestPoint::InElement(point) | ClosestPoint::ClosestPoint(point) => point,
        }
    }
}

/// A finite element you can query for the closest point to an arbitrary point.
pub trait ClosestPointInElement<T: Scalar>: FiniteElement<T>
where
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    fn closest_point(&self, p: &OPoint<T, Self::GeometryDim>) -> ClosestPoint<T, Self::ReferenceDim>;
}

/// A finite element that can be queried for its bounding box.
pub trait BoundsForElement<T: Scalar>: FiniteElement<T>
where
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    fn element_bounds(&self) -> AxisAlignedBoundingBox<T, Self::GeometryDim>;
}

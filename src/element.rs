use nalgebra::allocator::Allocator;
use nalgebra::{
    distance, DVectorSlice, DVectorSliceMut, DimName, Dynamic, Matrix1x6, Matrix2x6, Matrix3,
    Matrix3x4, Point2, Point3, Vector3,
};
use nalgebra::{
    DefaultAllocator, DimMin, Matrix1x3, Matrix1x4, Matrix2, Matrix2x3, Matrix2x4, MatrixMN,
    RealField, Scalar, Vector2, VectorN, U1, U10, U2, U20, U27, U3, U4, U6, U8, U9,
};
use nalgebra::{Matrix3x2, Point};

use crate::connectivity::{
    Connectivity, Hex20Connectivity, Hex27Connectivity, Hex8Connectivity, Quad4d2Connectivity,
    Quad9d2Connectivity, Tet10Connectivity, Tet4Connectivity, Tri3d2Connectivity,
    Tri3d3Connectivity, Tri6d2Connectivity,
};
use crate::geometry::{
    ConcavePolygonError, ConvexPolygon, LineSegment2d, Quad2d, Triangle, Triangle2d, Triangle3d,
};

use itertools::Itertools;
use numeric_literals::replace_float_literals;
use std::convert::{TryFrom, TryInto};
use std::fmt::Debug;

use crate::optimize::newton::NewtonSettings;
use num::Zero;
use std::error::Error;

use crate::allocators::{
    FiniteElementAllocator, ReferenceFiniteElementAllocator, VolumeFiniteElementAllocator,
};
use crate::connectivity::Segment2d2Connectivity;
use crate::nalgebra::Point1;
use crate::SmallDim;

/// TODO: Contribute these defaults to `nalgebra`
pub type MatrixSlice<'a, T, R, C> = nalgebra::base::MatrixSlice<'a, T, R, C, U1, R>;
pub type MatrixSliceMut<'a, T, R, C> = nalgebra::base::MatrixSliceMut<'a, T, R, C, U1, R>;

pub trait ReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: ReferenceFiniteElementAllocator<T, Self::ReferenceDim>,
{
    type ReferenceDim: SmallDim;

    /// Returns the number of nodes in the element.
    fn num_nodes(&self) -> usize;

    /// Evaluates each basis function at the given reference coordinates. The result is given
    /// in a row vector where each entry is the value of the corresponding basis function.
    ///
    /// TODO: Document that it should panic if the result does not have exactly the correct
    /// number of columns (==nodes)
    fn populate_basis(
        &self,
        basis_values: &mut [T],
        reference_coords: &Point<T, Self::ReferenceDim>,
    );

    /// Given nodal weights, construct a matrix whose columns are the
    /// gradients of each shape function in the element.
    fn populate_basis_gradients(
        &self,
        basis_gradients: MatrixSliceMut<T, Self::ReferenceDim, Dynamic>,
        reference_coords: &Point<T, Self::ReferenceDim>,
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
    DefaultAllocator: ReferenceFiniteElementAllocator<T, Self::ReferenceDim>
        + Allocator<T, U1, Self::NodalDim>
        + Allocator<T, Self::ReferenceDim, Self::NodalDim>,
{
    type ReferenceDim: SmallDim;
    type NodalDim: SmallDim;

    /// Evaluates each basis function at the given reference coordinates. The result is given
    /// in a row vector where each entry is the value of the corresponding basis function.
    fn evaluate_basis(
        &self,
        reference_coords: &Point<T, Self::ReferenceDim>,
    ) -> MatrixMN<T, U1, Self::NodalDim>;

    /// Given nodal weights, construct a matrix whose columns are the
    /// gradients of each shape function in the element.
    fn gradients(
        &self,
        reference_coords: &Point<T, Self::ReferenceDim>,
    ) -> MatrixMN<T, Self::ReferenceDim, Self::NodalDim>;
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
            DefaultAllocator: ReferenceFiniteElementAllocator<
                    T,
                    <$element as FixedNodesReferenceFiniteElement<T>>::ReferenceDim,
                > + Allocator<T, U1, <$element as FixedNodesReferenceFiniteElement<T>>::NodalDim>
                + Allocator<
                    T,
                    <$element as FixedNodesReferenceFiniteElement<T>>::ReferenceDim,
                    <$element as FixedNodesReferenceFiniteElement<T>>::NodalDim,
                >,
        {
            type ReferenceDim = <Self as FixedNodesReferenceFiniteElement<T>>::ReferenceDim;

            fn num_nodes(&self) -> usize {
                <$element as FixedNodesReferenceFiniteElement<T>>::NodalDim::dim()
            }

            fn populate_basis(
                &self,
                result: &mut [T],
                reference_coords: &Point<T, Self::ReferenceDim>,
            ) {
                let basis_values =
                    <$element as FixedNodesReferenceFiniteElement<T>>::evaluate_basis(
                        self,
                        reference_coords,
                    );
                result.clone_from_slice(&basis_values.as_slice());
            }

            fn populate_basis_gradients(
                &self,
                mut result: MatrixSliceMut<T, Self::ReferenceDim, Dynamic>,
                reference_coords: &Point<T, Self::ReferenceDim>,
            ) {
                let gradients = <$element as FixedNodesReferenceFiniteElement<T>>::gradients(
                    self,
                    reference_coords,
                );
                result.copy_from(&gradients);
            }
        }
    };
}

pub trait FiniteElement<T>: ReferenceFiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    type GeometryDim: SmallDim;

    /// Compute the Jacobian of the transformation from the reference element to the given
    /// element at the given reference coordinates.
    fn reference_jacobian(
        &self,
        reference_coords: &Point<T, Self::ReferenceDim>,
    ) -> MatrixMN<T, Self::GeometryDim, Self::ReferenceDim>;

    /// Maps reference coordinates to physical coordinates in the element.
    fn map_reference_coords(
        &self,
        reference_coords: &Point<T, Self::ReferenceDim>,
    ) -> Point<T, Self::GeometryDim>;

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
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    type Element: FiniteElement<
        T,
        GeometryDim = Self::GeometryDim,
        ReferenceDim = Self::ReferenceDim,
    >;
    type GeometryDim: DimName;
    type ReferenceDim: DimName;

    /// Returns the finite element associated with this connectivity.
    ///
    /// The vertices passed in should be the collection of *all* vertices in the mesh.
    fn element(&self, vertices: &[Point<T, Self::GeometryDim>]) -> Option<Self::Element>;

    /// TODO: Move this out of the trait itself?
    fn populate_element_variables<'a, SolutionDim>(
        &self,
        mut u_local: MatrixSliceMut<T, SolutionDim, Dynamic>,
        u_global: impl Into<DVectorSlice<'a, T>>,
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
pub trait VolumetricFiniteElement<T>:
    FiniteElement<T, ReferenceDim = <Self as FiniteElement<T>>::GeometryDim>
where
    T: Scalar,
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

impl<T, E> VolumetricFiniteElement<T> for E
where
    T: Scalar,
    E: FiniteElement<T, ReferenceDim = <Self as FiniteElement<T>>::GeometryDim>,
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
}

pub trait SurfaceFiniteElement<T>: FiniteElement<T>
where
    T: Scalar,
    DefaultAllocator: FiniteElementAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    /// Compute the normal at the point associated with the provided reference coordinate.
    fn normal(&self, xi: &Point<T, Self::ReferenceDim>) -> VectorN<T, Self::GeometryDim>;
}

// TODO: Move these?
pub type ElementForConnectivity<T, Connectivity> =
    <Connectivity as ElementConnectivity<T>>::Element;

pub type ConnectivityGeometryDim<T, Conn> = <Conn as ElementConnectivity<T>>::GeometryDim;
pub type ConnectivityReferenceDim<T, Conn> = <Conn as ElementConnectivity<T>>::ReferenceDim;

pub type ElementGeometryDim<T, Element> = <Element as FiniteElement<T>>::GeometryDim;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Quad4d2Element<T>
where
    T: Scalar,
{
    vertices: [Point2<T>; 4],
}

impl<T> Quad4d2Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point2<T>; 4]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point2<T>; 4] {
        &self.vertices
    }
}

impl<T> From<Quad2d<T>> for Quad4d2Element<T>
where
    T: Scalar,
{
    fn from(quad: Quad2d<T>) -> Self {
        Self::from_vertices(quad.0)
    }
}

impl<T> Quad4d2Element<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point2::new(-1.0, -1.0),
            Point2::new(1.0, -1.0),
            Point2::new(1.0, 1.0),
            Point2::new(-1.0, 1.0),
        ])
    }
}

impl<T> TryFrom<Quad4d2Element<T>> for ConvexPolygon<T>
where
    T: RealField,
{
    type Error = ConcavePolygonError;

    fn try_from(value: Quad4d2Element<T>) -> Result<Self, Self::Error> {
        ConvexPolygon::try_from(Quad2d(value.vertices))
    }
}

impl<T> ElementConnectivity<T> for Quad4d2Connectivity
where
    T: RealField,
{
    type Element = Quad4d2Element<T>;
    type ReferenceDim = U2;
    type GeometryDim = U2;

    fn element(&self, vertices: &[Point2<T>]) -> Option<Self::Element> {
        let Self(indices) = self;
        let lookup_vertex = |local_index| vertices.get(indices[local_index]).cloned();

        Some(Quad4d2Element::from_vertices([
            lookup_vertex(0)?,
            lookup_vertex(1)?,
            lookup_vertex(2)?,
            lookup_vertex(3)?,
        ]))
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Quad4d2Element<T>
where
    T: RealField,
{
    type NodalDim = U4;
    type ReferenceDim = U2;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point2<T>) -> Matrix1x4<T> {
        // We define the shape functions as N_{alpha, beta} evaluated at xi such that
        //  N_{alpha, beta}([alpha, beta]) = 1
        // with alpha, beta = 1 or -1
        let phi = |alpha, beta, xi: &Point2<T>| (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) / 4.0;
        Matrix1x4::from_row_slice(&[
            phi(-1.0, -1.0, xi),
            phi( 1.0, -1.0, xi),
            phi( 1.0,  1.0, xi),
            phi(-1.0,  1.0, xi),
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point2<T>) -> Matrix2x4<T> {
        let phi_grad = |alpha, beta, xi: &Point2<T>|
            Vector2::new(
                alpha * (1.0 + beta * xi[1]) / 4.0,
                beta * (1.0 + alpha * xi[0]) / 4.0,
            );

        Matrix2x4::from_columns(&[
            phi_grad(-1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, xi),
            phi_grad(-1.0,  1.0, xi),
        ])
    }
}

impl_reference_finite_element_for_fixed!(Quad4d2Element<T>);

impl<T> FiniteElement<T> for Quad4d2Element<T>
where
    T: RealField,
{
    type GeometryDim = U2;

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point2<T>) -> Point2<T> {
        // TODO: Store this X matrix directly in Self?
        let X: Matrix2x4<T> = Matrix2x4::from_fn(|i, j| self.vertices[j][i]);
        let N = self.evaluate_basis(xi);
        Point::from(&X * &N.transpose())
    }

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix2<T> {
        // TODO: Avoid redundant computation of gradient matrix by
        // offering a function which simultaneously computes the gradient matrix and the
        // Jacobian
        let X: Matrix2x4<T> = Matrix2x4::from_fn(|i, j| self.vertices[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.vertices
            .iter()
            .tuple_combinations()
            .map(|(x, y)| distance(x, y))
            .fold(T::zero(), |a, b| a.max(b.clone()))
    }
}

/// A finite element representing linear basis functions on a triangle, in two dimensions.
///
/// The reference element is chosen to be the triangle defined by the corners
/// (-1, -1), (1, -1), (-1, 1). This perhaps unorthodox choice is due to the quadrature rules
/// we employ.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Tri3d2Element<T>
where
    T: Scalar,
{
    vertices: [Point2<T>; 3],
}

impl<T> Tri3d2Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point2<T>; 3]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point2<T>; 3] {
        &self.vertices
    }
}

impl<T> From<Triangle2d<T>> for Tri3d2Element<T>
where
    T: Scalar,
{
    fn from(triangle: Triangle2d<T>) -> Self {
        Self::from_vertices(triangle.0)
    }
}

impl<T> Tri3d2Element<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point2::new(-1.0, -1.0),
            Point2::new(1.0, -1.0),
            Point2::new(-1.0, 1.0),
        ])
    }
}

impl<T> ElementConnectivity<T> for Tri3d2Connectivity
where
    T: RealField,
{
    type Element = Tri3d2Element<T>;
    type ReferenceDim = U2;
    type GeometryDim = U2;

    fn element(&self, vertices: &[Point2<T>]) -> Option<Self::Element> {
        let Self(indices) = self;
        let lookup_vertex = |local_index| vertices.get(indices[local_index]).cloned();

        Some(Tri3d2Element::from_vertices([
            lookup_vertex(0)?,
            lookup_vertex(1)?,
            lookup_vertex(2)?,
        ]))
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Tri3d2Element<T>
where
    T: RealField,
{
    type NodalDim = U3;
    type ReferenceDim = U2;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point2<T>) -> Matrix1x3<T> {
        Matrix1x3::from_row_slice(&[
            -0.5 * xi.x - 0.5 * xi.y,
            0.5 * xi.x + 0.5,
            0.5 * xi.y + 0.5
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, _: &Point2<T>) -> Matrix2x3<T> {
        // TODO: Precompute gradients
        Matrix2x3::from_columns(&[
            Vector2::new(-0.5, -0.5),
            Vector2::new(0.5, 0.0),
            Vector2::new(0.0, 0.5)
        ])
    }
}

impl_reference_finite_element_for_fixed!(Tri3d2Element<T>);

impl<T> FiniteElement<T> for Tri3d2Element<T>
where
    T: RealField,
{
    type GeometryDim = U2;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix2<T> {
        let X: Matrix2x3<T> = Matrix2x3::from_fn(|i, j| self.vertices[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point2<T>) -> Point2<T> {
        // TODO: Store this X matrix directly in Self...?
        let X: Matrix2x3<T> = Matrix2x3::from_fn(|i, j| self.vertices[j][i]);
        let N = self.evaluate_basis(xi);
        Point::from(&X * &N.transpose())
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.vertices
            .iter()
            .tuple_combinations()
            .map(|(x, y)| distance(x, y))
            .fold(T::zero(), |a, b| a.max(b.clone()))
    }
}

/// A finite element representing quadratic basis functions on a triangle, in two dimensions.
///
/// The reference element is chosen to be the triangle defined by the corners
/// (-1, -1), (1, -1), (-1, 1). This perhaps unorthodox choice is due to the quadrature rules
/// we employ.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Tri6d2Element<T>
where
    T: Scalar,
{
    vertices: [Point2<T>; 6],
    tri3: Tri3d2Element<T>,
}

impl<T> Tri6d2Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point2<T>; 6]) -> Self {
        let v = &vertices;
        let tri = [v[0].clone(), v[1].clone(), v[2].clone()];
        Self {
            vertices,
            tri3: Tri3d2Element::from_vertices(tri),
        }
    }

    pub fn vertices(&self) -> &[Point2<T>; 6] {
        &self.vertices
    }
}

impl<'a, T> From<&'a Tri3d2Element<T>> for Tri6d2Element<T>
where
    T: RealField,
{
    // TODO: Test this
    fn from(tri3: &'a Tri3d2Element<T>) -> Self {
        let midpoint =
            |a: &Point2<_>, b: &Point2<_>| LineSegment2d::new(a.clone(), b.clone()).midpoint();

        let tri3_v = &tri3.vertices;
        let mut vertices = [Point2::origin(); 6];
        vertices[0..=2].clone_from_slice(tri3_v);
        vertices[3] = midpoint(&tri3_v[0], &tri3_v[1]);
        vertices[4] = midpoint(&tri3_v[1], &tri3_v[2]);
        vertices[5] = midpoint(&tri3_v[2], &tri3_v[0]);

        Self::from_vertices(vertices)
    }
}

impl<'a, T> From<Tri3d2Element<T>> for Tri6d2Element<T>
where
    T: RealField,
{
    fn from(tri3: Tri3d2Element<T>) -> Self {
        Self::from(&tri3)
    }
}

impl<T> Tri6d2Element<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self {
            vertices: [
                Point2::new(-1.0, -1.0),
                Point2::new(1.0, -1.0),
                Point2::new(-1.0, 1.0),
                Point2::new(0.0, -1.0),
                Point2::new(0.0, 0.0),
                Point2::new(-1.0, 0.0),
            ],
            tri3: Tri3d2Element::reference(),
        }
    }
}

impl<T> ElementConnectivity<T> for Tri6d2Connectivity
where
    T: RealField,
{
    type Element = Tri6d2Element<T>;
    type ReferenceDim = U2;
    type GeometryDim = U2;

    fn element(&self, vertices: &[Point2<T>]) -> Option<Self::Element> {
        let Self(indices) = self;
        let lookup_vertex = |local_index| vertices.get(indices[local_index]).cloned();

        Some(Tri6d2Element::from_vertices([
            lookup_vertex(0)?,
            lookup_vertex(1)?,
            lookup_vertex(2)?,
            lookup_vertex(3)?,
            lookup_vertex(4)?,
            lookup_vertex(5)?,
        ]))
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Tri6d2Element<T>
where
    T: RealField,
{
    type NodalDim = U6;
    type ReferenceDim = U2;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point2<T>) -> Matrix1x6<T> {
        // We express the basis functions of Tri6 as products of
        // the Tri3 basis functions.
        let psi = self.tri3.evaluate_basis(xi);
        Matrix1x6::from_row_slice(&[
            psi[0] * (2.0 * psi[0] - 1.0),
            psi[1] * (2.0 * psi[1] - 1.0),
            psi[2] * (2.0 * psi[2] - 1.0),
            4.0 * psi[0] * psi[1],
            4.0 * psi[1] * psi[2],
            4.0 * psi[0] * psi[2],
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point2<T>) -> Matrix2x6<T> {
        // Similarly to `evaluate_basis`, we may implement the gradients of
        // Tri6 with the help of the function values and gradients of Tri3
        let psi = self.tri3.evaluate_basis(xi);
        let g = self.tri3.gradients(xi);

        // Gradient of vertex node i
        let vertex_gradient = |i| g.index((.., i)) * (4.0 * psi[i] - 1.0);

        // Gradient of edge node on the edge between vertex i and j
        let edge_gradient = |i, j|
            g.index((.., i)) * (4.0 * psi[j]) + g.index((.., j)) * (4.0 * psi[i]);

        Matrix2x6::from_columns(&[
            vertex_gradient(0),
            vertex_gradient(1),
            vertex_gradient(2),
            edge_gradient(0, 1),
            edge_gradient(1, 2),
            edge_gradient(0, 2)
        ])
    }
}

impl_reference_finite_element_for_fixed!(Tri6d2Element<T>);

impl<T> FiniteElement<T> for Tri6d2Element<T>
where
    T: RealField,
{
    type GeometryDim = U2;

    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix2<T> {
        self.tri3.reference_jacobian(xi)
    }

    fn map_reference_coords(&self, xi: &Point2<T>) -> Point2<T> {
        self.tri3.map_reference_coords(xi)
    }

    fn diameter(&self) -> T {
        self.tri3.diameter()
    }
}

/// A finite element representing quadratic basis functions on a quad, in two dimensions.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Quad9d2Element<T>
where
    T: Scalar,
{
    vertices: [Point2<T>; 9],
    // Store quad for easy computation of Jacobians and mapping reference coordinates
    quad: Quad4d2Element<T>,
}

impl<T> Quad9d2Element<T>
where
    T: Scalar,
{
    fn from_vertices(vertices: [Point2<T>; 9]) -> Self {
        let v = &vertices;
        let quad = [v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone()];
        Self {
            vertices,
            quad: Quad4d2Element::from_vertices(quad),
        }
    }

    pub fn vertices(&self) -> &[Point2<T>; 9] {
        &self.vertices
    }
}

impl<'a, T> From<&'a Quad4d2Element<T>> for Quad9d2Element<T>
where
    T: RealField,
{
    fn from(quad4: &'a Quad4d2Element<T>) -> Self {
        let midpoint =
            |a: &Point2<_>, b: &Point2<_>| LineSegment2d::new(a.clone(), b.clone()).midpoint();

        let quad4_v = &quad4.vertices;
        let mut vertices = [Point2::origin(); 9];
        vertices[0..=3].clone_from_slice(quad4_v);
        vertices[4] = midpoint(&quad4_v[0], &quad4_v[1]);
        vertices[5] = midpoint(&quad4_v[1], &quad4_v[2]);
        vertices[6] = midpoint(&quad4_v[2], &quad4_v[3]);
        vertices[7] = midpoint(&quad4_v[3], &quad4_v[0]);

        // Vertex 8 is in the middle of the element, i.e. the midpoint
        // between 5 and 7 or 4 and 6 (arbitrary choice)
        vertices[8] = midpoint(&vertices[4], &vertices[6]);

        Self::from_vertices(vertices)
    }
}

impl<'a, T> From<Quad4d2Element<T>> for Quad9d2Element<T>
where
    T: RealField,
{
    fn from(quad4: Quad4d2Element<T>) -> Self {
        Self::from(&quad4)
    }
}

impl<T> Quad9d2Element<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn reference() -> Self {
        let p = |x, y| Point2::new(x, y);
        Self::from_vertices([
            p(-1.0, -1.0),
            p(1.0, -1.0),
            p(1.0, 1.0),
            p(-1.0, 1.0),
            p(0.0, -1.0),
            p(1.0, 0.0),
            p(0.0, 1.0),
            p(-1.0, 0.0),
            p(0.0, 0.0),
        ])
    }
}

#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
fn quad9_phi_1d<T>(alpha: T, xi: T) -> T
where
    T: RealField,
{
    let alpha2 = alpha * alpha;
    let a = (3.0 / 2.0) * alpha2 - 1.0;
    let b = alpha / 2.0;
    let c = 1.0 - alpha2;
    a * xi * xi + b * xi + c
}

impl<T> FixedNodesReferenceFiniteElement<T> for Quad9d2Element<T>
where
    T: RealField,
{
    type ReferenceDim = U2;
    type NodalDim = U9;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point2<T>) -> MatrixMN<T, U1, U9> {
        // We define the shape functions as N_{alpha, beta} evaluated at xi such that
        //  N_{alpha, beta}([alpha, beta]) = 1
        // with alpha, beta = 1 or -1.
        // Furthermore, the basis functions are separable in the sense that we may write
        //  N_{alpha, beta) (xi, eta) = N_alpha(xi) * N_beta(eta).

        let phi_1d = quad9_phi_1d;
        let phi = |alpha, beta, xi: &Point2<T>| {
            let x = xi[0];
            let y = xi[1];
            phi_1d(alpha, x) * phi_1d(beta, y)
        };

        MatrixMN::<T, U1, U9>::from_row_slice(&[
            phi(-1.0, -1.0, xi),
            phi( 1.0, -1.0, xi),
            phi( 1.0,  1.0, xi),
            phi(-1.0,  1.0, xi),
            phi( 0.0, -1.0, xi),
            phi( 1.0,  0.0, xi),
            phi( 0.0,  1.0, xi),
            phi(-1.0,  0.0, xi),
            phi( 0.0,  0.0, xi)
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point2<T>) -> MatrixMN<T, U2, U9> {
        // See the implementation of `evaluate_basis` for a definition of the basis functions.
        let phi_1d = quad9_phi_1d::<T>;
        let phi_grad_1d = |alpha, xi| {
            let alpha2 = alpha * alpha;
            let a = (3.0 / 2.0) * alpha2 - 1.0;
            let b = alpha / 2.0;
            2.0 * a * xi + b
        };

        let phi_grad = |alpha, beta, xi: &Point2<T>| {
            let x = xi[0];
            let y = xi[1];
            Vector2::new(
                phi_1d(beta, y) * phi_grad_1d(alpha, x),
                phi_1d(alpha, x) * phi_grad_1d(beta, y)
            )
        };

        MatrixMN::<T, U2, U9>::from_columns(&[
            phi_grad(-1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, xi),
            phi_grad(-1.0,  1.0, xi),
            phi_grad( 0.0, -1.0, xi),
            phi_grad( 1.0,  0.0, xi),
            phi_grad( 0.0,  1.0, xi),
            phi_grad(-1.0,  0.0, xi),
            phi_grad( 0.0,  0.0, xi)
        ])
    }
}

impl_reference_finite_element_for_fixed!(Quad9d2Element<T>);

impl<T> FiniteElement<T> for Quad9d2Element<T>
where
    T: RealField,
{
    type GeometryDim = U2;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix2<T> {
        self.quad.reference_jacobian(xi)
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point2<T>) -> Point2<T> {
        self.quad.map_reference_coords(xi)
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.quad.diameter()
    }
}

impl<T> ElementConnectivity<T> for Quad9d2Connectivity
where
    T: RealField,
{
    type Element = Quad9d2Element<T>;
    type ReferenceDim = U2;
    type GeometryDim = U2;

    fn element(&self, vertices: &[Point2<T>]) -> Option<Self::Element> {
        let Self(indices) = self;
        let mut vertices_array: [Point2<T>; 9] = [Point2::origin(); 9];

        for (v, global_index) in vertices_array.iter_mut().zip(indices) {
            *v = vertices[*global_index];
        }

        Some(Quad9d2Element::from_vertices(vertices_array))
    }
}

impl<T> TryFrom<Quad9d2Element<T>> for ConvexPolygon<T>
where
    T: RealField,
{
    type Error = ConcavePolygonError;

    fn try_from(value: Quad9d2Element<T>) -> Result<Self, Self::Error> {
        ConvexPolygon::try_from(value.quad)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// A surface element embedded in two dimensions.
pub struct Segment2d2Element<T>
where
    T: Scalar,
{
    segment: LineSegment2d<T>,
}

impl<T> From<LineSegment2d<T>> for Segment2d2Element<T>
where
    T: Scalar,
{
    fn from(segment: LineSegment2d<T>) -> Self {
        Self { segment }
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Segment2d2Element<T>
where
    T: RealField,
{
    type NodalDim = U2;
    type ReferenceDim = U1;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point1<T>) -> MatrixMN<T, U1, U2> {
        // xi is a scalar
        let xi = xi.x;
        let phi_1 = (1.0 - xi) / 2.0;
        let phi_2 = (1.0 + xi) / 2.0;
        MatrixMN::<_, U1, U2>::new(phi_1, phi_2)
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, _xi: &Point1<T>) -> MatrixMN<T, U1, U2> {
        MatrixMN::<_, U1, U2>::new(-0.5, 0.5)
    }
}

impl_reference_finite_element_for_fixed!(Segment2d2Element<T>);

impl<T> FiniteElement<T> for Segment2d2Element<T>
where
    T: RealField,
{
    type GeometryDim = U2;

    #[allow(non_snake_case)]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn reference_jacobian(&self, _xi: &Point1<T>) -> Vector2<T> {
        let a = &self.segment.from().coords;
        let b = &self.segment.to().coords;
        (b - a) / 2.0
    }

    #[allow(non_snake_case)]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn map_reference_coords(&self, xi: &Point1<T>) -> Point2<T> {
        let a = &self.segment.from().coords;
        let b = &self.segment.to().coords;
        let phi = self.evaluate_basis(xi);
        Point::from(a * phi[0] + b * phi[1])
    }

    fn diameter(&self) -> T {
        self.segment.length()
    }
}

impl<T> SurfaceFiniteElement<T> for Segment2d2Element<T>
where
    T: RealField,
{
    fn normal(&self, _xi: &Point1<T>) -> Vector2<T> {
        self.segment.normal_dir().normalize()
    }
}

impl<T> ElementConnectivity<T> for Segment2d2Connectivity
where
    T: RealField,
{
    type Element = Segment2d2Element<T>;
    type ReferenceDim = U1;
    type GeometryDim = U2;

    fn element(&self, vertices: &[Point2<T>]) -> Option<Self::Element> {
        let a = vertices[self.0[0]].clone();
        let b = vertices[self.0[1]].clone();
        let segment = LineSegment2d::new(a, b);
        Some(Segment2d2Element::from(segment))
    }
}

impl<T> ElementConnectivity<T> for Tet4Connectivity
where
    T: RealField,
{
    type Element = Tet4Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[Point<T, Self::GeometryDim>]) -> Option<Self::Element> {
        Some(Tet4Element {
            vertices: [
                vertices.get(self.0[0])?.clone(),
                vertices.get(self.0[1])?.clone(),
                vertices.get(self.0[2])?.clone(),
                vertices.get(self.0[3])?.clone(),
            ],
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Tet4Element<T>
where
    T: Scalar,
{
    vertices: [Point3<T>; 4],
}

impl<T> Tet4Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 4]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point3<T>; 4] {
        &self.vertices
    }
}

impl<T> Tet4Element<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self {
            vertices: [
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, -1.0, -1.0),
                Point3::new(-1.0, 1.0, -1.0),
                Point3::new(-1.0, -1.0, 1.0),
            ],
        }
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
impl<T> FixedNodesReferenceFiniteElement<T> for Tet4Element<T>
where
    T: RealField,
{
    type ReferenceDim = U3;
    type NodalDim = U4;

    #[rustfmt::skip]
    fn evaluate_basis(&self, xi: &Point3<T>) -> Matrix1x4<T> {
        Matrix1x4::from_row_slice(&[
            -0.5 * xi.x - 0.5 * xi.y - 0.5 * xi.z - 0.5,
            0.5 * xi.x + 0.5,
            0.5 * xi.y + 0.5,
            0.5 * xi.z + 0.5
        ])
    }

    #[rustfmt::skip]
    fn gradients(&self, _reference_coords: &Point3<T>) -> Matrix3x4<T> {
        Matrix3x4::from_columns(&[
            Vector3::new(-0.5, -0.5, -0.5),
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.0, 0.0, 0.5)
        ])
    }
}

impl_reference_finite_element_for_fixed!(Tet4Element<T>);

impl<T> FiniteElement<T> for Tet4Element<T>
where
    T: RealField,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        // TODO: Could store this matrix directly in the element, in order
        // to avoid repeated computation
        let X = Matrix3x4::from_fn(|i, j| self.vertices[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        // TODO: Store this X matrix directly in Self...?
        let X = Matrix3x4::from_fn(|i, j| self.vertices[j][i]);
        let N = self.evaluate_basis(xi);
        Point::from(&X * &N.transpose())
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.vertices
            .iter()
            .tuple_combinations()
            .map(|(x, y)| distance(x, y))
            .fold(T::zero(), |a, b| a.max(b.clone()))
    }
}

impl<T> ElementConnectivity<T> for Hex8Connectivity
where
    T: RealField,
{
    type Element = Hex8Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[Point<T, Self::GeometryDim>]) -> Option<Self::Element> {
        Some(Hex8Element::from_vertices([
            vertices.get(self.0[0])?.clone(),
            vertices.get(self.0[1])?.clone(),
            vertices.get(self.0[2])?.clone(),
            vertices.get(self.0[3])?.clone(),
            vertices.get(self.0[4])?.clone(),
            vertices.get(self.0[5])?.clone(),
            vertices.get(self.0[6])?.clone(),
            vertices.get(self.0[7])?.clone(),
        ]))
    }
}

/// Linear basis function on the interval [-1, 1].
///
///`alpha == -1` denotes the basis function associated with the node at `x == -1`,
/// and `alpha == 1` for `x == 1`.
#[replace_float_literals(T::from_f64(literal).unwrap())]
#[inline(always)]
fn phi_linear_1d<T>(alpha: T, xi: T) -> T
where
    T: RealField,
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
    T: RealField,
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
    T: RealField,
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
    T: RealField,
{
    // The compiler should hopefully be able to use constant propagation to
    // precompute all expressions involving constants and alpha
    let alpha2 = alpha * alpha;
    2.0 * (3.0 / 2.0 * alpha2 - 1.0) * xi + 0.5 * alpha
}

impl<T> FixedNodesReferenceFiniteElement<T> for Hex8Element<T>
where
    T: RealField,
{
    type ReferenceDim = U3;
    type NodalDim = U8;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point3<T>) -> MatrixMN<T, U1, U8> {
        // We define the shape functions as N_{alpha, beta, gamma} evaluated at xi such that
        //  N_{alpha, beta, gamma}([alpha, beta, gamma]) = 1,
        let phi_1d = phi_linear_1d;
        let phi = |alpha, beta, gamma, xi: &Point3<T>|
            phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]);
        MatrixMN::<_, U1, U8>::from_row_slice(&[
            phi(-1.0, -1.0, -1.0, xi),
            phi( 1.0, -1.0, -1.0, xi),
            phi( 1.0,  1.0, -1.0, xi),
            phi(-1.0,  1.0, -1.0, xi),
            phi(-1.0, -1.0,  1.0, xi),
            phi( 1.0, -1.0,  1.0, xi),
            phi( 1.0,  1.0,  1.0, xi),
            phi(-1.0,  1.0,  1.0, xi),
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point3<T>) -> MatrixMN<T, U3, U8> {
        let phi_1d = phi_linear_1d;
        let grad_1d = phi_linear_1d_grad;
        let phi_grad = |alpha, beta, gamma, xi: &Point3<T>|
            Vector3::new(
                grad_1d(alpha) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * grad_1d(beta) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * grad_1d(gamma)
            );

        MatrixMN::from_columns(&[
            phi_grad(-1.0, -1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, -1.0, xi),
            phi_grad(-1.0,  1.0, -1.0, xi),
            phi_grad(-1.0, -1.0,  1.0, xi),
            phi_grad( 1.0, -1.0,  1.0, xi),
            phi_grad( 1.0,  1.0,  1.0, xi),
            phi_grad(-1.0,  1.0,  1.0, xi),
        ])
    }
}

impl_reference_finite_element_for_fixed!(Hex8Element<T>);

impl<T> FiniteElement<T> for Hex8Element<T>
where
    T: RealField,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        // TODO: Store this X matrix directly in Self...?
        let X = MatrixMN::<_, U3, U8>::from_fn(|i, j| self.vertices[j][i]);
        let N = self.evaluate_basis(xi);
        Point::from(&X * &N.transpose())
    }

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        // TODO: Could store this matrix directly in the element, in order
        // to avoid repeated computation
        let X = MatrixMN::<_, U3, U8>::from_fn(|i, j| self.vertices[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.vertices
            .iter()
            .tuple_combinations()
            .map(|(x, y)| distance(x, y))
            .fold(T::zero(), |a, b| a.max(b.clone()))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Hex8Element<T: Scalar> {
    vertices: [Point3<T>; 8],
}

impl<T> Hex8Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 8]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point3<T>; 8] {
        &self.vertices
    }
}

impl<T> Hex8Element<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
        ])
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Hex27Element<T: Scalar> {
    // Store a hex8 element for trilinear transformations from reference element
    hex8: Hex8Element<T>,
    vertices: [Point3<T>; 27],
}

impl<T: Scalar + Copy> Hex27Element<T> {
    pub fn from_vertices(vertices: [Point3<T>; 27]) -> Self {
        Self {
            hex8: Hex8Element::from_vertices(vertices[0..8].try_into().unwrap()),
            vertices,
        }
    }

    pub fn vertices(&self) -> &[Point3<T>] {
        &self.vertices
    }
}

impl<T: RealField> Hex27Element<T> {
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
            // Edge nodes
            Point3::new(0.0, -1.0, -1.0),
            Point3::new(-1.0, 0.0, -1.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, -1.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, -1.0, 1.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            // Face nodes
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            // Center node
            Point3::new(0.0, 0.0, 0.0),
        ])
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Hex27Element<T>
where
    T: RealField,
{
    type ReferenceDim = U3;
    type NodalDim = U27;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point3<T>) -> MatrixMN<T, U1, U27> {
        // We define the shape functions as N_{alpha, beta, gamma} evaluated at xi such that
        //  N_{alpha, beta, gamma}([alpha, beta, gamma]) = 1,
        let phi_1d = phi_quadratic_1d;
        let phi = |alpha, beta, gamma, xi: &Point3<T>|
            phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]);
        MatrixMN::<_, U1, U27>::from_row_slice(&[
            // Vertex nodes
            phi(-1.0, -1.0, -1.0, xi),
            phi( 1.0, -1.0, -1.0, xi),
            phi( 1.0,  1.0, -1.0, xi),
            phi(-1.0,  1.0, -1.0, xi),
            phi(-1.0, -1.0,  1.0, xi),
            phi( 1.0, -1.0,  1.0, xi),
            phi( 1.0,  1.0,  1.0, xi),
            phi(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi(0.0, -1.0, -1.0, xi),
            phi(-1.0, 0.0, -1.0, xi),
            phi(-1.0, -1.0, 0.0, xi),
            phi(1.0, 0.0, -1.0, xi),
            phi(1.0, -1.0, 0.0, xi),
            phi(0.0, 1.0, -1.0, xi),
            phi(1.0, 1.0, 0.0, xi),
            phi(-1.0, 1.0, 0.0, xi),
            phi(0.0, -1.0, 1.0, xi),
            phi(-1.0, 0.0, 1.0, xi),
            phi(1.0, 0.0, 1.0, xi),
            phi(0.0, 1.0, 1.0, xi),

            // Face nodes
            phi(0.0, 0.0, -1.0, xi),
            phi(0.0, -1.0, 0.0, xi),
            phi(-1.0, 0.0, 0.0, xi),
            phi(1.0, 0.0, 0.0, xi),
            phi(0.0, 1.0, 0.0, xi),
            phi(0.0, 0.0, 1.0, xi),

            // Center node
            phi(0.0, 0.0, 0.0, xi)
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point3<T>) -> MatrixMN<T, U3, U27> {
        let phi_1d = phi_quadratic_1d;
        let grad_1d = phi_quadratic_1d_grad;
        let phi_grad = |alpha, beta, gamma, xi: &Point3<T>|
            Vector3::new(
                grad_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * grad_1d(beta, xi[1]) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * grad_1d(gamma, xi[2])
            );

        MatrixMN::from_columns(&[
            // Vertex nodes
            phi_grad(-1.0, -1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, -1.0, xi),
            phi_grad(-1.0,  1.0, -1.0, xi),
            phi_grad(-1.0, -1.0,  1.0, xi),
            phi_grad( 1.0, -1.0,  1.0, xi),
            phi_grad( 1.0,  1.0,  1.0, xi),
            phi_grad(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi_grad(0.0, -1.0, -1.0, xi),
            phi_grad(-1.0, 0.0, -1.0, xi),
            phi_grad(-1.0, -1.0, 0.0, xi),
            phi_grad(1.0, 0.0, -1.0, xi),
            phi_grad(1.0, -1.0, 0.0, xi),
            phi_grad(0.0, 1.0, -1.0, xi),
            phi_grad(1.0, 1.0, 0.0, xi),
            phi_grad(-1.0, 1.0, 0.0, xi),
            phi_grad(0.0, -1.0, 1.0, xi),
            phi_grad(-1.0, 0.0, 1.0, xi),
            phi_grad(1.0, 0.0, 1.0, xi),
            phi_grad(0.0, 1.0, 1.0, xi),

            // Face nodes
            phi_grad(0.0, 0.0, -1.0, xi),
            phi_grad(0.0, -1.0, 0.0, xi),
            phi_grad(-1.0, 0.0, 0.0, xi),
            phi_grad(1.0, 0.0, 0.0, xi),
            phi_grad(0.0, 1.0, 0.0, xi),
            phi_grad(0.0, 0.0, 1.0, xi),

            // Center node
            phi_grad(0.0, 0.0, 0.0, xi)
        ])
    }
}

impl_reference_finite_element_for_fixed!(Hex27Element<T>);

impl<T> FiniteElement<T> for Hex27Element<T>
where
    T: RealField,
{
    type GeometryDim = U3;

    fn reference_jacobian(&self, reference_coords: &Point3<T>) -> Matrix3<T> {
        self.hex8.reference_jacobian(reference_coords)
    }

    fn map_reference_coords(&self, reference_coords: &Point<T, Self::ReferenceDim>) -> Point3<T> {
        self.hex8.map_reference_coords(reference_coords)
    }

    fn diameter(&self) -> T {
        self.hex8.diameter()
    }
}

impl<T> ElementConnectivity<T> for Hex27Connectivity
where
    T: RealField,
{
    type Element = Hex27Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, global_vertices: &[Point3<T>]) -> Option<Self::Element> {
        let mut hex_vertices = [Point::origin(); 27];

        for (local_idx, global_idx) in self.0.iter().enumerate() {
            hex_vertices[local_idx] = global_vertices.get(*global_idx)?.clone();
        }

        Some(Hex27Element::from_vertices(hex_vertices))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Hex20Element<T: Scalar> {
    // Store a hex8 element for trilinear transformations from reference element
    hex8: Hex8Element<T>,
    vertices: [Point3<T>; 20],
}

impl<T: Scalar + Copy> Hex20Element<T> {
    pub fn from_vertices(vertices: [Point3<T>; 20]) -> Self {
        Self {
            hex8: Hex8Element::from_vertices(vertices[0..8].try_into().unwrap()),
            vertices,
        }
    }

    pub fn vertices(&self) -> &[Point3<T>] {
        &self.vertices
    }
}

impl<T: RealField> Hex20Element<T> {
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
            // Edge nodes
            Point3::new(0.0, -1.0, -1.0),
            Point3::new(-1.0, 0.0, -1.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, -1.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, -1.0, 1.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ])
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Hex20Element<T>
where
    T: RealField,
{
    type ReferenceDim = U3;
    type NodalDim = U20;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point3<T>) -> MatrixMN<T, U1, U20> {
        // We define the shape functions as N_{alpha, beta, gamma} evaluated at xi such that
        //  N_{alpha, beta, gamma}([alpha, beta, gamma]) = 1,
        // but we define corner and edge nodes separately.

        // Formulas are adapted from the following website:
        // http://www.softeng.rl.ac.uk/st/projects/felib4/Docs/html/Level-0/brk20/brk20.html

        let phi_corner = |alpha, beta, gamma, xi: &Point3<T>|
            (1.0 / 8.0) * (1.0 + alpha * xi[0])
                * (1.0 + beta * xi[1])
                * (1.0 + gamma * xi[2])
                * (alpha * xi[0] + beta * xi[1] + gamma * xi[2] - 2.0);

        let phi_edge = |alpha, beta, gamma, xi: &Point3<T>| {
            let alpha2 = alpha * alpha;
            let beta2 = beta * beta;
            let gamma2 = gamma * gamma;
            (1.0 / 4.0) * (1.0 - (1.0 - alpha2) * xi[0]*xi[0])
                * (1.0 - (1.0 - beta2) * xi[1]*xi[1])
                * (1.0 - (1.0 - gamma2) * xi[2]*xi[2])
                * (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2])
        };

        MatrixMN::<_, U1, U20>::from_row_slice(&[
            // Corner nodes
            phi_corner(-1.0, -1.0, -1.0, xi),
            phi_corner( 1.0, -1.0, -1.0, xi),
            phi_corner( 1.0,  1.0, -1.0, xi),
            phi_corner(-1.0,  1.0, -1.0, xi),
            phi_corner(-1.0, -1.0,  1.0, xi),
            phi_corner( 1.0, -1.0,  1.0, xi),
            phi_corner( 1.0,  1.0,  1.0, xi),
            phi_corner(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi_edge(0.0, -1.0, -1.0, xi),
            phi_edge(-1.0, 0.0, -1.0, xi),
            phi_edge(-1.0, -1.0, 0.0, xi),
            phi_edge(1.0, 0.0, -1.0, xi),
            phi_edge(1.0, -1.0, 0.0, xi),
            phi_edge(0.0, 1.0, -1.0, xi),
            phi_edge(1.0, 1.0, 0.0, xi),
            phi_edge(-1.0, 1.0, 0.0, xi),
            phi_edge(0.0, -1.0, 1.0, xi),
            phi_edge(-1.0, 0.0, 1.0, xi),
            phi_edge(1.0, 0.0, 1.0, xi),
            phi_edge(0.0, 1.0, 1.0, xi),
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point3<T>) -> MatrixMN<T, U3, U20> {
        let phi_grad_corner = |alpha, beta, gamma, xi: &Point3<T>| {
            // Decompose shape function as phi(xi) = (1/8) * f(xi) * g(xi),
            // with
            //  f(xi) = sum_i (alpha_i xi_i) - 2
            //  g(xi) = product_i (1 + alpha_i xi_i)
            // and use product rule to arrive at the below expression
            let f = alpha * xi[0] + beta * xi[1] + gamma * xi[2] - 2.0;
            let g = (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2]);
            let s = 1.0 / 8.0;
            Vector3::new(
                s * (alpha * g + f * alpha * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2])),
                s * (beta * g + f * beta * (1.0 + alpha * xi[0]) * (1.0 + gamma * xi[2])),
                s * (gamma * g + f * gamma * (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]))
            )
        };

        let phi_grad_edge = |alpha, beta, gamma, xi: &Point3<T>| {
            // Decompose shape function as phi(xi) = (1/8) * h(xi) * g(xi),
            // with
            //  h(xi) = product_i (1.0 - (1.0 - alpha_i^2) xi_i^2)
            //  g(xi) = product_i (1 + alpha_i xi_i)
            // and use product rule to arrive at the below expression
            let alpha2 = alpha * alpha;
            let beta2 = beta * beta;
            let gamma2 = gamma * gamma;
            let h = (1.0 - (1.0 - alpha2) * xi[0]*xi[0])
                * (1.0 - (1.0 - beta2) * xi[1]*xi[1])
                * (1.0 - (1.0 - gamma2) * xi[2]*xi[2]);
            let g = (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2]);
            let s = 1.0 / 4.0;

            // Note: we hope that the optimizer is able to optimize away most of these operations,
            // since alpha2, beta2, gamma2 should be known at compile-time, which
            // makes many of the terms here zero.
            let dh_xi0 = -2.0 * (1.0 - alpha2) * xi[0]
                * (1.0 - (1.0 - beta2) * xi[1]*xi[1])
                * (1.0 - (1.0 - gamma2) * xi[2]*xi[2]);
            let dh_xi1 = -2.0 * (1.0 - beta2) * xi[1]
                * (1.0 - (1.0 - alpha2) * xi[0] * xi[0])
                * (1.0 - (1.0 - gamma2) * xi[2] * xi[2]);
            let dh_xi2 = -2.0 * (1.0 - gamma2) * xi[2]
                * (1.0 - (1.0 - alpha2) * xi[0] * xi[0])
                * (1.0 - (1.0 - beta2) * xi[1] * xi[1]);
            Vector3::new(
                s * (dh_xi0 * g + h * alpha * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2])),
                s * (dh_xi1 * g + h * beta * (1.0 + alpha * xi[0]) * (1.0 + gamma * xi[2])),
                s * (dh_xi2 * g + h * gamma * (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]))
            )
        };

        MatrixMN::from_columns(&[
            // Corner nodes
            phi_grad_corner(-1.0, -1.0, -1.0, xi),
            phi_grad_corner( 1.0, -1.0, -1.0, xi),
            phi_grad_corner( 1.0,  1.0, -1.0, xi),
            phi_grad_corner(-1.0,  1.0, -1.0, xi),
            phi_grad_corner(-1.0, -1.0,  1.0, xi),
            phi_grad_corner( 1.0, -1.0,  1.0, xi),
            phi_grad_corner( 1.0,  1.0,  1.0, xi),
            phi_grad_corner(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi_grad_edge(0.0, -1.0, -1.0, xi),
            phi_grad_edge(-1.0, 0.0, -1.0, xi),
            phi_grad_edge(-1.0, -1.0, 0.0, xi),
            phi_grad_edge(1.0, 0.0, -1.0, xi),
            phi_grad_edge(1.0, -1.0, 0.0, xi),
            phi_grad_edge(0.0, 1.0, -1.0, xi),
            phi_grad_edge(1.0, 1.0, 0.0, xi),
            phi_grad_edge(-1.0, 1.0, 0.0, xi),
            phi_grad_edge(0.0, -1.0, 1.0, xi),
            phi_grad_edge(-1.0, 0.0, 1.0, xi),
            phi_grad_edge(1.0, 0.0, 1.0, xi),
            phi_grad_edge(0.0, 1.0, 1.0, xi),
        ])
    }
}

impl_reference_finite_element_for_fixed!(Hex20Element<T>);

impl<T> FiniteElement<T> for Hex20Element<T>
where
    T: RealField,
{
    type GeometryDim = U3;

    fn reference_jacobian(&self, reference_coords: &Point3<T>) -> Matrix3<T> {
        self.hex8.reference_jacobian(reference_coords)
    }

    fn map_reference_coords(&self, reference_coords: &Point<T, Self::ReferenceDim>) -> Point3<T> {
        self.hex8.map_reference_coords(reference_coords)
    }

    fn diameter(&self) -> T {
        self.hex8.diameter()
    }
}

impl<T> ElementConnectivity<T> for Hex20Connectivity
where
    T: RealField,
{
    type Element = Hex20Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, global_vertices: &[Point3<T>]) -> Option<Self::Element> {
        let mut hex_vertices = [Point::origin(); 20];

        for (local_idx, global_idx) in self.0.iter().enumerate() {
            hex_vertices[local_idx] = global_vertices.get(*global_idx)?.clone();
        }

        Some(Hex20Element::from_vertices(hex_vertices))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// A (surface) finite element representing linear basis functions on a triangle,
/// in three dimensions.
///
/// The reference element is chosen to be the triangle defined by the corners
/// (-1, -1), (1, -1), (-1, 1). This perhaps unorthodox choice is due to the quadrature rules
/// we employ.
pub struct Tri3d3Element<T>
where
    T: Scalar,
{
    triangle: Triangle3d<T>,
}

impl<T> From<Triangle3d<T>> for Tri3d3Element<T>
where
    T: Scalar,
{
    fn from(triangle: Triangle3d<T>) -> Self {
        Self { triangle }
    }
}

impl<T> ElementConnectivity<T> for Tri3d3Connectivity
where
    T: RealField,
{
    type Element = Tri3d3Element<T>;
    type ReferenceDim = U2;
    type GeometryDim = U3;

    fn element(&self, vertices: &[Point3<T>]) -> Option<Self::Element> {
        let Self(indices) = self;
        let lookup_vertex = |local_index| vertices.get(indices[local_index]).cloned();

        Some(Tri3d3Element::from(Triangle([
            lookup_vertex(0)?,
            lookup_vertex(1)?,
            lookup_vertex(2)?,
        ])))
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Tri3d3Element<T>
where
    T: RealField,
{
    type NodalDim = U3;
    type ReferenceDim = U2;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point2<T>) -> Matrix1x3<T> {
        // TODO: Reuse implementation from Trid2Element instead
        Matrix1x3::from_row_slice(&[
            -0.5 * xi[0] - 0.5 * xi[1],
            0.5 * xi[0] + 0.5,
            0.5 * xi[1] + 0.5
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, _: &Point2<T>) -> Matrix2x3<T> {
        // TODO: Reuse implementation from Trid2Element instead
        // TODO: Precompute gradients
        Matrix2x3::from_columns(&[
            Vector2::new(-0.5, -0.5),
            Vector2::new(0.5, 0.0),
            Vector2::new(0.0, 0.5)
        ])
    }
}

impl_reference_finite_element_for_fixed!(Tri3d3Element<T>);

impl<T> FiniteElement<T> for Tri3d3Element<T>
where
    T: RealField,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix3x2<T> {
        let X: Matrix3<T> = Matrix3::from_fn(|i, j| self.triangle.0[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point2<T>) -> Point3<T> {
        // TODO: Store this X matrix directly in Self...?
        let X: Matrix3<T> = Matrix3::from_fn(|i, j| self.triangle.0[j][i]);
        let N = self.evaluate_basis(xi);
        Point::from(&X * &N.transpose())
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.triangle
            .0
            .iter()
            .tuple_combinations()
            .map(|(x, y)| distance(x, y))
            .fold(T::zero(), |a, b| a.max(b.clone()))
    }
}

impl<T> SurfaceFiniteElement<T> for Tri3d3Element<T>
where
    T: RealField,
{
    fn normal(&self, _xi: &Point2<T>) -> Vector3<T> {
        self.triangle.normal()
    }
}

impl<T> ElementConnectivity<T> for Tet10Connectivity
where
    T: RealField,
{
    type Element = Tet10Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[Point<T, Self::GeometryDim>]) -> Option<Self::Element> {
        let mut tet10_vertices = [Point3::origin(); 10];
        for (i, v) in tet10_vertices.iter_mut().enumerate() {
            *v = vertices.get(self.0[i])?.clone();
        }

        let mut tet4_vertices = [Point3::origin(); 4];
        tet4_vertices.copy_from_slice(&tet10_vertices[0..4]);

        Some(Tet10Element {
            tet4: Tet4Element::from_vertices(tet4_vertices),
            vertices: tet10_vertices,
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Tet10Element<T>
where
    T: Scalar,
{
    tet4: Tet4Element<T>,
    vertices: [Point3<T>; 10],
}

impl<T> Tet10Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 10]) -> Self {
        let tet4_v = [
            vertices[0].clone(),
            vertices[1].clone(),
            vertices[2].clone(),
            vertices[3].clone(),
        ];
        Self {
            tet4: Tet4Element::from_vertices(tet4_v),
            vertices,
        }
    }

    pub fn vertices(&self) -> &[Point3<T>; 10] {
        &self.vertices
    }
}

impl<T> Tet10Element<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self {
            tet4: Tet4Element::reference(),
            vertices: [
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, -1.0, -1.0),
                Point3::new(-1.0, 1.0, -1.0),
                Point3::new(-1.0, -1.0, 1.0),
                Point3::new(0.0, -1.0, -1.0),
                Point3::new(0.0, 0.0, -1.0),
                Point3::new(-1.0, 0.0, -1.0),
                Point3::new(-1.0, -1.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(0.0, -1.0, 0.0),
            ],
        }
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
impl<T> FixedNodesReferenceFiniteElement<T> for Tet10Element<T>
where
    T: RealField,
{
    type ReferenceDim = U3;
    type NodalDim = U10;

    #[rustfmt::skip]
    fn evaluate_basis(&self, xi: &Point3<T>) -> MatrixMN<T, U1, U10> {
        // We express the basis functions of Tet10 as products of
        // the Tet4 basis functions.
        let psi = self.tet4.evaluate_basis(xi);
        MatrixMN::from([
            psi[0] * (2.0 * psi[0] - 1.0),
            psi[1] * (2.0 * psi[1] - 1.0),
            psi[2] * (2.0 * psi[2] - 1.0),
            psi[3] * (2.0 * psi[3] - 1.0),
            4.0 * psi[0] * psi[1],
            4.0 * psi[1] * psi[2],
            4.0 * psi[0] * psi[2],
            4.0 * psi[0] * psi[3],
            4.0 * psi[2] * psi[3],
            4.0 * psi[1] * psi[3]
        ])
    }

    #[rustfmt::skip]
    fn gradients(&self, xi: &Point3<T>) -> MatrixMN<T, U3, U10> {
        // Similarly to `evaluate_basis`, we may implement the gradients of
        // Tet10 with the help of the function values and gradients of Tet4
        let psi = self.tet4.evaluate_basis(xi);
        let g = self.tet4.gradients(xi);

        // Gradient of vertex node i
        let vertex_gradient = |i| g.index((.., i)) * (4.0 * psi[i] - 1.0);

        // Gradient of edge node on the edge between vertex i and j
        let edge_gradient = |i, j|
            g.index((.., i)) * (4.0 * psi[j]) + g.index((.., j)) * (4.0 * psi[i]);

        MatrixMN::from_columns(&[
            vertex_gradient(0),
            vertex_gradient(1),
            vertex_gradient(2),
            vertex_gradient(3),
            edge_gradient(0, 1),
            edge_gradient(1, 2),
            edge_gradient(0, 2),
            edge_gradient(0, 3),
            edge_gradient(2, 3),
            edge_gradient(1, 3)
        ])
    }
}

impl_reference_finite_element_for_fixed!(Tet10Element<T>);

impl<T> FiniteElement<T> for Tet10Element<T>
where
    T: RealField,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        self.tet4.reference_jacobian(xi)
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        self.tet4.map_reference_coords(xi)
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.tet4.diameter()
    }
}

/// Maps physical coordinates `x` to reference coordinates `xi` by solving the equation
///  x - T(xi) = 0 using Newton's method.
///
pub fn map_physical_coordinates<T, Element, GeometryDim>(
    element: &Element,
    x: &Point<T, GeometryDim>,
) -> Result<Point<T, GeometryDim>, Box<dyn Error>>
where
    T: RealField,
    Element: FiniteElement<T, GeometryDim = GeometryDim, ReferenceDim = GeometryDim>,
    GeometryDim: DimName + DimMin<GeometryDim, Output = GeometryDim>,
    DefaultAllocator: VolumeFiniteElementAllocator<T, GeometryDim>,
{
    use crate::optimize::calculus::VectorFunctionBuilder;
    use crate::optimize::newton::newton;

    let f = VectorFunctionBuilder::with_dimension(GeometryDim::dim())
        .with_function(|f, xi| {
            // Need to create stack-allocated xi
            let xi = Point::from(xi.fixed_slice::<GeometryDim, U1>(0, 0).clone_owned());
            f.copy_from(&(element.map_reference_coords(&xi).coords - &x.coords));
        })
        .with_jacobian_solver(
            |sol: &mut DVectorSliceMut<T>, xi: &DVectorSlice<T>, rhs: &DVectorSlice<T>| {
                let xi = Point::from(xi.fixed_slice::<GeometryDim, U1>(0, 0).clone_owned());
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

    let mut xi = VectorN::<T, GeometryDim>::zeros();
    let mut f_val = VectorN::<T, GeometryDim>::zeros();
    let mut dx = VectorN::<T, GeometryDim>::zeros();

    // Because we cannot prove to the compiler that the strides of `VectorN<T, GeometryDim>`
    // are compatible (in a `DimEq` sense) without nasty additional trait bounds,
    // we first take slices of the vectors so that the stride is dynamic. At this point,
    // it is known that `DimEq<Dynamic, U1>` works, so we can use it with `newton`,
    // `which expects `Into<DMatrixSliceMut<T>>`.
    macro_rules! slice {
        ($e:expr) => {
            $e.fixed_slice_with_steps_mut::<GeometryDim, U1>((0, 0), (0, 0))
        };
    }

    newton(
        f,
        &mut slice!(xi),
        &mut slice!(f_val),
        &mut slice!(dx),
        settings,
    )?;

    Ok(Point::from(xi))
}

/// Projects physical coordinates `x` to reference coordinates `xi` by solving the equation
///  x - T(xi) = 0 using a generalized form of Newton's method.
///
/// Unlike `map_physical_coordinates`, this method is also applicable to e.g. surface finite
/// elements, in which the reference dimension and geometry dimension differ.
///
/// The method panics if `ReferenceDim` is greater than `GeometryDim`.
///
#[allow(non_snake_case)]
pub fn project_physical_coordinates<T, Element>(
    element: &Element,
    x: &Point<T, Element::GeometryDim>,
) -> Result<Point<T, Element::ReferenceDim>, Box<dyn Error>>
where
    T: RealField,
    Element: FiniteElement<T>,
    Element::ReferenceDim: DimName + DimMin<Element::ReferenceDim, Output = Element::ReferenceDim>,
    DefaultAllocator: FiniteElementAllocator<T, Element::GeometryDim, Element::ReferenceDim>,
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
    let mut xi = Point::<T, Element::ReferenceDim>::origin();
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

    Ok(Point::from(xi))
}

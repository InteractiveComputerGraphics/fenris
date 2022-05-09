use std::convert::TryFrom;

use itertools::Itertools;
use numeric_literals::replace_float_literals;

use crate::connectivity::{Quad4d2Connectivity, Quad9d2Connectivity};
use crate::element::{ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement};
use crate::geometry::{ConcavePolygonError, ConvexPolygon, LineSegment2d, Quad2d};
use crate::nalgebra::{
    distance, Matrix1x4, Matrix2, Matrix2x4, OMatrix, OPoint, Point2, RealField, Scalar, Vector2, U1, U2, U4, U9,
};

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
        OPoint::from(&X * &N.transpose())
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
    pub fn from_vertices(vertices: [Point2<T>; 9]) -> Self {
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
        let midpoint = |a: &Point2<_>, b: &Point2<_>| LineSegment2d::from_end_points(a.clone(), b.clone()).midpoint();

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
    fn evaluate_basis(&self, xi: &Point2<T>) -> OMatrix<T, U1, U9> {
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

        OMatrix::<T, U1, U9>::from_row_slice(&[
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
    fn gradients(&self, xi: &Point2<T>) -> OMatrix<T, U2, U9> {
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

        OMatrix::<T, U2, U9>::from_columns(&[
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

impl<T> TryFrom<Quad9d2Element<T>> for ConvexPolygon<T>
where
    T: RealField,
{
    type Error = ConcavePolygonError;

    fn try_from(value: Quad9d2Element<T>) -> Result<Self, Self::Error> {
        ConvexPolygon::try_from(value.quad)
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

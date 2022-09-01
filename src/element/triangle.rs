use itertools::Itertools;
use numeric_literals::replace_float_literals;

use crate::connectivity::{Tri3d2Connectivity, Tri3d3Connectivity, Tri6d2Connectivity};
use crate::element::{ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement, SurfaceFiniteElement};
use crate::geometry::{LineSegment2d, Triangle, Triangle2d, Triangle3d};
use crate::nalgebra::{
    distance, Matrix1x3, Matrix1x6, Matrix2, Matrix2x3, Matrix2x6, Matrix3, Matrix3x2, OPoint, Point2, Point3, Scalar,
    Vector2, Vector3, U2, U3, U6,
};
use crate::Real;

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
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self::from_vertices([Point2::new(-1.0, -1.0), Point2::new(1.0, -1.0), Point2::new(-1.0, 1.0)])
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Tri3d2Element<T>
where
    T: Real,
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

impl<T> FiniteElement<T> for Tri3d2Element<T>
where
    T: Real,
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
        OPoint::from(&X * &N.transpose())
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
    T: Real,
{
    // TODO: Test this
    fn from(tri3: &'a Tri3d2Element<T>) -> Self {
        let midpoint = |a: &Point2<_>, b: &Point2<_>| LineSegment2d::from_end_points(a.clone(), b.clone()).midpoint();

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
    T: Real,
{
    fn from(tri3: Tri3d2Element<T>) -> Self {
        Self::from(&tri3)
    }
}

impl<T> Tri6d2Element<T>
where
    T: Real,
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

impl<T> FixedNodesReferenceFiniteElement<T> for Tri6d2Element<T>
where
    T: Real,
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

impl<T> FiniteElement<T> for Tri6d2Element<T>
where
    T: Real,
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

impl<T> ElementConnectivity<T> for Tri3d2Connectivity
where
    T: Real,
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

impl<T> ElementConnectivity<T> for Tri6d2Connectivity
where
    T: Real,
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
    vertices: [Point3<T>; 3],
}

impl<T: Scalar> Tri3d3Element<T> {
    pub fn from_vertices(vertices: [Point3<T>; 3]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point3<T>; 3] {
        &self.vertices
    }
}

impl<'a, T: Scalar> From<&'a Tri3d3Element<T>> for Triangle3d<T> {
    fn from(element: &'a Tri3d3Element<T>) -> Self {
        Triangle(element.vertices.clone())
    }
}

impl<T> From<Triangle3d<T>> for Tri3d3Element<T>
where
    T: Scalar,
{
    fn from(triangle: Triangle3d<T>) -> Self {
        Self::from_vertices(triangle.0)
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Tri3d3Element<T>
where
    T: Real,
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

impl<T> FiniteElement<T> for Tri3d3Element<T>
where
    T: Real,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point2<T>) -> Matrix3x2<T> {
        let X: Matrix3<T> = Matrix3::from_fn(|i, j| self.vertices[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point2<T>) -> Point3<T> {
        // TODO: Store this X matrix directly in Self...?
        let X: Matrix3<T> = Matrix3::from_fn(|i, j| self.vertices[j][i]);
        let N = self.evaluate_basis(xi);
        OPoint::from(&X * &N.transpose())
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

impl<T> SurfaceFiniteElement<T> for Tri3d3Element<T>
where
    T: Real,
{
    fn normal(&self, _xi: &Point2<T>) -> Vector3<T> {
        Triangle3d::from(self).normal()
    }
}

impl<T> ElementConnectivity<T> for Tri3d3Connectivity
where
    T: Real,
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

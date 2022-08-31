use crate::connectivity::{Segment2d1Connectivity, Segment2d2Connectivity};
use crate::element::{ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement, SurfaceFiniteElement};
use crate::geometry::LineSegment2d;
use crate::nalgebra::{OMatrix, OPoint, Point1, Point2, Scalar, Vector2, U1, U2};
use crate::Real;
use nalgebra::{point, Vector1};
use numeric_literals::replace_float_literals;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// A segment in one dimension.
pub struct Segment2d1Element<T>
where
    T: Scalar,
{
    vertices: [Point1<T>; 2],
}

impl<T: Scalar> Segment2d1Element<T> {
    pub fn from_vertices(vertices: [Point1<T>; 2]) -> Self {
        Self { vertices }
    }

    pub fn from_interval(interval: [T; 2]) -> Self {
        Self::from_vertices([point!(interval[0].clone()), point!(interval[1].clone())])
    }

    pub fn vertices(&self) -> &[Point1<T>; 2] {
        &self.vertices
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// A surface element embedded in two dimensions.
pub struct Segment2d2Element<T>
where
    T: Scalar,
{
    vertices: [Point2<T>; 2],
}

impl<T: Scalar> Segment2d2Element<T> {
    pub fn from_vertices(vertices: [Point2<T>; 2]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point2<T>; 2] {
        &self.vertices
    }

    pub fn to_line_segment(&self) -> LineSegment2d<T> {
        self.into()
    }
}

impl<T> From<LineSegment2d<T>> for Segment2d2Element<T>
where
    T: Scalar,
{
    fn from(segment: LineSegment2d<T>) -> Self {
        Self::from(&segment)
    }
}

impl<'a, T> From<&'a LineSegment2d<T>> for Segment2d2Element<T>
where
    T: Scalar,
{
    fn from(segment: &'a LineSegment2d<T>) -> Self {
        Self {
            vertices: [segment.start().clone(), segment.end().clone()],
        }
    }
}

impl<'a, T: Scalar> From<&'a Segment2d2Element<T>> for LineSegment2d<T> {
    fn from(element: &'a Segment2d2Element<T>) -> Self {
        LineSegment2d::from_end_points(element.vertices()[0].clone(), element.vertices()[1].clone())
    }
}

#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
fn segment2_basis<T: Real>(xi: T) -> OMatrix<T, U1, U2> {
    let phi_1 = (1.0 - xi) / 2.0;
    let phi_2 = (1.0 + xi) / 2.0;
    OMatrix::<_, U1, U2>::new(phi_1, phi_2)
}

#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
fn segment2_gradients<T: Real>() -> OMatrix<T, U1, U2> {
    OMatrix::<_, U1, U2>::new(-0.5, 0.5)
}

impl<T> FixedNodesReferenceFiniteElement<T> for Segment2d1Element<T>
where
    T: Real,
{
    type NodalDim = U2;
    type ReferenceDim = U1;

    fn evaluate_basis(&self, xi: &Point1<T>) -> OMatrix<T, U1, U2> {
        segment2_basis(xi[0])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, _xi: &Point1<T>) -> OMatrix<T, U1, U2> {
        segment2_gradients()
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Segment2d2Element<T>
where
    T: Real,
{
    type NodalDim = U2;
    type ReferenceDim = U1;

    fn evaluate_basis(&self, xi: &Point1<T>) -> OMatrix<T, U1, U2> {
        segment2_basis(xi[0])
    }

    fn gradients(&self, _xi: &Point1<T>) -> OMatrix<T, U1, U2> {
        segment2_gradients()
    }
}

impl<T> FiniteElement<T> for Segment2d1Element<T>
where
    T: Real,
{
    type GeometryDim = U1;

    #[allow(non_snake_case)]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn reference_jacobian(&self, _xi: &Point1<T>) -> Vector1<T> {
        let a = &self.vertices[0].coords;
        let b = &self.vertices[1].coords;
        (b - a) / 2.0
    }

    #[allow(non_snake_case)]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn map_reference_coords(&self, xi: &Point1<T>) -> Point1<T> {
        let a = &self.vertices[0].coords;
        let b = &self.vertices[1].coords;
        let phi = self.evaluate_basis(xi);
        OPoint::from(a * phi[0] + b * phi[1])
    }

    fn diameter(&self) -> T {
        (self.vertices[1] - self.vertices[0]).norm()
    }
}

impl<T> FiniteElement<T> for Segment2d2Element<T>
where
    T: Real,
{
    type GeometryDim = U2;

    #[allow(non_snake_case)]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn reference_jacobian(&self, _xi: &Point1<T>) -> Vector2<T> {
        let a = &self.vertices[0].coords;
        let b = &self.vertices[1].coords;
        (b - a) / 2.0
    }

    #[allow(non_snake_case)]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn map_reference_coords(&self, xi: &Point1<T>) -> Point2<T> {
        let a = &self.vertices[0].coords;
        let b = &self.vertices[1].coords;
        let phi = self.evaluate_basis(xi);
        OPoint::from(a * phi[0] + b * phi[1])
    }

    fn diameter(&self) -> T {
        let s: &Segment2d2Element<T> = self;
        let line_segment: LineSegment2d<T> = s.into();
        line_segment.length()
    }
}

impl<T> SurfaceFiniteElement<T> for Segment2d2Element<T>
where
    T: Real,
{
    fn normal(&self, _xi: &Point1<T>) -> Vector2<T> {
        self.to_line_segment().normal_dir().normalize()
    }
}

impl<T> ElementConnectivity<T> for Segment2d2Connectivity
where
    T: Real,
{
    type Element = Segment2d2Element<T>;
    type ReferenceDim = U1;
    type GeometryDim = U2;

    fn element(&self, vertices: &[Point2<T>]) -> Option<Self::Element> {
        let a = vertices[self.0[0]].clone();
        let b = vertices[self.0[1]].clone();
        let segment = LineSegment2d::from_end_points(a, b);
        Some(Segment2d2Element::from(segment))
    }
}

impl<T> ElementConnectivity<T> for Segment2d1Connectivity
where
    T: Real,
{
    type Element = Segment2d1Element<T>;
    type GeometryDim = U1;
    type ReferenceDim = U1;

    fn element(&self, vertices: &[Point1<T>]) -> Option<Self::Element> {
        let a = vertices[self.0[0]].clone();
        let b = vertices[self.0[1]].clone();
        Some(Segment2d1Element::from_vertices([a, b]))
    }
}

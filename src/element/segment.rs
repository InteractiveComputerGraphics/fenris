use numeric_literals::replace_float_literals;
use crate::connectivity::Segment2d2Connectivity;
use crate::element::{ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement, SurfaceFiniteElement};
use crate::geometry::LineSegment2d;
use crate::nalgebra::{OMatrix, OPoint, Point1, Point2, RealField, Scalar, Vector2, U1, U2};

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
        Self {
            vertices: [segment.from().clone(), segment.to().clone()],
        }
    }
}

impl<'a, T: Scalar> From<&'a Segment2d2Element<T>> for LineSegment2d<T> {
    fn from(element: &'a Segment2d2Element<T>) -> Self {
        LineSegment2d::new(element.vertices()[0].clone(), element.vertices()[1].clone())
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
    fn evaluate_basis(&self, xi: &Point1<T>) -> OMatrix<T, U1, U2> {
        // xi is a scalar
        let xi = xi.x;
        let phi_1 = (1.0 - xi) / 2.0;
        let phi_2 = (1.0 + xi) / 2.0;
        OMatrix::<_, U1, U2>::new(phi_1, phi_2)
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, _xi: &Point1<T>) -> OMatrix<T, U1, U2> {
        OMatrix::<_, U1, U2>::new(-0.5, 0.5)
    }
}

impl<T> FiniteElement<T> for Segment2d2Element<T>
where
    T: RealField,
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
    T: RealField,
{
    fn normal(&self, _xi: &Point1<T>) -> Vector2<T> {
        self.to_line_segment().normal_dir().normalize()
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

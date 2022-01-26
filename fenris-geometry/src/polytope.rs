use crate::{GeneralPolygon, Triangle, Triangle2d};
use itertools::Itertools;
use nalgebra::{clamp, Matrix2, Point2, RealField, Scalar, Unit, Vector2};
use numeric_literals::replace_float_literals;

/// Type used to indicate conversion failure in the presence of concavity.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ConcavePolygonError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvexPolygon<T>
where
    T: Scalar,
{
    // TODO: SmallVec?
    // Edges are implicitly represented as (i, i + 1)
    vertices: Vec<Point2<T>>,
}

#[derive(Debug, Clone)]
pub struct HalfPlane<T>
where
    T: Scalar,
{
    point: Point2<T>,
    normal: Unit<Vector2<T>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LineSegment2d<T>
where
    T: Scalar,
{
    start: Point2<T>,
    end: Point2<T>,
}

impl<T> LineSegment2d<T>
where
    T: Scalar,
{
    pub fn new(from: Point2<T>, to: Point2<T>) -> Self {
        Self { start: from, end: to }
    }

    pub fn start(&self) -> &Point2<T> {
        &self.start
    }

    pub fn end(&self) -> &Point2<T> {
        &self.end
    }

    pub fn reverse(&self) -> Self {
        LineSegment2d {
            start: self.end.clone(),
            end: self.start.clone(),
        }
    }
}

impl<T> LineSegment2d<T>
where
    T: RealField,
{
    pub fn to_line(&self) -> Line2d<T> {
        let dir = &self.end - &self.start;
        Line2d::from_point_and_dir(self.start.clone(), dir)
    }

    /// Returns a vector tangent to the line segment.
    ///
    /// Note that the vector is **not** normalized.
    pub fn tangent_dir(&self) -> Vector2<T> {
        self.end().coords - self.start().coords
    }

    /// Returns a vector normal to the line segment, in the direction consistent with a
    /// counter-clockwise winding order when the edge is part of a polygon.
    ///
    /// Note that the vector is **not** normalized.
    pub fn normal_dir(&self) -> Vector2<T> {
        let tangent = self.tangent_dir();
        Vector2::new(tangent.y, -tangent.x)
    }

    pub fn length(&self) -> T {
        self.tangent_dir().norm()
    }

    pub fn midpoint(&self) -> Point2<T> {
        Point2::from((self.start.coords + self.end.coords) / (T::one() + T::one()))
    }

    pub fn intersect_line_parametric(&self, line: &Line2d<T>) -> Option<T> {
        self.to_line()
            .intersect_line_parametric(line)
            .map(|(t1, _)| t1)
    }

    /// Compute the closest point on the segment to the given point, represented in
    /// the parametric form x = a + t * (b - a).
    pub fn closest_point_parametric(&self, point: &Point2<T>) -> T {
        let t = self.to_line().project_point_parametric(point);
        clamp(t, T::zero(), T::one())
    }

    /// Computes the closest point on the line to the given point.
    pub fn closest_point(&self, point: &Point2<T>) -> Point2<T> {
        let t = self.closest_point_parametric(point);
        self.point_from_parameter(t)
    }

    pub fn point_from_parameter(&self, t: T) -> Point2<T> {
        Point2::from(self.start().coords + (self.end() - self.start()) * t)
    }

    pub fn segment_from_parameters(&self, t_begin: &T, t_end: &T) -> Self {
        let begin = self.point_from_parameter(t_begin.clone());
        let end = self.point_from_parameter(t_end.clone());
        Self::new(begin, end)
    }

    /// Computes the intersection of two line segments (if any), but returns the result as a parameter.
    ///
    /// Let all points on this line segment be defined by the relation x = a + t * (b - a)
    /// for 0 <= t <= 1. Then, if the two line segments intersect, t is returned. Otherwise,
    /// `None` is returned.
    pub fn intersect_segment_parametric(&self, other: &LineSegment2d<T>) -> Option<T> {
        // Represent the two lines as:
        //  x1 = a1 + t1 * d1
        //  x2 = a2 + t2 * d2
        // where di = bi - ai. This gives the linear system
        //  [ d1  -d2 ] t = a2 - a1,
        // where t = [t1, t2].

        let d1 = &self.end - &self.start;
        let d2 = &other.end - &other.start;

        let line1 = Line2d::from_point_and_dir(self.start.clone(), d1);
        let line2 = Line2d::from_point_and_dir(other.start.clone(), d2);

        line1
            .intersect_line_parametric(&line2)
            .and_then(|(t1, t2)| {
                // TODO: This may go very wrong if we're talking "exact" intersection
                // e.g. when a line segment intersects another segment only at a point,
                // in which case we might discard the intersection entirely.
                // I suppose the only way to deal with this is either arbitrary precision
                // or using epsilons? Also, keep in mind that the `from` and `to`
                // points may already be suffering from imprecision!
                if t2 < T::zero() || t2 > T::one() {
                    None
                } else if t1 < T::zero() || t1 > T::one() {
                    None
                } else {
                    Some(t1)
                }
            })
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn intersect_half_plane(&self, half_plane: &HalfPlane<T>) -> Option<Self> {
        let contains_start = half_plane.contains_point(self.start());
        let contains_end = half_plane.contains_point(self.end());

        match (contains_start, contains_end) {
            (true, true) => Some(self.clone()),
            (false, false) => None,
            (true, false) | (false, true) => {
                let t_intersect = self.intersect_line_parametric(&half_plane.surface())
                    // Technically the intersection should be in the interval [0, 1] already,
                    // but numerical errors may lead to values that are slightly outside, or, in the case of
                    // very nearly parallel lines, far outside.
                    .map(|t| clamp(t, 0.0, 1.0));

                let (t_start, t_end);
                if contains_start {
                    // The only case when the intersection returns None is when the half-plane line and the
                    // line segment are parallel, which we *technically* have excluded already.
                    // But due to floating-point imprecision we might still find ourselves in this situation.
                    // In this case the result may be more or less arbitrary, so we pick a reasonable default
                    // to fall back on
                    t_start = 0.0;
                    t_end = t_intersect.unwrap_or(1.0);
                } else {
                    t_start = t_intersect.unwrap_or(0.0);
                    t_end = 1.0;
                }

                Some(self.segment_from_parameters(&t_start, &t_end))
            },
        }
    }

    pub fn intersect_polygon(&self, other: &ConvexPolygon<T>) -> Option<LineSegment2d<T>> {
        let mut min = None;
        let mut max = None;

        let contains_start = other.contains_point(self.start());
        let contains_end = other.contains_point(self.end());
        let contained_in_poly = contains_start && contains_end;

        if contains_start {
            min = Some(T::zero());
        }
        if contains_end {
            max = Some(T::one());
        }

        if !contained_in_poly {
            for edge in other.edges() {
                let edge_segment = LineSegment2d::new(*edge.0, *edge.1);

                if let Some(t) = self.intersect_segment_parametric(&edge_segment) {
                    if t < *min.get_or_insert(t) {
                        min = Some(t);
                    }

                    if t > *max.get_or_insert(t) {
                        max = Some(t)
                    }
                }
            }
        }

        // TODO: I think this *can* actually occur if the polygon is e.g. a point
        assert!(min.is_none() == max.is_none());

        // Once we have t_min and t_max (or we don't and we return None),
        // we construct the resulting line segment
        min.and_then(|min| max.and_then(|max| Some((min, max))))
            .map(|(t_min, t_max)| {
                let a = self.start();
                let b = self.end();
                let d = b - a;
                debug_assert!(t_min <= t_max);
                LineSegment2d::new(a + d * t_min, a + d * t_max)
            })
    }
}

impl<T> From<LineSegment2d<T>> for ConvexPolygon<T>
where
    T: Scalar,
{
    fn from(segment: LineSegment2d<T>) -> Self {
        ConvexPolygon::from_vertices(vec![segment.start, segment.end])
    }
}

#[derive(Debug, Clone)]
pub struct Line2d<T>
where
    T: Scalar,
{
    point: Point2<T>,
    dir: Vector2<T>,
}

impl<T> Line2d<T>
where
    T: Scalar,
{
    pub fn from_point_and_dir(point: Point2<T>, dir: Vector2<T>) -> Self {
        // TODO: Make dir Unit?
        Self { point, dir }
    }
}

impl<T> Line2d<T>
where
    T: RealField,
{
    /// A normalized vector tangent to the line.
    pub fn tangent(&self) -> Vector2<T> {
        self.dir.normalize()
    }

    pub fn from_point_through_point(point: Point2<T>, through: &Point2<T>) -> Self {
        let dir = through - &point;
        Self::from_point_and_dir(point, dir)
    }

    /// Computes the projection of the given point onto the line, representing the point
    /// in parametric form.
    pub fn project_point_parametric(&self, point: &Point2<T>) -> T {
        let d2 = self.dir.magnitude_squared();
        (point - &self.point).dot(&self.dir) / d2
    }

    /// Computes the projection of the given point onto the line.
    pub fn project_point(&self, point: &Point2<T>) -> Point2<T> {
        let t = self.project_point_parametric(point);
        self.point_from_parameter(t)
    }

    pub fn point_from_parameter(&self, t: T) -> Point2<T> {
        &self.point + &self.dir * t
    }

    pub fn intersect(&self, other: &Line2d<T>) -> Option<Point2<T>> {
        self.intersect_line_parametric(other)
            .map(|(t1, _)| self.point_from_parameter(t1))
    }

    /// Computes the intersection of two lines, if any.
    ///
    /// Let all points on each line segment be defined by the relation `x1 = a1 + t1 * d1`
    /// for `0 <= t1 <= 1`, and similarly for `t2`. Here, `t1` is the parameter associated with
    /// `self`, and `t2` is the parameter associated with `other`.
    pub fn intersect_line_parametric(&self, other: &Line2d<T>) -> Option<(T, T)> {
        // Represent the two lines as:
        //  x1 = a1 + t1 * d1
        //  x2 = a2 + t2 * d2
        // where di = bi - ai. This gives the linear system
        //  [ d1  -d2 ] t = a2 - a1,
        // where t = [t1, t2].

        let rhs = &other.point - &self.point;
        let matrix = Matrix2::from_columns(&[self.dir, -other.dir]);

        // TODO: Rewrite to use LU decomposition?
        matrix
            .try_inverse()
            .map(|inv| inv * rhs)
            // Inverse returns vector, split it up into its components
            .map(|t| (t.x, t.y))
    }
}

impl<T> HalfPlane<T>
where
    T: RealField,
{
    /// Construct a half plane from a point on its surface and an *outward-facing* normal vector.
    pub fn from_point_and_normal(point: Point2<T>, normal: Unit<Vector2<T>>) -> Self {
        Self { point, normal }
    }
}

impl<T> HalfPlane<T>
where
    T: RealField,
{
    pub fn contains_point(&self, point: &Point2<T>) -> bool {
        self.signed_distance_to_point(point) <= T::zero()
    }

    pub fn signed_distance_to_point(&self, point: &Point2<T>) -> T {
        let d = point - &self.point;
        self.normal.dot(&d)
    }

    pub fn point(&self) -> &Point2<T> {
        &self.point
    }

    /// Returns the outwards-facing normal vector for the plane.
    ///
    /// This vector is normalized.
    pub fn normal(&self) -> &Vector2<T> {
        &self.normal
    }

    /// Returns a line representing the surface of the half plane
    pub fn surface(&self) -> Line2d<T> {
        let tangent = Vector2::new(self.normal.y, -self.normal.x);
        Line2d::from_point_and_dir(self.point.clone(), tangent)
    }
}

impl<T> ConvexPolygon<T>
where
    T: Scalar,
{
    /// Construct a new convex polygon from the given vertices, assumed to be ordered in a
    /// counter-clockwise way such that (i, i + 1) forms an edge between vertex i and i + 1.
    ///
    /// It is assumed that the polygon is convex.
    pub fn from_vertices(vertices: Vec<Point2<T>>) -> ConvexPolygon<T> {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point2<T>] {
        &self.vertices
    }

    /// Returns the number of edges in the polygon. Note that a single point has 1 edge,
    /// pointing from itself to itself, a line segment has two edges, and in general
    /// the number of edges is equal to the number of vertices.
    pub fn num_edges(&self) -> usize {
        self.vertices().len()
    }

    pub fn edges(&self) -> impl Iterator<Item = (&Point2<T>, &Point2<T>)> {
        let num_vertices = self.vertices().len();
        self.vertices()
            .iter()
            .cycle()
            .take(num_vertices + 1)
            .tuple_windows()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    pub fn is_point(&self) -> bool {
        self.vertices.len() == 1
    }

    pub fn is_line_segment(&self) -> bool {
        self.vertices.len() == 2
    }
}

impl<T> ConvexPolygon<T>
where
    T: RealField,
{
    /// Iterates over the half planes that define the polygon.
    ///
    /// Every non-degenerate polygon can be represented by the intersection of a finite number
    /// of closed half-planes.
    ///
    /// If the polygon is degenerate, the intersection of the half planes returned by this method
    /// will in general not be sufficient to describe the polygon.
    pub fn half_planes<'a>(&'a self) -> impl Iterator<Item = HalfPlane<T>> + 'a {
        self.edges().filter_map(|(v1, v2)| {
            if v1 != v2 {
                let edge_dir = v2 - v1;
                let edge_normal = Vector2::new(edge_dir.y, -edge_dir.x);
                Some(HalfPlane::from_point_and_normal(*v1, Unit::new_normalize(edge_normal)))
            } else {
                None
            }
        })
    }

    /// Determines if the (closed) convex polygon contains the given point.
    pub fn contains_point(&self, point: &Point2<T>) -> bool {
        if self.is_point() {
            self.vertices.first().unwrap() == point
        } else if self.is_line_segment() {
            unimplemented!()
        } else {
            self.half_planes()
                .all(|half_plane| half_plane.contains_point(point))
        }
    }

    /// Computes the intersection with the current polygon and the given half plane,
    /// and returns a new polygon that holds the result.
    ///
    /// Note: No steps have been made to make this routine numerically robust.
    /// TODO: Make numerically robust?
    pub fn intersect_halfplane(&self, half_plane: &HalfPlane<T>) -> ConvexPolygon<T> {
        let mut new_vertices = Vec::new();

        // Handle special case of the polygon consisting of a single vertex
        if self.vertices.len() == 1 {
            let first = self.vertices().first().unwrap();
            if half_plane.contains_point(first) {
                new_vertices.push(first.clone());
            }
        } else {
            for (v1, v2) in self.edges() {
                let v1_contained = half_plane.contains_point(v1);
                let v2_contained = half_plane.contains_point(v2);
                if v1_contained {
                    new_vertices.push(v1.clone());
                }

                if (v1_contained && !v2_contained) || (!v1_contained && v2_contained) {
                    // Edge is intersected, add vertex at intersection point
                    let dir = (v2 - v1).normalize();
                    let intersection_point = half_plane
                        .surface()
                        .intersect(&Line2d::from_point_and_dir(v1.clone(), dir))
                        .expect(
                            "We already know that the line must intersect the edge, \
                             so this should work unless we have some ugly numerical \
                             artifacts.",
                        );

                    new_vertices.push(intersection_point);
                }
            }
        }

        ConvexPolygon::from_vertices(new_vertices)
    }

    /// Computes the intersection of this polygon and the given convex polygon.
    pub fn intersect_polygon(&self, other: &ConvexPolygon<T>) -> Self {
        // TODO: Deal with degeneracies
        if self.is_point() || other.is_point() {
            unimplemented!()
        } else if self.is_line_segment() {
            let segment = LineSegment2d::new(self.vertices[0], self.vertices[1]);
            segment
                .intersect_polygon(other)
                .map(|segment| ConvexPolygon::from_vertices(vec![*segment.start(), *segment.end()]))
                .unwrap_or_else(|| ConvexPolygon::from_vertices(Vec::new()))
        } else if other.is_line_segment() {
            other.intersect_polygon(self)
        } else {
            let mut result = self.clone();
            for half_plane in other.half_planes() {
                result = result.intersect_halfplane(&half_plane);
            }
            result
        }
    }

    /// Splits the convex polygon into a set of disjoint triangles that exactly cover the area of the
    /// polygon.
    pub fn triangulate<'a>(&'a self) -> impl Iterator<Item = Triangle2d<T>> + 'a {
        self.edges()
            // Use saturating subtraction so that we don't overflow and get an empty
            // iterator in the case that the polygon has no vertices
            .take(self.num_edges().saturating_sub(1))
            .skip(1)
            .map(move |(a, b)| Triangle([*self.vertices.first().unwrap(), *a, *b]))
    }

    pub fn triangulate_into_vec(&self) -> Vec<Triangle2d<T>> {
        self.triangulate().collect()
    }
}

impl<T> From<ConvexPolygon<T>> for GeneralPolygon<T>
where
    T: Scalar,
{
    fn from(poly: ConvexPolygon<T>) -> Self {
        GeneralPolygon::from_vertices(poly.vertices)
    }
}

use crate::{SimplePolygon2d, HalfPlane, Line2d, LineSegment2d, Triangle, Triangle2d};
use itertools::Itertools;
use nalgebra::{Point2, RealField, Scalar, Unit, Vector2};

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

impl<T> From<LineSegment2d<T>> for ConvexPolygon<T>
where
    T: Scalar,
{
    fn from(segment: LineSegment2d<T>) -> Self {
        ConvexPolygon::from_vertices(vec![segment.start().clone(), segment.end().clone()])
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
            let segment = LineSegment2d::from_end_points(self.vertices[0], self.vertices[1]);
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

impl<T> From<ConvexPolygon<T>> for SimplePolygon2d<T>
where
    T: Scalar,
{
    fn from(poly: ConvexPolygon<T>) -> Self {
        SimplePolygon2d::from_vertices(poly.vertices)
    }
}

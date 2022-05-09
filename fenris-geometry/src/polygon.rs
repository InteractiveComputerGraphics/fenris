use crate::{AxisAlignedBoundingBox, BoundedGeometry, Convex, Distance, HalfSpace, LineSegment2d, LineSegment3d, Orientation, Triangle};
use itertools::{izip, Itertools};
use nalgebra::{Point2, RealField, Scalar, Vector2, U2, DimName, DefaultAllocator, OPoint, U3, clamp, Vector3, Point3, Isometry3};
use serde::{Deserialize, Serialize};
use std::iter::once;
use nalgebra::allocator::Allocator;

use numeric_literals::replace_float_literals;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "OPoint<T, D>: Serialize"))]
#[serde(bound(deserialize = "OPoint<T, D>: Deserialize<'de>"))]
pub struct SimplePolygon<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    vertices: Vec<OPoint<T, D>>,
}

pub type SimplePolygon2d<T> = SimplePolygon<T, U2>;
pub type SimplePolygon3d<T> = SimplePolygon<T, U3>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ClosestEdge<T>
where
    T: Scalar,
{
    pub signed_distance: T,
    pub edge_parameter: T,
    pub edge_point: Point2<T>,
    pub edge_index: usize,
}

pub trait Polygon2d<T>
where
    T: RealField,
{
    fn vertices(&self) -> &[Point2<T>];

    fn num_edges(&self) -> usize;

    fn get_edge(&self, index: usize) -> Option<LineSegment2d<T>>;

    fn num_vertices(&self) -> usize {
        self.vertices().len()
    }

    /// Returns the given pseudonormal (angle-weighted normal) given an edge index and a parameter
    /// representing a point on edge.
    ///
    /// If t == 0, then the average normal of this edge and its predecessor neighbor is returned.
    /// If t == 1, then the average normal of this edge and its successor neighbor is returned.
    /// Otherwise the normal of the edge is returned.
    fn pseudonormal_on_edge(&self, edge_index: usize, t: T) -> Option<Vector2<T>>;

    fn for_each_edge(&self, mut func: impl FnMut(usize, LineSegment2d<T>)) {
        for edge_idx in 0..self.num_edges() {
            let segment = self
                .get_edge(edge_idx)
                .expect("Edge index must exist, given that we're in the loop body.");
            func(edge_idx, segment);
        }
    }

    fn closest_edge(&self, x: &Point2<T>) -> Option<ClosestEdge<T>> {
        let mut closest_edge_index = None;
        let mut smallest_squared_dist = T::max_value();

        self.for_each_edge(|edge_idx, edge| {
            let closest_point_on_edge = edge.closest_point(x);
            let dist2 = (x - closest_point_on_edge).magnitude_squared();
            if dist2 < smallest_squared_dist {
                closest_edge_index = Some(edge_idx);
                smallest_squared_dist = dist2;
            }
        });

        let closest_edge_index = closest_edge_index?;
        // We unwrap all the results below, because since we have a concrete index,
        // all results *must exist*, otherwise it's an error
        let closest_edge = self.get_edge(closest_edge_index).unwrap();
        let t = closest_edge.closest_point_parametric(x);
        let pseudonormal = self.pseudonormal_on_edge(closest_edge_index, t).unwrap();
        let closest_point_on_edge = closest_edge.point_from_parameter(t);
        let d = x - &closest_point_on_edge;
        let distance = d.magnitude();
        let sign = d.dot(&pseudonormal).signum();

        Some(ClosestEdge {
            signed_distance: sign * distance,
            edge_parameter: t,
            edge_point: closest_point_on_edge,
            edge_index: closest_edge_index,
        })
    }

    fn intersects_segment(&self, segment: &LineSegment2d<T>) -> bool {
        // A segment either
        //  - Intersects an edge in the polygon
        //  - Is completely contained in the polygon
        //  - Does not intersect the polygon at all
        // To determine if it is completely contained in the polygon, we keep track of
        // the closest edge to each endpoint of the segment. Technically, only one would be
        // sufficient, but due to floating-point errors there are some cases where a segment may
        // be classified as not intersecting an edge, yet only one of its endpoints will have
        // a negative signed distance to the polygon. Thus, for robustness, we compute the signed
        // distance of both endpoints.
        if self.num_edges() == 0 {
            return false;
        }

        let mut closest_edges = [0, 0];
        let mut smallest_squared_dists = [T::max_value(), T::max_value()];
        let endpoints = [*segment.start(), *segment.end()];

        let mut intersects = false;

        self.for_each_edge(|edge_idx, edge| {
            if edge.intersect_segment_parametric(segment).is_some() {
                intersects = true;
            } else {
                for (endpoint, closest_edge, smallest_dist2) in
                    izip!(&endpoints, &mut closest_edges, &mut smallest_squared_dists)
                {
                    let closest_point_on_edge = edge.closest_point(endpoint);
                    let dist2 = (endpoint - closest_point_on_edge).magnitude_squared();
                    if dist2 < *smallest_dist2 {
                        *closest_edge = edge_idx;
                        *smallest_dist2 = dist2;
                    }
                }
            }
        });

        for (endpoint, closest_edge_idx) in izip!(&endpoints, &closest_edges) {
            // We can unwrap here, because we know that the Polygon has at least one edge
            let closest_edge = self.get_edge(*closest_edge_idx).unwrap();
            let t = closest_edge.closest_point_parametric(endpoint);
            let pseudonormal = self.pseudonormal_on_edge(*closest_edge_idx, t).unwrap();
            let closest_point_on_edge = closest_edge.point_from_parameter(t);
            let sign = (endpoint - closest_point_on_edge).dot(&pseudonormal);

            if sign <= T::zero() {
                return true;
            }
        }

        false
    }

    /// Computes the signed area of the (simple) polygon.
    ///
    /// The signed area of a simple polygon is positive if the polygon has a counter-clockwise
    /// orientation, or negative if it is oriented clockwise.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn signed_area(&self) -> T {
        // The formula for the signed area of a simple polygon can easily be obtained from
        // Green's formula by rewriting the surface integral that defines its area
        // as an integral over the curve defining the polygon's boundary,
        // which furthermore can be decomposed into a sum of integrals over each edge.
        // See e.g.
        //  https://math.blogoverflow.com/2014/06/04/greens-theorem-and-area-of-polygons/
        // for details.
        let vertices = self.vertices();
        let n = vertices.len();
        let mut area = T::zero();
        for i in 0 .. n {
            let a = &vertices[(i + 0) % n].coords;
            let b = &vertices[(i + 1) % n].coords;
            area += (b.y - a.y) * (b.x + a.x);
        }
        area * 0.5
    }

    /// Computes the area of the (simple) polygon.
    fn area(&self) -> T {
        self.signed_area().abs()
    }

    fn orientation(&self) -> Orientation {
        let signed_area = self.signed_area();
        if signed_area > T::zero() {
            Orientation::Counterclockwise
        } else {
            Orientation::Clockwise
        }
    }
}

impl<T, D> SimplePolygon<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    pub fn from_vertices(vertices: Vec<OPoint<T, D>>) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[OPoint<T, D>] {
        &self.vertices
    }

    pub fn transform_vertices<F>(&mut self, mut transform: F)
        where
            F: FnMut(&mut [OPoint<T, D>]),
    {
        transform(&mut self.vertices)

        // TODO: Update acceleration structure etc., if we decide to internally use one later on
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_edges(&self) -> usize {
        self.vertices.len()
    }

    pub fn assume_convex(&self) -> Convex<&Self> {
        Convex::assume_convex(self)
    }
}

impl<T: RealField> SimplePolygon2d<T> {
    /// Apply a similarity transform in order to construct a 3D simple polygon.
    ///
    /// Each 2D vertex is implicitly assumed to have z coordinate 0.
    pub fn apply_isometry(&self, similarity: &Isometry3<T>) -> SimplePolygon3d<T> {
        let vertices = self.vertices()
            .iter()
            .map(|v| similarity * Point3::new(v.x, v.y, T::zero()))
            .collect();
        SimplePolygon3d::from_vertices(vertices)
    }
}

impl<T: RealField> SimplePolygon3d<T>
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn area_vector(&self) -> Vector3<T> {
        let vertices = self.vertices();
        let n = vertices.len();
        let mut area = Vector3::zeros();
        for i in 0 .. n {
            let v_curr = &vertices[(i + 0) % n].coords;
            let v_next = &vertices[(i + 1) % n].coords;
            area += v_curr.cross(&v_next);
        }
        area * 0.5
    }

    pub fn area(&self) -> T {
        self.area_vector().norm()
    }

    pub fn intersect_half_space(&self, half_space: &HalfSpace<T>) -> SimplePolygon3d<T> {
        let mut new_vertices = Vec::new();

        let n = self.vertices().len();
        let plane = half_space.plane();

        for (a, b) in self.vertices().iter().cycle().take(n + 1).tuple_windows() {
            let a_contained = half_space.contains_point(a);
            let b_contained = half_space.contains_point(b);

            if a_contained {
                new_vertices.push(a.clone());
            }

            if a_contained != b_contained {
                let segment = LineSegment3d::from_end_points(a.clone(), b.clone());
                // The half space intersects the line segment between a and b,
                // so must add the intersection point

                // We're exceedingly unlikely to run into the case where there is no
                // intersection, since we've already established that there *should*
                // be an intersection on this edge. This can only happen due to a
                // floating-point imprecision problem.
                // To help prevent problems of a topological nature, we compute the
                // intersection parameter for a *line* (not segment) and clamp the result
                // to the [0, 1] interval in order to always produce some kind of vertex
                // That way, even though its placement may be inaccurate, we at least
                // are doing the topologically-speaking right thing with respect to
                // the fact that either a or b is contained in the half-space
                let t = segment.to_line()
                    .intersect_plane_parametric(&plane)
                    .map(|t| clamp(t, T::zero(), T::one()))
                    .unwrap_or(T::zero());
                new_vertices.push(segment.point_from_parameter(t));
            }
        }

        Self::from_vertices(new_vertices)
    }
}


impl<T> SimplePolygon2d<T>
where
    T: Scalar,
{
    /// An iterator over edges as line segments
    pub fn edge_iter<'a>(&'a self) -> impl 'a + Iterator<Item = LineSegment2d<T>> {
        self.vertices
            .iter()
            .chain(once(self.vertices.first().unwrap()))
            .tuple_windows()
            .map(|(a, b)| LineSegment2d::from_end_points(a.clone(), b.clone()))
    }
}

impl<T> Polygon2d<T> for SimplePolygon2d<T>
where
    T: RealField,
{
    fn vertices(&self) -> &[Point2<T>] {
        &self.vertices
    }

    fn num_edges(&self) -> usize {
        self.vertices.len()
    }

    fn get_edge(&self, index: usize) -> Option<LineSegment2d<T>> {
        let a = self.vertices.get(index)?;
        let b = self.vertices.get((index + 1) % self.num_vertices())?;
        Some(LineSegment2d::from_end_points(*a, *b))
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn pseudonormal_on_edge(&self, edge_index: usize, t: T) -> Option<Vector2<T>> {
        let edge = self.get_edge(edge_index)?;
        let edge_normal = edge.normal_dir().normalize();

        // TODO: Handle potentially degenerate line segments (i.e. they degenerate to a single
        // point, and so normalization of the normal is arbitrary, if it at all works)

        let pseudonormal = if t == T::zero() {
            // Have to take care not to underflow usize, so we cannot subtract directly
            let previous_idx = ((edge_index + self.num_edges()) - 1) % self.num_edges();
            let previous_edge = self.get_edge(previous_idx)?;
            let previous_edge_normal = previous_edge.normal_dir().normalize();
            ((previous_edge_normal + edge_normal) / 2.0).normalize()
        } else if t == T::one() {
            let next_idx = (edge_index + 1) % self.num_edges();
            let next_edge = self.get_edge(next_idx)?;
            let next_edge_normal = next_edge.normal_dir().normalize();
            ((next_edge_normal + edge_normal) / 2.0).normalize()
        } else {
            edge_normal
        };

        Some(pseudonormal)
    }
}

impl<T, D> BoundedGeometry<T> for SimplePolygon<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    type Dimension = D;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, D> {
        AxisAlignedBoundingBox::from_points(self.vertices()).expect("Vertex collection must be non-empty")
    }
}

impl<T> Distance<T, Point2<T>> for SimplePolygon2d<T>
where
    T: RealField,
{
    fn distance(&self, point: &Point2<T>) -> T {
        let closest_edge = self
            .closest_edge(point)
            .expect("We don't support empty polygons at the moment (do we want to?)");
        T::max(closest_edge.signed_distance, T::zero())
    }
}

impl<'a, T, D> Convex<&'a SimplePolygon<T, D>>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// Triangulates the convex polygon by connecting the provided point with each edge.
    pub fn triangulate_at_point(&self, point: &OPoint<T, D>) -> Vec<Triangle<T, D>> {
        let Self(polygon) = self;
        let n = polygon.vertices().len();
        let p = point;

        (0 .. n)
            .map(|i| {
                let a = polygon.vertices()[(i + 0) % n].clone();
                let b = polygon.vertices()[(i + 1) % n].clone();
                Triangle([p.clone(), a, b])
            }).collect()
    }

    /// Triangulates the convex polygon by creating a triangle fan starting from its
    /// first vertex.
    pub fn triangulate(&self) -> Vec<Triangle<T, D>> {
        let Self(polygon) = self;
        let n = polygon.vertices().len();
        if n == 0 {
            return Vec::default();
        }

        let p = polygon.vertices().first().unwrap();

        (1 .. (n - 1))
            .map(|i| {
                let a = polygon.vertices()[(i + 0) % n].clone();
                let b = polygon.vertices()[(i + 1) % n].clone();
                Triangle([p.clone(), a, b])
            }).collect()
    }
}

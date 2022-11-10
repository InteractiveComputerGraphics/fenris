use fenris_traits::Real;
use nalgebra::allocator::Allocator;
use nalgebra::{
    distance_squared, DefaultAllocator, DimName, OPoint, OVector, Point2, Point3, Scalar, Unit, Vector3, U2, U3,
};
use numeric_literals::replace_float_literals;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt::Debug;

mod polygon;
mod polytope;
mod primitives;
use crate::util::index_set_nth_power_iter;
pub use polygon::*;
pub use polytope::*;
pub use primitives::*;

pub mod polymesh;
pub mod sdf;
pub mod util;

#[cfg(feature = "vtkio")]
pub mod vtkio;

#[cfg(feature = "proptest-support")]
pub mod proptest;

pub trait BoundedGeometry<T>
where
    T: Scalar,
    DefaultAllocator: Allocator<T, Self::Dimension>,
{
    type Dimension: DimName;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, Self::Dimension>;
}

pub trait Distance<T, QueryGeometry>
where
    T: Scalar,
{
    /// Returns an interval `[l, u]` for the distance `d`, such that `d` is contained in `[l, u]`.
    fn distance_bound(&self, query_geometry: &QueryGeometry) -> [T; 2] {
        let d = self.distance(query_geometry);
        [d.clone(), d]
    }

    fn distance(&self, query_geometry: &QueryGeometry) -> T;
}

pub trait GeometryCollection<'a> {
    type Geometry;

    fn num_geometries(&self) -> usize;
    fn get_geometry(&'a self, index: usize) -> Option<Self::Geometry>;
}

pub trait DistanceQuery<'a, QueryGeometry>: GeometryCollection<'a> {
    //    type KNearestIter: Iterator<Item=usize>;

    //    fn k_nearest(&'a self, query_geometry: &'a QueryGeometry, k: usize) -> Self::KNearestIter;

    fn nearest(&'a self, query_geometry: &'a QueryGeometry) -> Option<usize>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedDistanceResult<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub feature_id: usize,
    pub point: OPoint<T, D>,
    pub signed_distance: T,
}

pub trait SignedDistance<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn query_signed_distance(&self, point: &OPoint<T, D>) -> Option<SignedDistanceResult<T, D>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "OPoint<T, D>: Serialize",
    deserialize = "OPoint<T, D>: Deserialize<'de>"
))]
pub struct AxisAlignedBoundingBox<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    min: OPoint<T, D>,
    max: OPoint<T, D>,
}

impl<T, D> Copy for AxisAlignedBoundingBox<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    OPoint<T, D>: Copy,
{
}

pub type AxisAlignedBoundingBox2d<T> = AxisAlignedBoundingBox<T, U2>;
pub type AxisAlignedBoundingBox3d<T> = AxisAlignedBoundingBox<T, U3>;

impl<T, D> AxisAlignedBoundingBox<T, D>
where
    T: Scalar + PartialOrd,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn new(min: OPoint<T, D>, max: OPoint<T, D>) -> Self {
        for i in 0..D::dim() {
            assert!(min[i] <= max[i]);
        }
        Self { min, max }
    }

    pub fn min(&self) -> &OPoint<T, D> {
        &self.min
    }

    pub fn max(&self) -> &OPoint<T, D> {
        &self.max
    }
}

impl<T, D> From<OPoint<T, D>> for AxisAlignedBoundingBox<T, D>
where
    T: Scalar + PartialOrd,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn from(point: OPoint<T, D>) -> Self {
        AxisAlignedBoundingBox::new(point.clone(), point)
    }
}

impl<T, D> AxisAlignedBoundingBox<T, D>
where
    T: Real,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// Computes the minimal bounding box which encloses both `self` and `other`.
    pub fn enclose(&self, other: &AxisAlignedBoundingBox<T, D>) -> Self {
        let min = self
            .min
            .iter()
            .zip(&other.min.coords)
            .map(|(a, b)| T::min(*a, *b));
        let min = OVector::<T, D>::from_iterator(min);

        let max = self
            .max
            .iter()
            .zip(&other.max.coords)
            .map(|(a, b)| T::max(*a, *b));
        let max = OVector::<T, D>::from_iterator(max);

        AxisAlignedBoundingBox::new(min.into(), max.into())
    }

    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a OPoint<T, D>>) -> Option<Self> {
        let mut points = points.into_iter();
        points.next().map(|first_point| {
            points.fold(AxisAlignedBoundingBox::from(first_point.clone()), |aabb, point| {
                aabb.enclose(&AxisAlignedBoundingBox::from(point.clone()))
            })
        })
    }

    pub fn extents(&self) -> OVector<T, D> {
        self.max() - self.min()
    }

    pub fn max_extent(&self) -> T {
        (self.max() - self.min()).amax()
    }

    pub fn center(&self) -> OPoint<T, D> {
        OPoint::from((&self.max().coords + &self.min().coords) / T::from_f64(2.0).unwrap())
    }

    /// Uniformly scales each axis by the given scale amount, with respect to the center of
    /// the box.
    ///
    /// ```rust
    /// # use fenris_geometry::AxisAlignedBoundingBox;
    /// use nalgebra::{point, vector};
    /// use matrixcompare::assert_matrix_eq;
    ///
    /// let aabb = AxisAlignedBoundingBox::new(point![0.0, 0.0], point![1.0, 1.0]);
    /// let scaled = aabb.uniformly_scale(0.5);
    ///
    /// assert_matrix_eq!(scaled.min().coords, vector![0.25, 0.25], comp = float);
    /// assert_matrix_eq!(scaled.max().coords, vector![0.75, 0.75], comp = float);
    /// ```
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn uniformly_scale(&self, scale: T) -> Self {
        assert!(scale >= T::zero());
        let s = scale;
        let (a, b) = (&self.min, &self.max);
        let ref c = self.center();
        Self {
            min: c + (a - c) * s,
            max: c + (b - c) * s,
        }
    }

    pub fn contains_point(&self, point: &OPoint<T, D>) -> bool {
        (0..D::dim()).all(|dim| point[dim] >= self.min[dim] && point[dim] <= self.max[dim])
    }

    pub fn intersects(&self, other: &Self) -> bool {
        for i in 0..D::dim() {
            if !intervals_intersect([self.min[i], self.max[i]], [other.min[i], other.max[i]]) {
                return false;
            }
        }
        true
    }

    /// Grows the bounding box by `distance` in all directions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use fenris_geometry::AxisAlignedBoundingBox;
    /// # use nalgebra::point;
    /// let aabb = AxisAlignedBoundingBox::new(point![0.0, 0.0], point![1.0, 1.0]);
    /// let grown = aabb.grow_uniformly(1.0);
    /// assert_eq!(grown.min(), &point![-1.0, -1.0]);
    /// assert_eq!(grown.max(), &point![2.0, 2.0]);
    /// ```
    ///
    pub fn grow_uniformly(&self, distance: T) -> Self {
        let min = self.min().map(|b_i| b_i - distance);
        let max = self.max().map(|b_i| b_i + distance);
        Self::new(min, max)
    }

    /// Creates an iterator over the corners of the bounding box.
    pub fn corners_iter<'a>(&'a self) -> impl 'a + Iterator<Item = OPoint<T, D>>
    where
        DefaultAllocator: Allocator<usize, D>,
    {
        // We can enumerate the corners by looking at {0, 1}^D, i.e. the D-th power of the
        // set {0, 1}, and associating 0 and 1 with min and max coordinates for the i-th axis.
        index_set_nth_power_iter::<D>(2).map(move |multi_idx| {
            OVector::<T, D>::from_fn(|idx, _| match multi_idx[idx] {
                0 => self.min[idx].clone(),
                1 => self.max[idx].clone(),
                _ => unreachable!(),
            })
            .into()
        })
    }

    /// Computes the point in the bounding box closest to the given point.
    pub fn closest_point_to(&self, point: &OPoint<T, D>) -> OPoint<T, D> {
        point
            .coords
            .zip_zip_map(&self.min.coords, &self.max.coords, |p_i, a_i, b_i| {
                if p_i <= a_i {
                    a_i
                } else if p_i >= b_i {
                    b_i
                } else {
                    p_i
                }
            })
            .into()
    }

    /// Computes the distance between the bounding box and the given point.
    pub fn dist_to(&self, point: &OPoint<T, D>) -> T {
        self.dist2_to(point).sqrt()
    }

    /// Computes the squared distance between the bounding box and the given point.
    pub fn dist2_to(&self, point: &OPoint<T, D>) -> T {
        (self.closest_point_to(point) - point).norm_squared()
    }

    /// Compute the point in the bounding box furthest away from the given point.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn furthest_point_to(&self, point: &OPoint<T, D>) -> OPoint<T, D>
    where
        DefaultAllocator: Allocator<usize, D>,
    {
        // It turns out that we can choose, along each dimension, the point in the interval
        // [a_i, b_i] furthest away from p_i.
        point
            .coords
            .zip_zip_map(&self.min.coords, &self.max.coords, |p_i, a_i, b_i| {
                let mid = (a_i + b_i) / 2.0;
                if p_i < mid {
                    b_i
                } else {
                    a_i
                }
            })
            .into()
    }

    /// The squared distance to the point in the bounding box furthest away from the given point.
    ///
    /// # Panics
    ///
    /// Panic behavior is identical to [`furthest_point_to`](Self::furthest_point_to).
    pub fn max_dist2_to(&self, point: &OPoint<T, D>) -> T
    where
        // TODO: Use DimAllocator and SmallDim
        DefaultAllocator: Allocator<usize, D>,
    {
        (self.furthest_point_to(point) - point).norm_squared()
    }

    /// The distance to the point in the bounding box furthest away from the given point.
    ///
    /// # Panics
    ///
    /// Panic behavior is identical to [`max_dist2_to`](Self::max_dist2_to).
    pub fn max_dist_to(&self, point: &OPoint<T, D>) -> T
    where
        // TODO: Use DimAllocator and SmallDim
        DefaultAllocator: Allocator<usize, D>,
    {
        self.max_dist2_to(point).sqrt()
    }
}

fn intervals_intersect<T: Real>([l1, u1]: [T; 2], [l2, u2]: [T; 2]) -> bool {
    l2 <= u1 && u2 >= l1
}

impl<T, D> BoundedGeometry<T> for AxisAlignedBoundingBox<T, D>
where
    T: Real,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Dimension = D;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, D> {
        self.clone()
    }
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub enum Orientation {
    Clockwise,
    Counterclockwise,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OrientationTestResult {
    Positive,
    Zero,
    Negative,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolygonClosestPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub closest_point: OPoint<T, D>,
    pub distance: T,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct PolygonPointProjection3d<T: Scalar> {
    /// The projection of the point onto the polygon.
    pub projected_point: Point3<T>,
    /// The (absolute) distance from the point to the projected point.
    pub distance: T,
    /// The signed distance from the point to the polygon plane. The sign is positive
    /// if the vector from the plane projection to the point is in the direction of the polygon
    /// normal.
    pub signed_plane_distance: T,
}

/// A convex polygon in 3D.
///
/// Vertices are assumed to be ordered counter-clockwise.
pub trait ConvexPolygon3d<'a, T: Scalar>: Debug {
    fn num_vertices(&self) -> usize;
    fn get_vertex(&self, index: usize) -> Option<Point3<T>>;

    fn compute_plane(&self) -> Option<Plane<T>>
    where
        T: Real,
    {
        let normal = self.compute_normal();
        let point = self.get_vertex(0)?;
        Some(Plane::from_point_and_normal(point, Unit::new_normalize(normal)))
    }

    fn compute_half_space(&self) -> Option<HalfSpace<T>>
    where
        T: Real,
    {
        let normal = self.compute_normal();
        let point = self.get_vertex(0)?;
        Some(HalfSpace::from_point_and_normal(point, Unit::new_unchecked(-normal)))
    }

    /// Computes a vector normal to the polygon (oriented outwards w.r.t. a counter-clockwise
    /// orientation), whose absolute magnitude is the area of the polygon.
    ///
    /// Note that if the polygon is non-planar or not properly oriented, there are no
    /// guarantees on the quality of the result.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn compute_area_vector(&self) -> Vector3<T>
    where
        T: Real,
    {
        assert!(self.num_vertices() >= 3, "Polygons must have at least 3 vertices.");

        let mut area_vector = Vector3::zeros();

        let a = self.get_vertex(0).unwrap();
        for i in 1..self.num_vertices() - 1 {
            let b = self.get_vertex(i).unwrap();
            let c = self.get_vertex(i + 1).unwrap();
            let ab = &b.coords - &a.coords;
            let ac = &c.coords - &a.coords;
            area_vector += ab.cross(&ac) * 0.5;
        }

        area_vector
    }

    /// Computes an outwards-facing normalized vector perpendicular to the polygon.
    fn compute_normal(&self) -> Vector3<T>
    where
        T: Real,
    {
        // In principle, we could compute the face normal simply by
        // taking the cross product of the first two segments. However, the first M segments
        // may all be co-linear, in which case this will fail. Of course, *all* segments
        // may be co-linear, in which we can't do anything about it.
        //
        // To do this robustly, we instead consider a triangle fan starting in the first vertex.
        // For each triangle ABC, we compute the quantity AB x AC. Taking the sum of all these
        // quantities and finally normalizing it should give a fairly reliable quantity,
        // given that the polygon itself is not degenerate.

        self.compute_area_vector().normalize()
    }

    fn closest_point(&self, point: &Point3<T>) -> PolygonClosestPoint<T, U3>
    where
        T: Real,
    {
        assert!(self.num_vertices() >= 3, "Polygon must have at least 3 vertices.");

        // First, "extrude" the polygon by extruding each edge perpendicular to the
        // face. Then check if the point is contained in this extruded prism
        // by checking it against all half-spaces defined by the extruded edges.
        let n = self.compute_normal();

        let mut inside = true;

        for i in 0..self.num_vertices() {
            let v1 = self.get_vertex(i).unwrap();
            let v2 = self.get_vertex((i + 1) % self.num_vertices()).unwrap();

            // Vector parallel to edge
            let e = &v2.coords - &v1.coords;

            // Half space normal points towards the interior of the polygon
            // TODO: This currently assumes a normal direction where the polygon is
            // *clockwise* oriented, which is weird. Things are a little
            // inconsistent right now, gotta fix
            let half_space_normal = Unit::new_normalize(e.cross(&n));
            let half_space = HalfSpace::from_point_and_normal(v1, half_space_normal);

            if !half_space.contains_point(point) {
                inside = false;
                break;
            }
        }

        if inside {
            // Point is contained inside the extruded prism, so the projection onto the
            // polygon is simply a projection onto the polygon plane.

            // Pick any point in the polygon plane
            let x0 = self.get_vertex(0).unwrap();
            // TODO: Use methods on Plane
            let signed_plane_distance = n.dot(&(point - x0));
            let closest_point = point - &n * signed_plane_distance;
            PolygonClosestPoint {
                closest_point,
                distance: signed_plane_distance.abs(),
            }
        } else {
            // Point is *not* contained inside the extruded prism. Thus we must pick the
            // closest point on any of the edges of the polygon.

            let mut closest_dist2 = T::max_value().unwrap();
            let mut closest_point = Point3::origin();

            for i in 0..self.num_vertices() {
                let v1 = self.get_vertex(i).unwrap();
                let v2 = self.get_vertex((i + 1) % self.num_vertices()).unwrap();
                let segment = LineSegment3d::from_end_points(v1, v2);
                let projected = segment.closest_point(point);
                let dist2 = distance_squared(&projected, point);

                if dist2 < closest_dist2 {
                    closest_dist2 = dist2;
                    closest_point = projected;
                }
            }

            PolygonClosestPoint {
                closest_point,
                distance: closest_dist2.sqrt(),
            }
        }
    }
}

pub trait ConvexPolyhedron<'a, T: Scalar>: Debug {
    type Face: ConvexPolygon3d<'a, T>;

    fn num_faces(&self) -> usize;
    fn get_face(&self, index: usize) -> Option<Self::Face>;

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn compute_signed_distance(&self, point: &Point3<T>) -> SignedDistanceResult<T, U3>
    where
        T: Real,
    {
        assert!(self.num_faces() >= 4, "Polyhedron must have at least 4 faces.");
        let mut inside = true;
        let mut closest_dist = T::max_value().unwrap();
        let mut closest_point = Point3::origin();
        let mut closest_face_index = 0;

        for i in 0..self.num_faces() {
            let face = self.get_face(i).unwrap();
            let closest_point_result = face.closest_point(point);

            if closest_point_result.distance < closest_dist {
                closest_dist = closest_point_result.distance;
                closest_face_index = i;
                closest_point = closest_point_result.closest_point;
            }

            let n = face.compute_normal();
            let x0 = closest_point_result.closest_point;
            // If the point is outside any of the half-spaces defined by the negative face normals,
            // the point must be outside the polyhedron
            let half_space = HalfSpace::from_point_and_normal(x0, Unit::new_unchecked(-n));
            if !half_space.contains_point(&point) {
                inside = false;
            }
        }

        let sign = if inside { -1.0 } else { 1.0 };

        debug_assert!(closest_dist >= 0.0);
        SignedDistanceResult {
            feature_id: closest_face_index,
            point: closest_point,
            signed_distance: sign * closest_dist,
        }
    }

    fn compute_volume(&'a self) -> T
    where
        T: Real,
    {
        let faces = (0..self.num_faces()).map(move |idx| {
            self.get_face(idx)
                .expect("Number of faces reported must be correct.")
        });
        compute_polyhedron_volume_from_faces(faces)
    }

    /// Check if this polyhedron contains the given point.
    ///
    /// TODO: Write tests
    fn contains_point(&'a self, point: &Point3<T>) -> bool
    where
        T: Real,
    {
        // The convex polyhedron contains the point if all half-spaces associated with
        // faces contain the point
        for i in 0..self.num_faces() {
            let face = self.get_face(i).unwrap();
            let n = face.compute_normal();
            let x0 = face
                .get_vertex(0)
                .expect("TODO: How to handle empty polygon?");
            // Half-space normal must point opposite of the face normal
            let half_space = HalfSpace::from_point_and_normal(x0, Unit::new_unchecked(-n));
            if !half_space.contains_point(point) {
                return false;
            }
        }

        true
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn compute_polyhedron_volume_from_faces<'a, T, F>(boundary_faces: impl 'a + IntoIterator<Item = F>) -> T
where
    T: Real,
    F: ConvexPolygon3d<'a, T>,
{
    // We use the formula given on the Wikipedia page for Polyhedra:
    // https://en.wikipedia.org/wiki/Polyhedron#Volume

    let mut volume = T::zero();
    for face in boundary_faces {
        // Ignore degenerate faces consisting of zero vertices
        // TODO: Handle this different somehow? It's a little inconsistent what we
        // require in various methods
        if face.num_vertices() > 2 {
            let x0 = face.get_vertex(0).unwrap();
            // TODO: Be consistent about what direction normal should have!
            let area_vector = face.compute_area_vector();
            let area = area_vector.magnitude();
            if area > T::zero() {
                let normal = area_vector.normalize();
                volume += (normal.dot(&x0.coords)) * area;
            }
        }
    }

    // The absolute value should negate issues caused by flipped normals,
    // as long as the normals are consistently oriented
    volume = volume.abs() / 3.0;

    volume
}

// TODO: Actually implement SignedDistance for Tetrahedron
//impl<T> SignedDistance<T, U3> for Tetrahedron<T>
//where
//    T: Real
//{
//    #[replace_float_literals(T::from_f64(literal).unwrap())]
//    fn query_signed_distance(&self, point: &Point3<T>) -> Option<SignedDistanceResult<T, U3>> {
//        let triangle = |i, j, k| Triangle([self.vertices[i], self.vertices[j], self.vertices[k]]);
//
//        let tri_faces = [
//            // We must carefully choose the ordering of vertices so that the
//            // resulting faces have outwards-pointing normals
//            triangle(2, 1, 0),
//            triangle(1, 2, 3),
//            triangle(0, 1, 3),
//            triangle(2, 0, 3)
//        ];
//
//        let mut point_inside = true;
//        let mut min_dist = T::max_value();
//
//        for tri_face in &tri_faces {
//            // Remember that the triangles are oriented such that *outwards* is the positive
//            // direction, so for the point to be inside of the cell, we need its orientation
//            // with respect to each face to be *negative*
//            if tri_face.point_orientation(point) == OrientationTestResult::Positive {
//                point_inside = false;
//            }
//
//            min_dist = T::min(min_dist, tri_face.distance(point));
//        }
//
//        let sign = if point_inside { -1.0 } else { 1.0 };
//
//        SignedDistanceResult {
//            feature_id: 0,
//            point: Point {},
//            signed_distance: ()
//        }
//    }
//}

impl<T> From<Triangle2d<T>> for ConvexPolygon<T>
where
    T: Scalar,
{
    fn from(triangle: Triangle2d<T>) -> Self {
        // TODO: Use Point2 in Mesh
        ConvexPolygon::from_vertices(triangle.0.iter().map(|v| Point2::from(v.clone())).collect())
    }
}

impl<T> TryFrom<Quad2d<T>> for ConvexPolygon<T>
where
    T: Real,
{
    type Error = ConcavePolygonError;

    fn try_from(value: Quad2d<T>) -> Result<Self, Self::Error> {
        if value.concave_corner().is_none() {
            // TODO: Change Quad2d to have Point2 members instead of Vector2
            Ok(ConvexPolygon::from_vertices(
                value.0.iter().map(|v| Point2::from(v.clone())).collect(),
            ))
        } else {
            Err(ConcavePolygonError)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Convex<T>(pub T);

impl<T> Convex<T> {
    pub fn assume_convex(obj: T) -> Self {
        Self(obj)
    }
}

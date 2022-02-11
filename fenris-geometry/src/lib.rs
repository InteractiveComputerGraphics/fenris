use nalgebra::allocator::Allocator;
use nalgebra::{
    distance_squared, DefaultAllocator, DimName, OPoint, OVector, Point2, Point3, RealField, Scalar, Unit, Vector3, U2,
    U3,
};
use numeric_literals::replace_float_literals;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::fmt::Debug;

mod polygon;
mod polytope;
mod primitives;
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
    serialize = "OVector<T, D>: Serialize",
    deserialize = "OVector<T, D>: Deserialize<'de>"
))]
pub struct AxisAlignedBoundingBox<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    min: OVector<T, D>,
    max: OVector<T, D>,
}

impl<T, D> Copy for AxisAlignedBoundingBox<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    OVector<T, D>: Copy,
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
    pub fn new(min: OVector<T, D>, max: OVector<T, D>) -> Self {
        for i in 0..D::dim() {
            assert!(min[i] <= max[i]);
        }
        Self { min, max }
    }

    pub fn min(&self) -> &OVector<T, D> {
        &self.min
    }

    pub fn max(&self) -> &OVector<T, D> {
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
        AxisAlignedBoundingBox::new(point.coords.clone(), point.coords)
    }
}

impl<T, D> AxisAlignedBoundingBox<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// Computes the minimal bounding box which encloses both `this` and `other`.
    pub fn enclose(&self, other: &AxisAlignedBoundingBox<T, D>) -> Self {
        let min = self.min.iter().zip(&other.min).map(|(a, b)| T::min(*a, *b));
        let min = OVector::<T, D>::from_iterator(min);

        let max = self.max.iter().zip(&other.max).map(|(a, b)| T::max(*a, *b));
        let max = OVector::<T, D>::from_iterator(max);

        AxisAlignedBoundingBox::new(min, max)
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
        OPoint::from((self.max() + self.min()) / T::from_f64(2.0).unwrap())
    }

    pub fn uniformly_scale(&self, scale: T) -> Self {
        Self {
            min: &self.min * scale,
            max: &self.max * scale,
        }
    }

    pub fn contains_point(&self, point: &OPoint<T, D>) -> bool {
        (0..D::dim()).all(|dim| point[dim] > self.min[dim] && point[dim] < self.max[dim])
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
    /// # use nalgebra::vector;
    /// let aabb = AxisAlignedBoundingBox::new(vector![0.0, 0.0], vector![1.0, 1.0]);
    /// let grown = aabb.grow_uniformly(1.0);
    /// assert_eq!(grown.min(), &vector![-1.0, -1.0]);
    /// assert_eq!(grown.max(), &vector![2.0, 2.0]);
    /// ```
    ///
    pub fn grow_uniformly(&self, distance: T) -> Self {
        let min = self.min().map(|b_i| b_i - distance);
        let max = self.max().map(|b_i| b_i + distance);
        Self::new(min, max)
    }
}

fn intervals_intersect<T: RealField>([l1, u1]: [T; 2], [l2, u2]: [T; 2]) -> bool {
    l2 <= u1 && u2 >= l1
}

impl<T, D> BoundedGeometry<T> for AxisAlignedBoundingBox<T, D>
where
    T: RealField,
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

    fn compute_half_space(&self) -> Option<HalfSpace<T>>
    where
        T: RealField,
    {
        let normal = self.compute_face_normal();
        let point = self.get_vertex(0)?;
        Some(HalfSpace::from_point_and_normal(point, Unit::new_unchecked(normal)))
    }

    /// Computes a vector normal to the polygon (oriented outwards w.r.t. a counter-clockwise
    /// orientation), whose absolute magnitude is the area of the polygon.
    ///
    /// Note that if the polygon is non-planar or not properly oriented, there are no
    /// guarantees on the quality of the result.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn compute_area_vector(&self) -> Vector3<T>
    where
        T: RealField,
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

    fn compute_face_normal(&self) -> Vector3<T>
    where
        T: RealField,
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

    fn project_point(&self, point: &Point3<T>) -> PolygonPointProjection3d<T>
    where
        T: RealField,
    {
        assert!(self.num_vertices() >= 3, "Polygon must have at least 3 vertices.");

        // First, "extrude" the polygon by extruding each edge perpendicular to the
        // face. Then check if the point is contained in this extruded prism
        // by checking it against all half-spaces defined by the extruded edges.
        let n = self.compute_face_normal();

        let mut inside = true;

        for i in 0..self.num_vertices() {
            let v1 = self.get_vertex(i).unwrap();
            let v2 = self.get_vertex((i + 1) % self.num_vertices()).unwrap();

            // Vector parallel to edge
            let e = &v2.coords - &v1.coords;

            // Half space normal points towards the interior of the polygon
            let half_space_normal = Unit::new_normalize(-e.cross(&n));
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
            let signed_plane_distance = n.dot(&(point - x0));
            let projected_point = point - &n * signed_plane_distance;
            PolygonPointProjection3d {
                projected_point,
                signed_plane_distance,
                // the projected point is equal to the projection onto the plane
                distance: signed_plane_distance.abs(),
            }
        } else {
            // Point is *not* contained inside the extruded prism. Thus we must pick the
            // closest point on any of the edges of the polygon.

            let mut closest_dist2 = T::max_value();
            let mut closest_point = Point3::origin();

            for i in 0..self.num_vertices() {
                let v1 = self.get_vertex(i).unwrap();
                let v2 = self.get_vertex((i + 1) % self.num_vertices()).unwrap();
                let segment = LineSegment3d::from_end_points([v1, v2]);
                let projected = segment.project_point(point);
                let dist2 = distance_squared(&projected, point);

                if dist2 < closest_dist2 {
                    closest_dist2 = dist2;
                    closest_point = projected;
                }
            }

            let signed_plane_distance = n.dot(&(point - &closest_point));
            PolygonPointProjection3d {
                projected_point: closest_point,
                signed_plane_distance,
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
        T: RealField,
    {
        assert!(self.num_faces() >= 4, "Polyhedron must have at least 4 faces.");
        let mut inside = true;
        let mut closest_dist = T::max_value();
        let mut closest_point = Point3::origin();
        let mut closest_face_index = 0;

        for i in 0..self.num_faces() {
            let face = self.get_face(i).unwrap();
            let projection = face.project_point(point);

            if projection.distance < closest_dist {
                closest_dist = projection.distance;
                closest_face_index = i;
                closest_point = projection.projected_point;
            }

            // If the point is outside any of the half-spaces defined by the faces,
            // the point must be outside the polyhedron
            if projection.signed_plane_distance < T::zero() {
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
        T: RealField,
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
        T: RealField,
    {
        // The convex polyhedron contains the point if all half-spaces associated with
        // faces contain the point
        for i in 0..self.num_faces() {
            let face = self.get_face(i).unwrap();
            let half_space = face
                .compute_half_space()
                .expect("TODO: What to do if we cannot compute half space?");

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
    T: RealField,
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
//    T: RealField
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
    T: RealField,
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

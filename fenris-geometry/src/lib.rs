mod polytope;
use itertools::izip;
use nalgebra::{
    distance_squared, DefaultAllocator, DimName, OPoint, OVector, Point2, Point3, RealField, Scalar, Unit, Vector3, U2,
    U3,
};
pub use polytope::*;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

mod polygon;
pub use polygon::*;

use nalgebra::allocator::Allocator;
use numeric_literals::replace_float_literals;
use std::fmt::Debug;

pub mod polymesh;
pub mod sdf;

#[cfg(feature = "vtkio")]
pub mod vtkio;

#[cfg(feature = "proptest")]
pub mod proptest_strategies;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "OPoint<T, D>: Serialize",
    deserialize = "OPoint<T, D>: Deserialize<'de>"
))]
pub struct Triangle<T, D>(pub [OPoint<T, D>; 3])
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>;

impl<T, D> Copy for Triangle<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    OPoint<T, D>: Copy,
{
}

/// A triangle in two dimensions, consisting of three vertices.
///
/// For most purposes, the triangle is assumed to be specified with a counter-clockwise
/// winding order, but it also provides facilities for determining the winding order
/// of an arbitrarily constructed triangle.
pub type Triangle2d<T> = Triangle<T, U2>;
pub type Triangle3d<T> = Triangle<T, U3>;

impl<T, D> Triangle<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn swap_vertices(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
    }
}

impl<T, D> Triangle<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn centroid(&self) -> OPoint<T, D> {
        let mut centroid = OVector::zeros();
        for p in &self.0 {
            centroid += &p.coords * T::from_f64(1.0 / 3.0).unwrap();
        }
        OPoint::from(centroid)
    }

    /// Returns an array of vectors corresponding to the three sides of the triangle.
    pub fn sides(&self) -> [OVector<T, D>; 3] {
        let a = &self.0[0];
        let b = &self.0[1];
        let c = &self.0[2];
        [b - a, c - b, a - c]
    }
}

impl<T> Triangle2d<T>
where
    T: RealField,
{
    pub fn orientation(&self) -> Orientation {
        if self.signed_area() >= T::zero() {
            Orientation::Clockwise
        } else {
            Orientation::Counterclockwise
        }
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T."))]
    pub fn signed_area(&self) -> T {
        let a = &self.0[0];
        let b = &self.0[1];
        let c = &self.0[2];
        let ab = b - a;
        let ac = c - a;
        0.5 * ab.perp(&ac)
    }

    pub fn area(&self) -> T {
        self.signed_area().abs()
    }
}

impl<T> Triangle3d<T>
where
    T: RealField,
{
    /// Returns a vector normal to the triangle. The vector is *not* normalized.
    pub fn normal_dir(&self) -> Vector3<T> {
        let a = &self.0[0];
        let b = &self.0[1];
        let c = &self.0[2];
        let ab = b - a;
        let ac = c - a;
        let n = ab.cross(&ac);
        n
    }

    pub fn normal(&self) -> Vector3<T> {
        self.normal_dir().normalize()
    }

    /// TODO: Remove this. It makes no sense for 3D.
    pub fn orientation(&self) -> Orientation {
        if self.signed_area() >= T::zero() {
            Orientation::Clockwise
        } else {
            Orientation::Counterclockwise
        }
    }

    /// TODO: Remove this. It makes no sense for 3D (moreover, the current implementation
    /// gives the non-negative ara in any case).
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T."))]
    pub fn signed_area(&self) -> T {
        let a = &self.0[0];
        let b = &self.0[1];
        let c = &self.0[2];
        let ab = b - a;
        let ac = c - a;
        0.5 * ab.cross(&ac).norm()
    }

    pub fn area(&self) -> T {
        self.signed_area().abs()
    }

    /// Determines the orientation of a point with respect to the plane containing the triangle.
    ///
    /// This is the common "orientation test" used in computational geometry. The test returns
    /// a value whose sign is the same as `dot(n, x - x0)`, where `x0` is the projection of
    /// the point onto the triangle plane.
    ///
    /// Note that at the moment, this implementation is **NOT** robust (in the sense of exact/robust
    /// geometric predicates).
    pub fn point_orientation(&self, point: &Point3<T>) -> OrientationTestResult {
        // Note: This is by no means robust in the sense of floating point accuracy.
        let point_in_plane = &self.0[0];

        let x = point;
        let x0 = point_in_plane;
        let n = self.normal_dir().normalize();
        let projected_dist = n.dot(&(x - x0));

        if projected_dist > T::zero() {
            OrientationTestResult::Positive
        } else if projected_dist < T::zero() {
            OrientationTestResult::Negative
        } else {
            OrientationTestResult::Zero
        }
    }
}

impl<T, D> BoundedGeometry<T> for Triangle<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Dimension = D;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, D> {
        AxisAlignedBoundingBox::from_points(&self.0).unwrap()
    }
}

impl<T> Distance<T, Point2<T>> for Triangle2d<T>
where
    T: RealField,
{
    fn distance(&self, point: &Point2<T>) -> T {
        let sdf = self.query_signed_distance(point).unwrap();
        T::max(T::zero(), sdf.signed_distance)
    }
}

impl<T> SignedDistance<T, U2> for Triangle2d<T>
where
    T: RealField,
{
    fn query_signed_distance(&self, point: &Point2<T>) -> Option<SignedDistanceResult<T, U2>> {
        // TODO: This is not the most efficient way to compute this
        let mut inside = true;

        let mut closest_segment = 0;
        let mut closest_dist2 = T::max_value();
        let mut closest_point = Point2::origin();

        // We assume counterclockwise orientation.
        for i in 0..3 {
            let a = &self.0[i];
            let b = &self.0[(i + 1) % 3];
            let segment = LineSegment2d::new(*a, *b);
            // Normal point outwards, i.e. towards the "right"
            let normal_dir = segment.normal_dir();
            let projected_point = segment.closest_point(point);
            let d = &point.coords - &projected_point.coords;

            if d.dot(&normal_dir) > T::zero() {
                inside = false;
            }

            let dist2 = d.magnitude_squared();
            if dist2 < closest_dist2 {
                closest_segment = i;
                closest_dist2 = dist2;
                closest_point = projected_point;
            }
        }

        let sign = if inside { T::from_f64(-1.0).unwrap() } else { T::one() };

        Some(SignedDistanceResult {
            feature_id: closest_segment,
            point: closest_point,
            signed_distance: sign * closest_dist2.sqrt(),
        })
    }
}

impl<T> Distance<T, Point3<T>> for Triangle3d<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn distance(&self, point: &OPoint<T, U3>) -> T {
        self.project_point(point).distance
    }
}

impl<'a, T: RealField> ConvexPolygon3d<'a, T> for Triangle3d<T> {
    fn num_vertices(&self) -> usize {
        3
    }

    fn get_vertex(&self, index: usize) -> Option<OPoint<T, U3>> {
        self.0.get(index).copied()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OrientationTestResult {
    Positive,
    Zero,
    Negative,
}

#[derive(Debug, Copy, Clone, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound(serialize = "Point3<T>: Serialize"))]
#[serde(bound(deserialize = "Point3<T>: Deserialize<'de>"))]
pub struct Tetrahedron<T>
where
    T: Scalar,
{
    // Ordering uses same conventions as Tet4Connectivity
    vertices: [Point3<T>; 4],
}

impl<T> Tetrahedron<T>
where
    T: Scalar,
{
    /// Construct tetrahedron from the given points.
    ///
    /// Ordering is the same as for `Tet4Connectivity`.
    pub fn from_vertices(vertices: [Point3<T>; 4]) -> Self {
        Self { vertices }
    }
}

impl<T> Tetrahedron<T>
where
    T: RealField,
{
    /// Reference tetrahedron.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
        ])
    }
}

impl<T: RealField> BoundedGeometry<T> for Tetrahedron<T> {
    type Dimension = U3;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, U3> {
        AxisAlignedBoundingBox::from_points(&self.vertices).unwrap()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub struct Hexahedron<T>
where
    T: Scalar,
{
    // Ordering uses same conventions as Hex8Connectivity
    vertices: [Point3<T>; 8],
}

impl<T> BoundedGeometry<T> for Hexahedron<T>
where
    T: RealField,
{
    type Dimension = U3;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, U3> {
        AxisAlignedBoundingBox::from_points(&self.vertices).unwrap()
    }
}

impl<T> Hexahedron<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 8]) -> Self {
        Self { vertices }
    }
}

impl<T> Hexahedron<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
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

impl<T> Distance<T, Point3<T>> for Hexahedron<T>
where
    T: RealField,
{
    fn distance(&self, point: &Point3<T>) -> T {
        let signed_dist = self.compute_signed_distance(point).signed_distance;
        T::max(signed_dist, T::zero())
    }
}

impl<T> SignedDistance<T, U3> for Hexahedron<T>
where
    T: RealField,
{
    fn query_signed_distance(&self, point: &OPoint<T, U3>) -> Option<SignedDistanceResult<T, U3>> {
        Some(self.compute_signed_distance(point))
    }
}

impl<'a, T> ConvexPolyhedron<'a, T> for Hexahedron<T>
where
    T: RealField,
{
    type Face = Quad3d<T>;

    fn num_faces(&self) -> usize {
        6
    }

    fn get_face(&self, index: usize) -> Option<Self::Face> {
        let v = &self.vertices;
        let quad = |i, j, k, l| Some(Quad3d::from_vertices([v[i], v[j], v[k], v[l]]));

        // Must choose faces carefully so that they point towards the interior
        match index {
            0 => quad(0, 1, 2, 3),
            1 => quad(4, 5, 1, 0),
            2 => quad(5, 6, 2, 1),
            3 => quad(6, 7, 3, 2),
            4 => quad(0, 3, 7, 4),
            5 => quad(4, 7, 6, 5),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Quad3d<T: Scalar> {
    vertices: [Point3<T>; 4],
}

impl<T: Scalar> Quad3d<T> {
    pub fn from_vertices(vertices: [Point3<T>; 4]) -> Self {
        Self { vertices }
    }
}

impl<'a, T> ConvexPolygon3d<'a, T> for Quad3d<T>
where
    T: Scalar,
{
    fn num_vertices(&self) -> usize {
        4
    }

    fn get_vertex(&self, index: usize) -> Option<Point3<T>> {
        self.vertices.get(index).cloned()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct HalfSpace<T: Scalar> {
    point: Point3<T>,
    normal: Unit<Vector3<T>>,
}

impl<T> HalfSpace<T>
where
    T: RealField,
{
    pub fn contains_point(&self, point: &Point3<T>) -> bool {
        let x = point.coords;
        let x0 = self.point.coords;
        self.normal.dot(&(x - x0)) >= T::zero()
    }

    pub fn from_point_and_normal(point: Point3<T>, normal: Unit<Vector3<T>>) -> Self {
        Self { point, normal }
    }

    pub fn plane(&self) -> Plane3d<T> {
        Plane3d::from_point_and_normal(self.point, self.normal)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Plane3d<T: Scalar> {
    point: Point3<T>,
    normal: Unit<Vector3<T>>,
}

impl<T> Plane3d<T>
where
    T: RealField,
{
    pub fn normal(&self) -> &Unit<Vector3<T>> {
        &self.normal
    }

    pub fn point(&self) -> &Point3<T> {
        &self.point
    }

    pub fn from_point_and_normal(point: Point3<T>, normal: Unit<Vector3<T>>) -> Self {
        Self { point, normal }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LineSegment3d<T: Scalar> {
    end_points: [Point3<T>; 2],
}

impl<T: Scalar> LineSegment3d<T> {
    pub fn from_end_points(end_points: [Point3<T>; 2]) -> Self {
        Self { end_points }
    }
}

impl<T: RealField> LineSegment3d<T> {
    pub fn project_point_parametric(&self, point: &Point3<T>) -> T {
        let a = self.end_points[0].coords;
        let b = self.end_points[1].coords;
        let d = &b - &a;
        let d2 = d.magnitude_squared();
        if d2 == T::zero() {
            // If the endpoints are the same, the segment collapses to a single point,
            // in which case e.g. t == 0 gives the correct solution.
            T::zero()
        } else {
            let x = point.coords;
            let t = (x - &a).dot(&d) / d2;
            t
        }
    }

    pub fn project_point(&self, point: &Point3<T>) -> Point3<T> {
        let t = self.project_point_parametric(point);
        if t <= T::zero() {
            self.end_points[0]
        } else if t >= T::one() {
            self.end_points[1]
        } else {
            self.point_from_parameter(t)
        }
    }

    pub fn point_from_parameter(&self, t: T) -> Point3<T> {
        let a = self.end_points[0];
        let b = self.end_points[1];
        Point3::from(a.coords * (T::one() - t) + &b.coords * t)
    }

    #[allow(non_snake_case)]
    pub fn closest_point_to_plane_parametric(&self, plane: &Plane3d<T>) -> T {
        let n = plane.normal();
        let x0 = plane.point();
        let [a, b] = &self.end_points;
        let d = &b.coords - &a.coords;
        let y = &x0.coords - &a.coords;

        let nTd = n.dot(&d);
        let nTy = n.dot(&y);

        // The parameter t is generally given by the equation
        //  dot(n, d) * t = dot(n, y)
        // but we must be careful, since dot(n, d) can get arbitrarily close to 0,
        // which causes some challenges.
        let t = if nTd.signum() == nTy.signum() {
            // Sign is the same, thus t >= 0
            if nTy.abs() >= nTd.abs() {
                T::one()
            } else {
                nTy / nTd
            }
        } else {
            // t must be negative, directly clamp to zero
            T::zero()
        };

        t
    }

    pub fn closest_point_to_plane(&self, plane: &Plane3d<T>) -> Point3<T> {
        let t = self.closest_point_to_plane_parametric(plane);
        self.point_from_parameter(t)
    }
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

impl<'a, T> ConvexPolyhedron<'a, T> for Tetrahedron<T>
where
    T: RealField,
{
    type Face = Triangle3d<T>;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face(&self, index: usize) -> Option<Self::Face> {
        let v = &self.vertices;
        let tri = |i, j, k| Some(Triangle([v[i], v[j], v[k]]));

        // Must choose faces carefully so that they point towards the interior
        match index {
            0 => tri(0, 1, 2),
            1 => tri(0, 3, 1),
            2 => tri(1, 3, 2),
            3 => tri(0, 2, 3),
            _ => None,
        }
    }
}

impl<T> Distance<T, Point3<T>> for Tetrahedron<T>
where
    T: RealField,
{
    fn distance(&self, point: &OPoint<T, U3>) -> T {
        let triangle = |i, j, k| Triangle([self.vertices[i], self.vertices[j], self.vertices[k]]);

        let tri_faces = [
            // We must carefully choose the ordering of vertices so that the
            // resulting faces have outwards-pointing normals
            triangle(2, 1, 0),
            triangle(1, 2, 3),
            triangle(0, 1, 3),
            triangle(2, 0, 3),
        ];

        let mut point_inside = true;
        let mut min_dist = T::max_value();

        for tri_face in &tri_faces {
            // Remember that the triangles are oriented such that *outwards* is the positive
            // direction, so for the point to be inside of the cell, we need its orientation
            // with respect to each face to be *negative*
            if tri_face.point_orientation(point) == OrientationTestResult::Positive {
                point_inside = false;
            }

            min_dist = T::min(min_dist, tri_face.distance(point));
        }

        if point_inside {
            T::zero()
        } else {
            min_dist
        }
    }
}

impl<T> From<Triangle2d<T>> for ConvexPolygon<T>
where
    T: Scalar,
{
    fn from(triangle: Triangle2d<T>) -> Self {
        // TODO: Use Point2 in Mesh
        ConvexPolygon::from_vertices(triangle.0.iter().map(|v| Point2::from(v.clone())).collect())
    }
}

impl<T> BoundedGeometry<T> for Quad2d<T>
where
    T: RealField,
{
    type Dimension = U2;

    fn bounding_box(&self) -> AxisAlignedBoundingBox2d<T> {
        AxisAlignedBoundingBox2d::from_points(&self.0).expect("Triangle always has > 0 vertices")
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// A quadrilateral consisting of four vertices, assumed to be specified in counter-clockwise
/// winding order.
pub struct Quad2d<T: Scalar>(pub [Point2<T>; 4]);

impl<T> Quad2d<T>
where
    T: RealField,
{
    /// Returns the index of a concave corner of the quadrilateral, if there is any.
    pub fn concave_corner(&self) -> Option<usize> {
        for i in 0..4 {
            let x_next = self.0[(i + 2) % 4];
            let x_curr = self.0[(i + 1) % 4];
            let x_prev = self.0[(i + 1) % 4];

            let a = x_next - x_curr;
            let b = x_prev - x_curr;
            // perp gives "2d cross product", which when negative means that the interior angle
            // is creater than 180 degrees, and so the corner must be concave
            if a.perp(&b) < T::zero() {
                return Some(i + 1);
            }
        }

        None
    }

    /// Splits the quad into two triangles represented by local indices { 0, 1, 2, 3 }
    /// which correspond to the quad's vertices.
    ///
    /// While the quad may be concave, it is assumed that it has no self-intersections and that
    /// all vertices are unique.
    pub fn split_into_triangle_connectivities(&self) -> ([usize; 3], [usize; 3]) {
        if let Some(concave_corner_index) = self.concave_corner() {
            let i = concave_corner_index;
            let triangle1 = [(i + 2) % 4, (i + 3) % 4, (i + 0) % 4];
            let triangle2 = [(i + 2) % 4, (i + 0) % 4, (i + 1) % 4];
            (triangle1, triangle2)
        } else {
            // Split arbitrarily, but in a regular fashion
            let triangle1 = [0, 1, 2];
            let triangle2 = [0, 2, 3];
            (triangle1, triangle2)
        }
    }

    pub fn split_into_triangles(&self) -> (Triangle2d<T>, Triangle2d<T>) {
        let (conn1, conn2) = self.split_into_triangle_connectivities();
        let mut vertices1 = [Point2::origin(); 3];
        let mut vertices2 = [Point2::origin(); 3];

        for (v, idx) in izip!(&mut vertices1, &conn1) {
            *v = self.0[*idx];
        }

        for (v, idx) in izip!(&mut vertices2, &conn2) {
            *v = self.0[*idx];
        }

        let tri1 = Triangle(vertices1);
        let tri2 = Triangle(vertices2);

        (tri1, tri2)
    }

    pub fn area(&self) -> T {
        let (tri1, tri2) = self.split_into_triangles();
        tri1.area() + tri2.area()
    }
}

impl<T> Distance<T, Point2<T>> for Quad2d<T>
where
    T: RealField,
{
    fn distance(&self, point: &Point2<T>) -> T {
        // TODO: Avoid heap allocation
        GeneralPolygon::from_vertices(self.0.to_vec()).distance(point)
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

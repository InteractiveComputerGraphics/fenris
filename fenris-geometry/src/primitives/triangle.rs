use crate::{
    AxisAlignedBoundingBox, BoundedGeometry, ConvexPolygon3d, Distance, LineSegment2d, Orientation,
    OrientationTestResult, SignedDistance, SignedDistanceResult,
};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, OVector, Point2, Point3, Scalar, Vector3, U2, U3};
use nalgebra::{Matrix3, RealField};
use numeric_literals::replace_float_literals;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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

/// A triangle in two dimensions, consisting of three vertices.
///
/// For most purposes, the triangle is assumed to be specified with a counter-clockwise
/// winding order, but it also provides facilities for determining the winding order
/// of an arbitrarily constructed triangle.
pub type Triangle2d<T> = Triangle<T, U2>;
pub type Triangle3d<T> = Triangle<T, U3>;

impl<T, D> Copy for Triangle<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    OPoint<T, D>: Copy,
{
}

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
            let segment = LineSegment2d::from_end_points(*a, *b);
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
        self.closest_point(point).distance
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

impl<T: RealField> Triangle3d<T> {
    /// Compute the solid angle to the given point.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn compute_solid_angle(&self, p: &Point3<T>) -> T {
        // Based on equation (6) in Jacobson et al.,
        // "Robust Inside-Outside Segmentation using Generalized Winding Numbers"
        let [a, b, c] = self.0.clone().map(|v_i| v_i - p);
        let abc_matrix = Matrix3::from_columns(&[a.clone(), b.clone(), c.clone()]);

        let anorm = a.norm();
        let bnorm = b.norm();
        let cnorm = c.norm();

        let denominator = anorm * bnorm * cnorm + a.dot(&b) * cnorm + b.dot(&c) * anorm + c.dot(&a) * bnorm;
        let tan_omega_half = abc_matrix.determinant() / denominator;
        2.0 * tan_omega_half.atan()
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn compute_winding_number_for_triangles_3d<T, I>(triangles: I, point: &Point3<T>) -> T
where
    T: RealField,
    I: IntoIterator<Item = Triangle3d<T>>,
{
    let angle_sum = triangles
        .into_iter()
        .map(|triangle| triangle.compute_solid_angle(point))
        .reduce(|acc, angle| acc + angle)
        .unwrap_or(T::zero());
    angle_sum / (4.0 * T::pi())
}

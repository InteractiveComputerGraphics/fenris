use crate::{Line2d, Plane3d};
use nalgebra::{Point2, Point3, RealField, Scalar, Unit, Vector2, Vector3};

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

#[derive(Debug, Clone)]
pub struct HalfPlane<T>
where
    T: Scalar,
{
    point: Point2<T>,
    normal: Unit<Vector2<T>>,
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

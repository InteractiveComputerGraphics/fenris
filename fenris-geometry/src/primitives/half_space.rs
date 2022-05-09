use crate::{Line2d, Plane};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, OVector, RealField, Scalar, Unit, Vector2, U2, U3};

/// A $D$-dimensional half space.
///
/// A half space is defined by a point $\vec x$ and a normalized unit vector (normal) $\vec n$.
/// The points belonging to the half space is then given by the points $\vec y$ that satisfy
///
/// <div>
/// $$
///   (\vec y - \vec x) \cdot \vec n \leq 0.
/// $$
/// </div>
#[derive(Debug, Clone, PartialEq)]
pub struct HalfSpace<T: Scalar, D: DimName = U3>
where
    DefaultAllocator: Allocator<T, D>,
{
    point: OPoint<T, D>,
    normal: Unit<OVector<T, D>>,
}

pub type HalfPlane<T> = HalfSpace<T, U2>;

impl<T, D> HalfSpace<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn signed_distance_to_point(&self, point: &OPoint<T, D>) -> T {
        let d = point - &self.point;
        self.normal.dot(&d)
    }

    pub fn contains_point(&self, point: &OPoint<T, D>) -> bool {
        self.signed_distance_to_point(point) <= T::zero()
    }

    pub fn from_point_and_normal(point: OPoint<T, D>, normal: Unit<OVector<T, D>>) -> Self {
        Self { point, normal }
    }

    pub fn point(&self) -> &OPoint<T, D> {
        &self.point
    }

    /// Returns the outwards-facing normal vector for the plane.
    ///
    /// This vector is normalized.
    pub fn normal(&self) -> &OVector<T, D> {
        &self.normal
    }

    pub fn complement(&self) -> Self {
        Self::from_point_and_normal(self.point.clone(), -self.normal.clone())
    }
}

impl<T> HalfSpace<T>
where
    T: RealField,
{
    pub fn plane(&self) -> Plane<T> {
        Plane::from_point_and_normal(self.point, self.normal)
    }

    pub fn from_plane(plane: &Plane<T>) -> Self {
        Self::from_point_and_normal(plane.point().clone(), plane.normal().clone())
    }
}

impl<T> HalfPlane<T>
where
    T: RealField,
{
    /// Returns a line representing the surface of the half plane
    pub fn surface(&self) -> Line2d<T> {
        let tangent = Vector2::new(self.normal.y, -self.normal.x);
        Line2d::from_point_and_dir(self.point.clone(), tangent)
    }
}

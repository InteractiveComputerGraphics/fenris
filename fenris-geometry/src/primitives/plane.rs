use fenris_traits::Real;
use nalgebra::{Point3, Scalar, Unit, UnitVector3, Vector3};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Plane<T: Scalar> {
    point: Point3<T>,
    normal: Unit<Vector3<T>>,
}

impl<T> Plane<T>
where
    T: Real,
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

    pub fn flipped(&self) -> Self {
        Self {
            point: self.point.clone(),
            normal: Unit::new_unchecked(-self.normal.into_inner()),
        }
    }

    pub fn compute_tangent_vectors(&self) -> [UnitVector3<T>; 2] {
        crate::util::compute_orthonormal_vectors_3d(self.normal())
    }
}

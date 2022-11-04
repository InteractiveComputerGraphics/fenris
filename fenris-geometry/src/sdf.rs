use fenris_traits::Real;
use nalgebra::{OPoint, Point2, Scalar, Vector2, U2};

use crate::{AxisAlignedBoundingBox2d, BoundedGeometry};
use numeric_literals::replace_float_literals;

pub trait SignedDistanceFunction2d<T>
where
    T: Scalar,
{
    fn eval(&self, x: &Point2<T>) -> T;
    fn gradient(&self, x: &Point2<T>) -> Option<Vector2<T>>;

    fn union<Other>(self, other: Other) -> SdfUnion<Self, Other>
    where
        Self: Sized,
        Other: Sized + SignedDistanceFunction2d<T>,
    {
        SdfUnion {
            left: self,
            right: other,
        }
    }
}

pub trait BoundedSdf<T>: SignedDistanceFunction2d<T> + BoundedGeometry<T, Dimension = U2>
where
    T: Scalar,
{
}

impl<X, T> BoundedSdf<T> for X
where
    T: Scalar,
    X: SignedDistanceFunction2d<T> + BoundedGeometry<T, Dimension = U2>,
{
}

#[derive(Copy, Clone, Debug)]
pub struct SdfCircle<T>
where
    T: Scalar,
{
    pub radius: T,
    pub center: Vector2<T>,
}

#[derive(Copy, Clone, Debug)]
pub struct SdfUnion<Left, Right> {
    pub left: Left,
    pub right: Right,
}

#[derive(Copy, Clone, Debug)]
pub struct SdfAxisAlignedBox<T>
where
    T: Scalar,
{
    pub aabb: AxisAlignedBoundingBox2d<T>,
}

impl<T> BoundedGeometry<T> for SdfCircle<T>
where
    T: Real,
{
    type Dimension = U2;

    fn bounding_box(&self) -> AxisAlignedBoundingBox2d<T> {
        let eps = self.radius * T::from_f64(0.01).unwrap();
        AxisAlignedBoundingBox2d::new(
            OPoint::from(self.center - Vector2::repeat(T::one()) * (self.radius + eps)),
            OPoint::from(self.center + Vector2::repeat(T::one()) * (self.radius + eps)),
        )
    }
}

impl<T> SignedDistanceFunction2d<T> for SdfCircle<T>
where
    T: Real,
{
    fn eval(&self, x: &Point2<T>) -> T {
        let y = x - self.center;
        y.coords.norm() - self.radius
    }

    fn gradient(&self, x: &Point2<T>) -> Option<Vector2<T>> {
        let y = x - self.center;
        let y_norm = y.coords.norm();

        if y_norm == T::zero() {
            None
        } else {
            Some(y.coords / y_norm)
        }
    }
}

impl<T, Left, Right> BoundedGeometry<T> for SdfUnion<Left, Right>
where
    T: Real,
    Left: BoundedGeometry<T, Dimension = U2>,
    Right: BoundedGeometry<T, Dimension = U2>,
{
    type Dimension = U2;

    fn bounding_box(&self) -> AxisAlignedBoundingBox2d<T> {
        self.left.bounding_box().enclose(&self.right.bounding_box())
    }
}

impl<T, Left, Right> SignedDistanceFunction2d<T> for SdfUnion<Left, Right>
where
    T: Real,
    Left: SignedDistanceFunction2d<T>,
    Right: SignedDistanceFunction2d<T>,
{
    fn eval(&self, x: &Point2<T>) -> T {
        self.left.eval(x).min(self.right.eval(x))
    }

    fn gradient(&self, x: &Point2<T>) -> Option<Vector2<T>> {
        // TODO: Is this actually correct? It might give funky results if exactly
        // at points where either SDF is non-differentiable
        if self.left.eval(x) < self.right.eval(x) {
            self.left.gradient(x)
        } else {
            self.right.gradient(x)
        }
    }
}

impl<T> BoundedGeometry<T> for SdfAxisAlignedBox<T>
where
    T: Real,
{
    type Dimension = U2;

    fn bounding_box(&self) -> AxisAlignedBoundingBox2d<T> {
        self.aabb
    }
}

impl<T> SignedDistanceFunction2d<T> for SdfAxisAlignedBox<T>
where
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn eval(&self, x: &Point2<T>) -> T {
        let b = self.aabb.extents() / 2.0;
        let p = x - self.aabb.center();
        let d = p.abs() - b;

        // TODO: Use d.max() when fixed. See https://github.com/rustsim/nalgebra/issues/620
        d.sup(&Vector2::zeros()).norm() + T::min(T::zero(), d[d.imax()])
    }

    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradient(&self, x: &Point2<T>) -> Option<Vector2<T>> {
        // TODO: Replace finite differences with "proper" gradient
        // Note: arbitrary "step"/resolution h
        let h = 1e-4;
        let mut gradient = Vector2::zeros();
        for i in 0..2 {
            let mut dx = Vector2::zeros();
            dx[i] = h;
            gradient[i] = (self.eval(&(x + dx)) - self.eval(&(x - dx))) / (2.0 * h)
        }
        gradient.normalize_mut();
        Some(gradient)
    }
}

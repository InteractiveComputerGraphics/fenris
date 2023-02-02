use fenris_traits::Real;
use nalgebra::{matrix, Point2};

pub fn orient2d_inexact<T: Real>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> T {
    matrix![a.x, a.y, T::one();
            b.x, b.y, T::one();
            c.x, c.y, T::one()]
    .determinant()
}

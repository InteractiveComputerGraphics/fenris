use std::ops::{Add, AddAssign, Deref, Mul};

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Point, Scalar, U2, U3};
use num::Zero;

/// Errors returned by quadrature methods.
///
/// TODO: How to prevent collapse?
pub use fenris_quadrature::Error as QuadratureError;

use crate::nalgebra::{convert, Point2, Point3, RealField};

pub mod tensor;
pub mod total_order;

pub type QuadraturePair<T, D> = (Vec<T>, Vec<Point<T, D>>);
pub type QuadraturePair2d<T> = QuadraturePair<T, U2>;
pub type QuadraturePair3d<T> = QuadraturePair<T, U3>;

pub trait Quadrature<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn weights(&self) -> &[T];
    fn points(&self) -> &[Point<T, D>];

    /// Approximates the integral of the given function using this quadrature rule.
    fn integrate<U, Function>(&self, f: Function) -> U
    where
        Function: Fn(&Point<T, D>) -> U,
        U: Zero + Mul<T, Output = U> + Add<T, Output = U> + AddAssign<U>,
    {
        let mut integral = U::zero();
        for (w, p) in self.weights().iter().zip(self.points()) {
            integral += f(p) * w.clone();
        }
        integral
    }
}

/// Helper trait for 2D quadratures.
pub trait Quadrature2d<T>: Quadrature<T, U2>
where
    T: Scalar,
{
}

impl<T, X> Quadrature2d<T> for X
where
    T: Scalar,
    X: Quadrature<T, U2>,
{
}

impl<T, D, A, B> Quadrature<T, D> for (A, B)
where
    T: Scalar,
    D: DimName,
    A: Deref<Target = [T]>,
    B: Deref<Target = [Point<T, D>]>,
    DefaultAllocator: Allocator<T, D>,
{
    fn weights(&self) -> &[T] {
        self.0.deref()
    }

    fn points(&self) -> &[Point<T, D>] {
        self.1.deref()
    }
}

impl<T, D, X> Quadrature<T, D> for &X
where
    T: Scalar,
    D: DimName,
    X: Quadrature<T, D>,
    DefaultAllocator: Allocator<T, D>,
{
    fn weights(&self) -> &[T] {
        X::weights(self)
    }

    fn points(&self) -> &[Point<T, D>] {
        X::points(self)
    }
}

fn convert_quadrature_rule_from_2d_f64<T>(quadrature: fenris_quadrature::Rule<2>) -> QuadraturePair2d<T>
where
    T: RealField,
{
    let (weights, points) = quadrature;
    let weights = weights.into_iter().map(convert).collect();
    let points = points.into_iter().map(Point2::from).map(convert).collect();
    (weights, points)
}

fn convert_quadrature_rule_from_3d_f64<T>(quadrature: fenris_quadrature::Rule<3>) -> QuadraturePair3d<T>
where
    T: RealField,
{
    let (weights, points) = quadrature;
    let weights = weights.into_iter().map(convert).collect();
    let points = points.into_iter().map(Point3::from).map(convert).collect();
    (weights, points)
}

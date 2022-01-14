use std::ops::{Add, AddAssign, Deref, Mul};

use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Point1, Scalar, U2, U3};
use num::Zero;

pub use canonical::*;
/// Errors returned by quadrature methods.
///
/// TODO: How to prevent collapse?
pub use fenris_quadrature::Error as QuadratureError;

use crate::nalgebra::{convert, Point2, Point3, RealField, U1};

pub mod subdivide;
pub mod tensor;
pub mod total_order;
pub mod univariate;

mod canonical;

pub type QuadraturePair<T, D> = (Vec<T>, Vec<OPoint<T, D>>);
pub type QuadraturePair1d<T> = QuadraturePair<T, U1>;
pub type QuadraturePair2d<T> = QuadraturePair<T, U2>;
pub type QuadraturePair3d<T> = QuadraturePair<T, U3>;

/// A quadrature rule consisting of weights, points and data.
pub trait Quadrature<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Data;

    fn weights(&self) -> &[T];
    fn points(&self) -> &[OPoint<T, D>];
    fn data(&self) -> &[Self::Data];

    /// Approximates the integral of the given function using this quadrature rule.
    fn integrate<U, Function>(&self, f: Function) -> U
    where
        Function: Fn(&OPoint<T, D>) -> U,
        U: Zero + Mul<T, Output = U> + Add<T, Output = U> + AddAssign<U>,
    {
        let mut integral = U::zero();
        for (w, p) in self.weights().iter().zip(self.points()) {
            integral += f(p) * w.clone();
        }
        integral
    }

    fn to_parts(&self) -> QuadratureParts<&[T], &[OPoint<T, D>], &[Self::Data]> {
        QuadratureParts {
            weights: self.weights(),
            points: self.points(),
            data: self.data(),
        }
    }
}

/// Trait alias for 1D quadrature rules.
pub trait Quadrature1d<T>: Quadrature<T, U1>
where
    T: Scalar,
{
}

/// Trait alias for 2D quadrature rules.
pub trait Quadrature2d<T>: Quadrature<T, U2>
where
    T: Scalar,
{
}

/// Trait alias for 3D quadrature rules.
pub trait Quadrature3d<T>: Quadrature<T, U3>
where
    T: Scalar,
{
}

impl<T, X> Quadrature1d<T> for X
where
    T: Scalar,
    X: Quadrature<T, U1>,
{
}

impl<T, X> Quadrature2d<T> for X
where
    T: Scalar,
    X: Quadrature<T, U2>,
{
}

impl<T, X> Quadrature3d<T> for X
where
    T: Scalar,
    X: Quadrature<T, U3>,
{
}

impl<T, D, A, B> Quadrature<T, D> for (A, B)
where
    T: Scalar,
    D: DimName,
    A: AsRef<[T]>,
    B: AsRef<[OPoint<T, D>]>,
    DefaultAllocator: Allocator<T, D>,
{
    type Data = ();

    fn weights(&self) -> &[T] {
        self.0.as_ref()
    }

    fn points(&self) -> &[OPoint<T, D>] {
        self.1.as_ref()
    }

    fn data(&self) -> &[()] {
        // This may look absurd, but since we're just returning a slice to a zero-sized type (the unit type),
        // the (global) allocator never allocates anything and most likely the whole thing will get completely
        // optimized out
        vec![(); self.weights().len()].leak()
    }
}

impl<T, D, X> Quadrature<T, D> for &X
where
    T: Scalar,
    D: DimName,
    X: Quadrature<T, D>,
    DefaultAllocator: Allocator<T, D>,
{
    type Data = X::Data;

    fn weights(&self) -> &[T] {
        X::weights(self)
    }

    fn points(&self) -> &[OPoint<T, D>] {
        X::points(self)
    }

    fn data(&self) -> &[Self::Data] {
        X::data(self)
    }
}

/// Marker to indicate that a quadrature rule stored in [`QuadratureParts`] has no associated data.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct NoData;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct QuadratureParts<WeightsArray, PointsArray, DataArray> {
    pub weights: WeightsArray,
    pub points: PointsArray,
    pub data: DataArray,
}

impl<WeightsArray, PointsArray, DataArray> QuadratureParts<WeightsArray, PointsArray, DataArray> {
    pub fn with_data<DataArray2>(self, data: DataArray2) -> QuadratureParts<WeightsArray, PointsArray, DataArray2> {
        QuadratureParts {
            weights: self.weights,
            points: self.points,
            data,
        }
    }
}

impl<T, D, WeightsArray, PointsArray, DataArray, Data> Quadrature<T, D>
    for QuadratureParts<WeightsArray, PointsArray, DataArray>
where
    T: Scalar,
    D: DimName,
    WeightsArray: AsRef<[T]>,
    PointsArray: AsRef<[OPoint<T, D>]>,
    DataArray: Deref<Target = [Data]>,
    DefaultAllocator: Allocator<T, D>,
{
    type Data = Data;

    fn weights(&self) -> &[T] {
        self.weights.as_ref()
    }

    fn points(&self) -> &[OPoint<T, D>] {
        self.points.as_ref()
    }

    fn data(&self) -> &[Self::Data] {
        self.data.deref()
    }
}

impl<T, D, WeightsArray, PointsArray> Quadrature<T, D> for QuadratureParts<WeightsArray, PointsArray, NoData>
where
    T: Scalar,
    D: DimName,
    WeightsArray: AsRef<[T]>,
    PointsArray: AsRef<[OPoint<T, D>]>,
    DefaultAllocator: Allocator<T, D>,
{
    type Data = ();

    fn weights(&self) -> &[T] {
        self.weights.as_ref()
    }

    fn points(&self) -> &[OPoint<T, D>] {
        self.points.as_ref()
    }

    fn data(&self) -> &[()] {
        // This is a "sound" way of constructing a unit type slice of arbitrary size.
        // Since it's zero-sized, it won't actually allocate any memory and the leak is elided
        vec![(); self.weights().len()].leak()
    }
}

fn convert_quadrature_rule_from_1d_f64<T>(quadrature: fenris_quadrature::Rule<1>) -> QuadraturePair1d<T>
where
    T: RealField,
{
    let (weights, points) = quadrature;
    let weights = weights.into_iter().map(convert).collect();
    let points = points.into_iter().map(Point1::from).map(convert).collect();
    (weights, points)
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

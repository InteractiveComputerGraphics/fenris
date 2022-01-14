//! Quadrature rules constructed by subdividing the reference domain.
use crate::quadrature::{Quadrature1d, QuadraturePair1d};
use nalgebra::{Point1, RealField};
use numeric_literals::replace_float_literals;

/// Construct a univariate quadrature rule by subdivision.
///
/// This function constructs a quadrature rule for the univariate reference domain by subdividing the domain
/// into the prescribed number of pieces and applying the given quadrature rule in each subdivision.
/// The resulting weights and points are transformed back to the reference domain, thereby constructing
/// an aggregate quadrature rule from the individual pieces.
pub fn subdivide_univariate<T>(quadrature: impl Quadrature1d<T>, subdivision_pieces: usize) -> QuadraturePair1d<T>
where
    T: RealField,
{
    subdivide_univariate_(quadrature.weights(), quadrature.points(), subdivision_pieces)
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn subdivide_univariate_<T>(
    reference_weights: &[T],
    reference_points: &[Point1<T>],
    subdivision_pieces: usize,
) -> QuadraturePair1d<T>
where
    T: RealField,
{
    let mut points = Vec::new();
    let mut weights = Vec::new();

    let pieces_as_scalar = T::from_usize(subdivision_pieces)
        // This should never panic, because in the worst case, it gets truncated. However, it is
        // possible that for a custom/niche "real" type, the method returns `None`.
        .expect("Internal error: Failed to convert usize to scalar type");

    let subdivision_size = 2.0 / pieces_as_scalar;
    for i in 0..subdivision_pieces {
        // This should never panic, because we would likely have already panicked in the previous integer -> scalar
        // conversion.
        let i = T::from_usize(i).expect("Internal error: Failed to convert usize to scalar type.");
        let a = i * subdivision_size - 1.0;
        let b = a + subdivision_size;
        let jacobian = (b - a) / 2.0;

        for (ref_weight, ref_point) in reference_weights.iter().zip(reference_points) {
            let weight = ref_weight.clone() * jacobian;
            let point = ((b - a) * ref_point[0] + (b + a)) / 2.0;
            weights.push(weight);
            points.push(Point1::new(point));
        }
    }

    (weights, points)
}

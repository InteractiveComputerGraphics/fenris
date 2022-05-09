use nalgebra::{RealField, UnitVector3, Vector3};

pub fn compute_orthonormal_vectors_3d<T: RealField>(vector: &UnitVector3<T>) -> [UnitVector3<T>; 2] {
    // Ported from
    // https://github.com/dimforge/parry/blob/ac8dcf0197066cd2413a20e4420961b4694996c0/src/utils/wops.rs#L120-L138
    // originally based on the Pixar paper "Building an Orthonormal Basis, Revisited",
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    let v = vector;
    let sign = T::copysign(T::one(), v.z);
    let a = -T::one() / (sign + v.z);
    let b = v.x * v.y * a;

    [
        Vector3::new(T::one() + sign * v.x * v.x * a, sign * b, -sign * v.x),
        Vector3::new(b, sign + v.y * v.y * a, -v.y),
    ]
    .map(UnitVector3::new_unchecked)
}

/// Compares two arrays for *shift-invariant* equality with the given comparator function.
///
/// If `X` and `Y` are the two arrays, then the two arrays are shift-invariant equal if X can be shifted/rotated
/// by some constant `n` such that `Shift(X) == Y`.
pub fn slices_are_equal_shift_invariant<T, C: Fn(&T, &T) -> bool>(x: &[T], y: &[T], comparator: C) -> bool {
    let n = x.len();
    if y.len() != n {
        return false;
    } else if n == 0 {
        // Empty arrays are always equal
        return true;
    }

    'outer_loop: for i_start in 0..n {
        for (j, y_j) in y.iter().enumerate() {
            let x_j_shifted = &x[(j + i_start) % n];
            if !comparator(x_j_shifted, y_j) {
                continue 'outer_loop;
            }
        }
        return true;
    }

    return false;
}

// Base macro for line segment assertions
#[doc(hidden)]
#[macro_export]
macro_rules! assert_line_segments_approx_equal_base {
    ($msg_handler:expr, $segment1:expr, $segment2:expr, abstol = $tol:expr) => {{
        use fenris_geometry::util::slices_are_equal_shift_invariant;
        use matrixcompare::comparators::AbsoluteElementwiseComparator;
        use matrixcompare::compare_matrices;
        use nalgebra::Point2;

        let tol = $tol.clone();
        // Type check: Makes for an easier error message than failing specific methods
        let (segment1, segment2): (&LineSegment2d<_>, &LineSegment2d<_>) = (&$segment1, &$segment2);
        let vertices1 = [segment1.start().clone(), segment1.end().clone()];
        let vertices2 = [segment2.start().clone(), segment2.end().clone()];
        let comparator = AbsoluteElementwiseComparator { tol };
        let comparator = |a: &Point2<f64>, b: &Point2<f64>| compare_matrices(&a.coords, &b.coords, &comparator).is_ok();
        let vertices_are_shift_invariant_equal = slices_are_equal_shift_invariant(&vertices1, &vertices2, comparator);
        if !vertices_are_shift_invariant_equal {
            let msg = format!(
                "Line segments are not (approximately) equal to absolute tolerance {tol}.
Segment1: {:?}
Segment2: {:?}",
                segment1, segment2
            );

            return $msg_handler(msg);
        }
    }};
}

#[macro_export]
macro_rules! assert_line_segments_approx_equal {
    ($segment1:expr, $segment2:expr, abstol = $tol:expr) => {{
        let msg_handler = |msg| panic!("{}", msg);
        $crate::assert_line_segments_approx_equal_base!(msg_handler, $segment1, $segment2, abstol = $tol);
    }};
}

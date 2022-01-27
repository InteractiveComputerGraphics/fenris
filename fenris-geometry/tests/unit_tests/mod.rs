mod aabb;
mod geometry;
mod polygon;
mod polymesh;
mod polytope;

use fenris_geometry::LineSegment2d;
use nalgebra::{point, Point2};
use proptest::collection::{hash_set, vec};
use proptest::prelude::*;
use util::assert_panics;

// Base macro for line segment assertions
#[doc(hidden)]
macro_rules! assert_line_segments_approx_equal_base {
    ($msg_handler:expr, $segment1:expr, $segment2:expr, abstol = $tol:expr) => {{
        use matrixcompare::comparators::AbsoluteElementwiseComparator;
        use matrixcompare::compare_matrices;
        use $crate::unit_tests::slices_are_equal_shift_invariant;

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

macro_rules! assert_line_segments_approx_equal {
    ($segment1:expr, $segment2:expr, abstol = $tol:expr) => {{
        let msg_handler = |msg| panic!("{}", msg);
        $crate::unit_tests::assert_line_segments_approx_equal_base!(msg_handler, $segment1, $segment2, abstol = $tol);
    }};
}

macro_rules! prop_assert_line_segments_approx_equal {
    ($segment1:expr, $segment2:expr, abstol = $tol:expr) => {{
        let msg_handler = |msg| {
            // Add filename and line numbers to message (since we don't panic, it's useful
            // to have this information in the output).
            let amended_message = format!("Proptest assertion failure at {}:{}. {}", file!(), line!(), msg);
            return ::core::result::Result::Err(::proptest::test_runner::TestCaseError::fail(amended_message));
        };
        $crate::unit_tests::assert_line_segments_approx_equal_base!(msg_handler, $segment1, $segment2, abstol = $tol);
    }};
}

pub(crate) use assert_line_segments_approx_equal;
pub(crate) use assert_line_segments_approx_equal_base;
pub(crate) use prop_assert_line_segments_approx_equal;

#[test]
fn test_line_segment_assert() {
    let tol = 1e-14;
    let a = point![2.0, 3.0];
    let b = point![3.0, 4.0];
    let c = point![1.0, 3.0];
    let segment1 = LineSegment2d::new(a, b);
    let segment2 = LineSegment2d::new(b, a);
    let segment3 = LineSegment2d::new(a, c);

    assert_line_segments_approx_equal!(segment1, segment2, abstol = tol);
    assert_line_segments_approx_equal!(segment2, segment1, abstol = tol);

    assert_panics! { assert_line_segments_approx_equal!(segment1, segment3, abstol=tol) };
    assert_panics! { assert_line_segments_approx_equal!(segment2, segment3, abstol=tol) };
}

/// Compares two arrays for *shift-invariant* equality with the given comparator function.
///
/// If `X` and `Y` are the two arrays, then the two arrays are shift-invariant equal if X can be shifted/rotated
/// by some constant `n` such that `Shift(X) == Y`.
fn slices_are_equal_shift_invariant<T, C: Fn(&T, &T) -> bool>(x: &[T], y: &[T], comparator: C) -> bool {
    let n = x.len();
    if y.len() != n {
        return false;
    } else if n == 0 {
        // Empty arrays are always equal
        return true;
    }

    for i_start in 0..n {
        let mut all_equal = true;

        for (j, y_j) in y.iter().enumerate() {
            let x_j_shifted = &x[(j + i_start) % n];
            if !comparator(x_j_shifted, y_j) {
                all_equal = false;
                break;
            }
        }
        if all_equal {
            return true;
        }
    }

    return false;
}

#[test]
fn slices_are_equal_shift_invariant_basic_examples() {
    let cmp = |a: &u32, b: &u32| a == b;

    // Examples that evaluate to true (no duplicates)
    assert!(slices_are_equal_shift_invariant(&[], &[], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1], &[1], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 2], &[1, 2], &cmp));
    assert!(slices_are_equal_shift_invariant(&[2, 1], &[1, 2], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 2], &[2, 1], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 2, 3], &[1, 2, 3], &cmp));
    assert!(slices_are_equal_shift_invariant(&[3, 1, 2], &[1, 2, 3], &cmp));
    assert!(slices_are_equal_shift_invariant(&[2, 3, 1], &[1, 2, 3], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 2, 3], &[3, 1, 2], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 2, 3], &[2, 3, 1], &cmp));

    // Examples that evaluate to false (no duplicates)
    assert!(!slices_are_equal_shift_invariant(&[1, 2, 3], &[2, 1, 3], &cmp));
    assert!(!slices_are_equal_shift_invariant(&[2, 1, 3], &[1, 2, 3], &cmp));
    assert!(!slices_are_equal_shift_invariant(&[2, 1, 3], &[2, 3, 1], &cmp));
    assert!(!slices_are_equal_shift_invariant(&[1, 2, 3, 4], &[1, 3, 2, 4], &cmp));
    assert!(!slices_are_equal_shift_invariant(&[1, 2, 3, 4], &[3, 1, 4, 2], &cmp));

    // Examples that evaluate to true (with duplicates)
    assert!(slices_are_equal_shift_invariant(&[], &[], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 1], &[1, 1], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 1, 2], &[1, 1, 2], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 1, 2], &[2, 1, 1], &cmp));
    assert!(slices_are_equal_shift_invariant(&[1, 1, 2], &[1, 2, 1], &cmp));

    // Examples that evaluate to false (with duplicates)
    assert!(!slices_are_equal_shift_invariant(&[1, 1, 2, 3], &[2, 1, 1, 3], &cmp));
    assert!(!slices_are_equal_shift_invariant(&[1, 2, 1, 3], &[2, 1, 1, 3], &cmp));
}

proptest! {
    #[test]
    fn slices_are_equal_shift_invariant_for_rotated_arrays(array in vec(0 .. 10usize, 0 .. 10), n in 0 .. 10usize) {
        let n = n % (array.len() + 1);
        let mut rotated = array.clone();
        rotated.rotate_left(n);
        prop_assert!(slices_are_equal_shift_invariant(&array, &rotated, |a, b| a == b));
    }

    #[test]
    fn slices_are_not_shift_invariant_equal_for_swapped_entries(
        // Use a hash set to ensure unique entries, otherwise duplicates may cause the swap to still compare
        // shift-invariant equal
        (i, j, array) in hash_set(0 .. 10usize, 3 .. 10)
            .prop_flat_map(|v| ((0 .. v.len()), (0 .. v.len()), Just(v)))
            .prop_map(|(i, j, v)| (i, j, v.into_iter().collect::<Vec<_>>()))
            .prop_filter("Require v[i] != v[j] to ensure the swap is not a no-op", |(i, j, v)| v[*i] != v[*j])
        )
    {
        let mut modified = array.clone();
        modified.swap(i, j);
        prop_assert!(!slices_are_equal_shift_invariant(&array, &modified, |a, b| a == b));
    }
}

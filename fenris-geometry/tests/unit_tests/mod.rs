mod aabb;
mod geometry;
mod polygon;
mod polymesh;
mod polytope;
mod predicates;
mod primitives;
mod util;

use ::util::assert_panics;
use fenris_geometry::{assert_line_segments_approx_equal, LineSegment2d};
use nalgebra::point;
use proptest::collection::{hash_set, vec};
use proptest::prelude::*;

use fenris_geometry::util::slices_are_equal_shift_invariant;

#[test]
fn test_line_segment_assert() {
    let tol = 1e-14;
    let a = point![2.0, 3.0];
    let b = point![3.0, 4.0];
    let c = point![1.0, 3.0];
    let segment1 = LineSegment2d::from_end_points(a, b);
    let segment2 = LineSegment2d::from_end_points(b, a);
    let segment3 = LineSegment2d::from_end_points(a, c);

    assert_line_segments_approx_equal!(segment1, segment2, abstol = tol);
    assert_line_segments_approx_equal!(segment2, segment1, abstol = tol);

    assert_panics! { assert_line_segments_approx_equal!(segment1, segment3, abstol=tol) };
    assert_panics! { assert_line_segments_approx_equal!(segment2, segment3, abstol=tol) };
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

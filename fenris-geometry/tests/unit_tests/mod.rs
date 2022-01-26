mod aabb;
mod geometry;
mod polygon;
mod polymesh;
mod polytope;

use proptest::prelude::*;
use proptest::collection::{hash_set, vec};

/// Compares two arrays for *shift-invariant* equality with the given comparator function.
///
/// If `X` and `Y` are the two arrays, then the two arrays are shift-invariant equal if X can be shifted/rotated
/// by some constant `n` such that `Shift(X) == Y`.
fn slices_are_equal_shift_invariant<T, C: Fn(&T, &T) -> bool>(
    x: &[T],
    y: &[T],
    comparator: C)
-> bool {
    let n = x.len();
    if y.len() != n {
        return false;
    } else if n == 0 {
        // Empty arrays are always equal
        return true;
    }

    for i_start in 0 .. n {
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
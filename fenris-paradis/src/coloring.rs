use std::cmp::max;
use crate::DisjointSubsets;
use fenris_nested_vec::NestedVec;
use std::collections::BTreeSet;
use std::f32::consts::PI;
use std::mem;

#[derive(Debug)]
struct Color {
    subsets: NestedVec<usize>,
    labels: Vec<usize>,
    indices: BTreeSet<usize>,
}

impl Color {
    fn new_with_subset(subset: &[usize], label: usize) -> Self {
        let mut subsets = NestedVec::new();
        subsets.push(subset);
        Self {
            subsets,
            labels: vec![label],
            indices: subset.iter().copied().collect(),
        }
    }

    fn try_add_subset(&mut self, subset: &[usize], label: usize, local_workspace_set: &mut BTreeSet<usize>) -> bool {
        local_workspace_set.clear();
        for idx in subset {
            local_workspace_set.insert(*idx);
        }

        if self.indices.is_disjoint(&local_workspace_set) {
            self.subsets.push(subset);
            self.labels.push(label);

            for &idx in local_workspace_set.iter() {
                self.indices.insert(idx);
            }
            true
        } else {
            false
        }
    }

    fn max_index(&self) -> Option<usize> {
        // Use the fact that the last element in a BTreeSet is the largest value in the set
        self.indices.iter().copied().last()
    }
}

fn sequential_greedy_coloring_old(subsets: &NestedVec<usize>) -> Vec<DisjointSubsets> {
    let mut colors = Vec::<Color>::new();
    let mut workspace_set = BTreeSet::new();

    'subset_loop: for (label, subset) in subsets.iter().enumerate() {
        for color in &mut colors {
            if color.try_add_subset(subset, label, &mut workspace_set) {
                continue 'subset_loop;
            }
        }

        // We did not succeed in adding the subset to an existing color,
        // so create a new one instead
        colors.push(Color::new_with_subset(subset, label));
    }

    colors
        .into_iter()
        .map(|color| {
            let max_index = color.max_index();
            // Subsets must be disjoint by construction, so skip checks
            unsafe { DisjointSubsets::from_disjoint_subsets_unchecked(color.subsets, color.labels, max_index) }
        })
        .collect()
}

pub fn sequential_greedy_coloring(subsets: &NestedVec<usize>) -> Vec<DisjointSubsets> {
    let mut colors = Vec::new();
    let mut postponed_subset_indices = Vec::new();
    let mut current_subset_indices: Vec<_> = (0 .. subsets.len()).collect();

    // Keep a table of the index of the last color to visit any given node.
    // Since we don't know how many nodes we have, we dynamically resize the table
    // as we run into new indices that are out of bounds.
    let mut last_visited_color = vec![-1i32; 0];

    let mut color_idx = 0i32;
    while !current_subset_indices.is_empty() {
        let mut color_subsets = NestedVec::new();
        let mut color_subset_indices = Vec::new();
        let mut max_node_idx = 0usize;
        for &subset_idx in &current_subset_indices {
            let subset = subsets.get(subset_idx).unwrap();
            let is_blocked = subset.iter().any(|node_idx| last_visited_color
                .get(*node_idx)
                .map(|&idx_of_last_visitor| idx_of_last_visitor == color_idx)
                .unwrap_or(false));
            if is_blocked {
                postponed_subset_indices.push(subset_idx);
            } else {
                for node_idx in subset {
                    max_node_idx = max(*node_idx, max_node_idx);
                    // Update table of visitors
                    if let Some(current_visitor) = last_visited_color.get_mut(*node_idx) {
                        *current_visitor = color_idx;
                    } else {
                        // Try to amortize resizes by creating a larger table than we need right now
                        // (otherwise we might perform many small resizes in succession)
                        last_visited_color.resize(2 * node_idx + 1, -1);
                        last_visited_color[*node_idx] = color_idx;
                    }
                }
                color_subsets.push(subset);
                color_subset_indices.push(subset_idx);
            }
        }

        // We perform some expensive consistency checks in debug builds.
        debug_assert!(DisjointSubsets::try_from_disjoint_subsets(
            color_subsets.clone(),
            color_subset_indices.clone()).is_ok());

        // Subsets must be disjoint by construction, so skip checks
        let color = unsafe { DisjointSubsets::from_disjoint_subsets_unchecked(
            color_subsets,
            color_subset_indices,
            Some(max_node_idx)
        ) };
        colors.push(color);
        mem::swap(&mut postponed_subset_indices, &mut current_subset_indices);
        postponed_subset_indices.clear();
        color_idx = color_idx.checked_add(1)
            .expect("Number of colors exceeded i32::MAX.\
                     Please file an issue with your use case if you actually need that many colors.");
    }

    colors
}

#[cfg(test)]
mod tests {
    use super::sequential_greedy_coloring;
    use crate::DisjointSubsets;
    use fenris_nested_vec::NestedVec;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::proptest;

    proptest! {
        #[test]
        fn sequential_greedy_coloring_produces_disjoint_subsets(
            integer_subsets in vec(vec(0 .. 100usize, 0 .. 10), 0 .. 10)
        ) {
            // Generate a random Vec<Vec<usize>>, which can be interpreted as a set of
            // subsets, which we then color
            let subsets = NestedVec::from(&integer_subsets);
            let colors = sequential_greedy_coloring(&subsets);

            // There can not be more colors than there are original subsets
            prop_assert!(colors.len() <= subsets.len());

            let num_subsets_across_colors: usize = colors
                .iter()
                .map(|disjoint_subsets| disjoint_subsets.subsets().len())
                .sum();

            prop_assert_eq!(num_subsets_across_colors, subsets.len());

            // Actually assert that each color has disjoint subsets by running it through
            // the `try...` constructor for DisjointSubsets
            for subset in colors {
                let checked_subsets = DisjointSubsets::try_from_disjoint_subsets(
                    subset.subsets().clone(), subset.labels().to_vec());
                prop_assert_eq!(checked_subsets, Ok(subset.clone()));
            }
        }
    }
}

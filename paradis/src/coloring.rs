use crate::DisjointSubsets;
use nested_vec::NestedVec;
use std::collections::BTreeSet;

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

    fn try_add_subset(
        &mut self,
        subset: &[usize],
        label: usize,
        local_workspace_set: &mut BTreeSet<usize>,
    ) -> bool {
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

pub fn sequential_greedy_coloring(subsets: &NestedVec<usize>) -> Vec<DisjointSubsets> {
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
            unsafe {
                DisjointSubsets::from_disjoint_subsets_unchecked(
                    color.subsets,
                    color.labels,
                    max_index,
                )
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::sequential_greedy_coloring;
    use crate::DisjointSubsets;
    use nested_vec::NestedVec;
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

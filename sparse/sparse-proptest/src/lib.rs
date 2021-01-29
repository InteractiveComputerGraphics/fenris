use nalgebra::Scalar;
use num::Zero;
use proptest::collection::{btree_set, vec};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Reason, TestRunner};
use sparse::{CooMatrix, CsrMatrix, SparsityPattern};
use std::cmp::{max, min};
use std::iter::once;
use std::sync::Arc;
use util::prefix_sum;

/// Generates `coo` matrices, possibly with duplicate elements, with the given
/// maximum rows, columns and number of elements per row, with values being drawn
/// from the provided strategy.
pub fn coo<T, S>(
    max_rows: usize,
    max_cols: usize,
    max_elem_per_row: usize,
    value_strategy: S,
) -> impl Strategy<Value = CooMatrix<T>>
where
    T: Scalar + Zero,
    S: Strategy<Value = T> + Clone,
{
    let num_rows = 0..=max_rows;
    let num_cols = 0..=max_cols;

    (num_rows, num_cols)
        .prop_flat_map(move |(r, c)| (Just(r), Just(c), 0..=max(max_elem_per_row, c)))
        .prop_flat_map(move |(r, c, num_elem_per_row)| {
            // The minimum ensures that `nnz == 0` if `c == 0` or `r == 0` (i.e. empty matrix)
            let nnz = num_elem_per_row * r * min(1, c);
            // 0 .. 0 causes problems for `proptest`/`rand`, so
            // we must give it a valid range in all circumstances,
            // yet at the same time we must preserve the same type for things to compile.
            // However, if `r == 0` or `c == 0`, then `nnz == 0`, so we will not actually
            // generate any elements.
            let r_range = if r > 0 { 0..r } else { 0..1 };
            let c_range = if c > 0 { 0..c } else { 0..1 };
            (
                Just(r),
                Just(c),
                vec((r_range, c_range, value_strategy.clone()), nnz),
            )
        })
        .prop_map(|(num_rows, num_cols, triplets)| {
            let mut coo = CooMatrix::new(num_rows, num_cols);
            for (i, j, v) in triplets {
                coo.push(i, j, v);
            }
            coo
        })
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparsityPatternStrategy<ShapeStrategy, MinorsPerMajorStrategy> {
    shape_strategy: ShapeStrategy,
    minors_per_major: MinorsPerMajorStrategy,
}

impl SparsityPatternStrategy<(), ()> {
    pub fn new() -> Self {
        Self {
            shape_strategy: (),
            minors_per_major: (),
        }
    }
}

impl<ShapeStrategy, MinorsPerMajorStrategy>
    SparsityPatternStrategy<ShapeStrategy, MinorsPerMajorStrategy>
{
    pub fn with_shapes<S>(
        self,
        shape_strategy: S,
    ) -> SparsityPatternStrategy<S, MinorsPerMajorStrategy>
    where
        S: Strategy<Value = (usize, usize)>,
    {
        SparsityPatternStrategy {
            shape_strategy,
            minors_per_major: self.minors_per_major,
        }
    }

    pub fn with_num_minors_per_major<N>(
        self,
        strategy: N,
    ) -> SparsityPatternStrategy<ShapeStrategy, N>
    where
        N: Strategy<Value = usize>,
    {
        SparsityPatternStrategy {
            shape_strategy: self.shape_strategy,
            minors_per_major: strategy,
        }
    }
}

impl<ShapeStrategy, MinorsPerMajorStrategy> Strategy
    for SparsityPatternStrategy<ShapeStrategy, MinorsPerMajorStrategy>
where
    ShapeStrategy: Clone + 'static + Strategy<Value = (usize, usize)>,
    MinorsPerMajorStrategy: Clone + 'static + Strategy<Value = usize>,
{
    type Tree = Box<dyn ValueTree<Value = Self::Value>>;
    type Value = SparsityPattern;

    fn new_tree(&self, runner: &mut TestRunner) -> Result<Self::Tree, Reason> {
        let shape_strategy = self.shape_strategy.clone();
        let minors_per_major = self.minors_per_major.clone();
        shape_strategy
            .prop_flat_map(move |(major_dim, minor_dim)| {
                // Given major_dim and minor_dim, generate a vector of counts,
                // corresponding to the number of minor indices per major dimension entry
                let minors_per_major = minors_per_major
                    .clone()
                    .prop_map(move |count| min(count, minor_dim));
                vec(minors_per_major, major_dim)
                    .prop_flat_map(move |counts| {
                        // Construct offsets from counts
                        let offsets = prefix_sum(counts.iter().cloned().chain(once(0)), 0)
                            .collect::<Vec<_>>();

                        // We build one strategy per major entry (i.e. per row in a CSR matrix)
                        let mut major_strategies = Vec::with_capacity(major_dim);
                        for count in counts {
                            if 10 * count <= minor_dim {
                                // If we require less than approx. 10% of minor_dim,
                                // every pick is at least 90% likely to not be an index
                                // we already picked, so we can generate a set
                                major_strategies.push(
                                    btree_set(0..minor_dim, count)
                                        .prop_map(|indices| indices.into_iter().collect::<Vec<_>>())
                                        .boxed(),
                                )
                            } else {
                                // Otherwise, we simply shuffle the integers
                                // [0, minor_dim) and take the `count` first
                                let strategy = Just((0..minor_dim).collect::<Vec<_>>())
                                    .prop_shuffle()
                                    .prop_map(move |mut indices| {
                                        let indices = &mut indices[0..count];
                                        indices.sort_unstable();
                                        indices.to_vec()
                                    })
                                    .boxed();
                                major_strategies.push(strategy);
                            }
                        }
                        (
                            Just(major_dim),
                            Just(minor_dim),
                            Just(offsets),
                            major_strategies,
                        )
                    })
                    .prop_map(
                        move |(major_dim, minor_dim, offsets, minor_indices_by_major)| {
                            let minor_indices: Vec<usize> =
                                minor_indices_by_major.into_iter().flatten().collect();
                            SparsityPattern::from_offsets_and_indices(
                                major_dim,
                                minor_dim,
                                offsets,
                                minor_indices,
                            )
                        },
                    )
            })
            .boxed()
            .new_tree(runner)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CsrStrategy<ElementStrategy, ShapeStrategy, ColsPerRowStrategy> {
    pattern_strategy: SparsityPatternStrategy<ShapeStrategy, ColsPerRowStrategy>,
    element_strategy: ElementStrategy,
}

impl CsrStrategy<(), (), ()> {
    pub fn new() -> Self {
        Self {
            pattern_strategy: SparsityPatternStrategy::new(),
            element_strategy: (),
        }
    }
}

impl<ElementStrategy, ShapeStrategy, ColsPerRowStrategy>
    CsrStrategy<ElementStrategy, ShapeStrategy, ColsPerRowStrategy>
{
    pub fn with_elements<E>(
        self,
        element_strategy: E,
    ) -> CsrStrategy<E, ShapeStrategy, ColsPerRowStrategy>
    where
        E: Strategy,
    {
        CsrStrategy {
            pattern_strategy: self.pattern_strategy,
            element_strategy,
        }
    }

    pub fn with_shapes<S>(
        self,
        shape_strategy: S,
    ) -> CsrStrategy<ElementStrategy, S, ColsPerRowStrategy>
    where
        S: Strategy<Value = (usize, usize)>,
    {
        let pattern = self.pattern_strategy.with_shapes(shape_strategy);
        CsrStrategy {
            pattern_strategy: pattern,
            element_strategy: self.element_strategy,
        }
    }

    pub fn with_cols_per_row<N>(
        self,
        cols_per_row_strategy: N,
    ) -> CsrStrategy<ElementStrategy, ShapeStrategy, N>
    where
        N: Strategy<Value = usize>,
    {
        let pattern = self
            .pattern_strategy
            .with_num_minors_per_major(cols_per_row_strategy);
        CsrStrategy {
            pattern_strategy: pattern,
            element_strategy: self.element_strategy,
        }
    }
}

impl<ElementStrategy, ShapeStrategy, MinorsPerMajorStrategy> Strategy
    for CsrStrategy<ElementStrategy, ShapeStrategy, MinorsPerMajorStrategy>
where
    ElementStrategy: Clone + 'static + Strategy,
    ShapeStrategy: Clone + 'static + Strategy<Value = (usize, usize)>,
    MinorsPerMajorStrategy: Clone + 'static + Strategy<Value = usize>,
{
    type Tree = Box<dyn ValueTree<Value = Self::Value>>;
    type Value = CsrMatrix<ElementStrategy::Value>;

    fn new_tree(&self, runner: &mut TestRunner) -> Result<Self::Tree, Reason> {
        let element_strategy = self.element_strategy.clone();
        let pattern_strategy = self.pattern_strategy.clone();
        pattern_strategy
            .prop_flat_map(move |pattern| {
                let nnz = pattern.nnz();
                (Just(pattern), vec(element_strategy.clone(), nnz))
            })
            .prop_map(|(pattern, values)| {
                CsrMatrix::from_pattern_and_values(Arc::new(pattern), values)
            })
            .boxed()
            .new_tree(runner)
    }
}

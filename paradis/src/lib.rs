//! paradis
//! =======
//!
//! Parallel processing of disjoint subsets.

pub mod adapter;
pub mod coloring;
pub mod slice;

use fenris_nested_vec::NestedVec;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::HashSet;
use std::fmt;
use std::fmt::Debug;

pub struct SubsetAccess<'data, Access> {
    subset_label: usize,
    global_indices: &'data [usize],
    access: Access,
}

impl<'data, Access> SubsetAccess<'data, Access> {
    pub fn global_indices(&self) -> &[usize] {
        &self.global_indices
    }

    pub fn label(&self) -> usize {
        self.subset_label
    }

    pub fn len(&self) -> usize {
        self.global_indices().len()
    }

    pub fn get<'b>(&'b self, local_index: usize) -> <Access as ParallelIndexedAccess<'b>>::Record
    where
        'data: 'b,
        Access: ParallelIndexedAccess<'b>,
    {
        let global_index = self.global_indices[local_index];
        unsafe { self.access.get_unchecked(global_index) }
    }

    pub fn get_mut<'b>(&'b mut self, local_index: usize) -> <Access as ParallelIndexedAccess<'b>>::RecordMut
    where
        'data: 'b,
        Access: ParallelIndexedAccess<'b>,
    {
        let global_index = self.global_indices[local_index];
        unsafe { self.access.get_unchecked_mut(global_index) }
    }
}

// TODO: Does this trait need to be unsafe, or does it suffice to have unsafe methods?
/// Facilitates parallel access to (mutable) records stored in the collection.
///
/// The trait provides parallel access to (possibly mutable) *records*, defined by the
/// associated types [`Record`][`ParallelIndexedAccess::Record`] and
/// [`RecordMut`][`ParallelIndexedAccess::RecordMut`].
///
/// # Safety
///
/// An implementor must ensure that it is sound for multiple threads to access a single record
/// *immutably*, provided that no thread accesses the same record mutably.
///
/// An implementor must furthermore ensure that it is sound for multiple threads to access
/// *disjoint* records mutably.
///
/// It is the responsibility of the consumer that:
///
/// - If any thread accesses a record mutably, then no other thread must access the same record.
/// - A mutable record must always be exclusive, even on a single thread.
///   In particular, a single thread is not permitted to obtain two records associated with the
///   same index in the collection if either record is mutable.
///
/// TODO: Make the invariants more precise
///
/// TODO: Can consider a slightly different API in which each thread must obtain its own access.
pub unsafe trait ParallelIndexedAccess<'record>: Sync + Send + Clone {
    type Record;
    type RecordMut;

    unsafe fn get_unchecked(&self, index: usize) -> Self::Record;
    unsafe fn get_unchecked_mut(&self, index: usize) -> Self::RecordMut;
}

/// An indexed collection that exposes parallel indexed access to its contents.
///
/// The typical pattern for a generic parallel algorithm is to take any collection satisfying
/// this trait as input, at which point we can guarantee that we are able to create the only
/// parallel access to the collection.
///
/// # Examples
///
/// Let's consider a contrived example in which we want to multiply every number in a list by 2,
/// and we want to multiply even and odd numbers on two separate threads. This is of course a
/// *very bad* idea, for a number of reasons, but it does serve as a very simple example of
/// a *sound* use of parallelism that nonetheless falls outside of what we can accomplish with
/// safe Rust.
///
/// Although we'll have to use some `unsafe` code to accomplish this, the task sounds fairly
/// trivial. However, we would soon discover that the only safe way of accessing different
/// (scattered) elements of a slice mutably in parallel is through careful pointer manipulation.
/// This is generally error prone and hard to get right.
///
/// The abstractions made available by `paradis` significantly simplifies this. Since slices
/// implement the [`ParallelIndexedCollection`] trait, we can obtain *parallel access* to its
/// elements, which lets us (unsafely) obtain access to any element in the slice without
/// working with pointer arithmetic. We are of course wholly responsible for making sure that
/// we never access the same element from two threads.
///
/// ```rust
/// use paradis::{ParallelIndexedCollection, ParallelIndexedAccess};
/// use crossbeam::scope;
///
/// fn par_double_all_numbers(numbers: &mut [i32]) {
///     let n = numbers.len();
///
///     // Since creating an access takes a mutable reference to [i32], we know that we are the
///     // only ones to hold a parallel access to the data throughout the program, so we can
///     // soundly manipulate its data in parallel, provided we take some care
///     let access = unsafe { numbers.create_access() };
///
///     // The standard library does not have a concept of scoped threads,
///     // so we use crossbeam::scope for this purpose.
///     // Otherwise we wouldn't be able to use our access across threads.
///     scope(|s| {
///         s.spawn(|_| {
///             // Transform the even numbers
///             for i in (0 .. n).step_by(2) {
///                 unsafe { *access.get_unchecked_mut(i) *= 2; }
///             }
///         });
///         s.spawn(|_| {
///             // Transform the odd numbers
///             for i in (1 .. n).step_by(2) {
///                 unsafe { *access.get_unchecked_mut(i) *= 2; }
///             }
///         });
///     }).expect("One of our threads panicked!");
/// }
///
/// fn main() {
///     let mut numbers = [0, 1, 2, 3, 4, 5, 6, 7];
///     par_double_all_numbers(&mut numbers);
///     assert_eq!(numbers, [0, 2, 4, 6, 8, 10, 12, 14]);
/// }
/// ```
///
/// # Safety
///
/// This trait is unsafe because the soundness of consuming code relies on the correctness of
/// the implementation of [`ParallelIndexedCollection::len`].
/// Consumers of this trait are permitted to access records
/// (accessed through [`ParallelIndexedCollection::Access`]) with indices `[0, len)`. Therefore,
/// an incorrect length may lead to unsoundness.
pub unsafe trait ParallelIndexedCollection<'a> {
    type Access;

    unsafe fn create_access(&'a mut self) -> Self::Access;
    fn len(&self) -> usize;
}

/// A set of subsets of indices, in which the intersection of indices between any two subsets is
/// empty.
///
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisjointSubsets {
    // Store the max global index present in any of the subsets. We need this to
    // ensure that none of the indices are out of bounds when accessing a storage.
    max_index: Option<usize>,
    // Each subset consists of a set of indices. Indices are allowed to overlap within a subset,
    // but the intersection between the indices of any two subsets must be empty. In other words,
    // no two subsets share a common index.
    subsets: NestedVec<usize>,
    // Store a label for each subset
    labels: Vec<usize>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SubsetsNotDisjointError;

impl DisjointSubsets {
    pub fn try_from_disjoint_subsets<Subsets: Into<NestedVec<usize>>>(
        subsets: Subsets,
        labels: Vec<usize>,
    ) -> Result<Self, SubsetsNotDisjointError> {
        let subsets = subsets.into();
        assert_eq!(subsets.len(), labels.len(), "Must have exactly one label per subset.");

        let mut max_index = None;
        let mut global_index_set = HashSet::new();
        // Subsets are allowed to contain duplicate entries, so we therefore build a local index
        // set for each subset before checking against and adding them to the global index set.
        let mut local_index_set = HashSet::new();

        // Verify that subsets are disjoint
        for subset in subsets.iter() {
            local_index_set.clear();
            for idx in subset {
                if let Some(ref mut current_max) = max_index {
                    *current_max = max(*current_max, *idx);
                } else {
                    max_index = Some(*idx);
                }
                local_index_set.insert(*idx);
            }

            for idx in &local_index_set {
                let idx_already_present = !global_index_set.insert(*idx);
                if idx_already_present {
                    return Err(SubsetsNotDisjointError);
                }
            }
        }

        let disjoint_subsets = DisjointSubsets {
            max_index,
            subsets,
            labels,
        };

        Ok(disjoint_subsets)
    }

    pub unsafe fn from_disjoint_subsets_unchecked<Subsets: Into<NestedVec<usize>>>(
        subsets: Subsets,
        labels: Vec<usize>,
        max_index: Option<usize>,
    ) -> Self {
        let subsets = subsets.into();
        assert_eq!(subsets.len(), labels.len(), "Must have exactly one label per subset.");
        Self {
            max_index,
            subsets: subsets.into(),
            labels,
        }
    }

    pub fn subsets(&self) -> &NestedVec<usize> {
        &self.subsets
    }

    pub fn into_subsets(self) -> NestedVec<usize> {
        self.subsets
    }

    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Create a parallel iterator over the subsets, fetching data from the provided storage.
    ///
    /// Panics if any subset contains an index that exceeds the length reported by `storage`.
    pub fn subsets_par_iter<'a, Storage>(
        &'a self,
        storage: &'a mut Storage,
    ) -> DisjointSubsetsParIter<'a, Storage::Access>
    where
        Storage: ?Sized + ParallelIndexedCollection<'a>,
    {
        assert!(
            self.max_index.is_none() || storage.len() > self.max_index.unwrap(),
            "Subsets contain indices out of bounds."
        );
        // Sanity check: if we don't have a max index, then we also cannot have any subsets
        debug_assert_eq!(self.max_index.is_none(), self.subsets.len() == 0);
        let access = unsafe { storage.create_access() };

        DisjointSubsetsParIter {
            access,
            subsets: &self.subsets,
            labels: &self.labels,
        }
    }
}

pub struct DisjointSubsetsParIter<'a, Access> {
    access: Access,
    subsets: &'a NestedVec<usize>,
    labels: &'a [usize],
}

impl<'a, Access: Send + Clone> ParallelIterator for DisjointSubsetsParIter<'a, Access> {
    type Item = SubsetAccess<'a, Access>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'a, Access: Send + Clone> IndexedParallelIterator for DisjointSubsetsParIter<'a, Access> {
    fn len(&self) -> usize {
        self.subsets.len()
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> <C as Consumer<Self::Item>>::Result {
        bridge(self, consumer)
    }

    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let num_subsets = self.subsets.len();
        callback.callback(DisjointSubsetsProducer {
            access: self.access,
            subsets: &self.subsets,
            labels: self.labels,
            range_start_idx: 0,
            range_len: num_subsets,
        })
    }
}

struct DisjointSubsetsProducer<'a, Access> {
    access: Access,
    subsets: &'a NestedVec<usize>,
    labels: &'a [usize],
    // Range start/len represents the range represented by this producer
    range_start_idx: usize,
    range_len: usize,
}

impl<'a, Access> Debug for DisjointSubsetsProducer<'a, Access> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjointSubsetsProducer")
            // .field("access", &"<not debuggable>")
            // .field("subsets", &self.subsets)
            .field("range_start_idx", &self.range_start_idx)
            .field("range_len", &self.range_len)
            .finish()
    }
}

impl<'a, Access: Send + Clone> Producer for DisjointSubsetsProducer<'a, Access> {
    type Item = SubsetAccess<'a, Access>;
    type IntoIter = DisjointSubsetsIter<'a, Access>;

    fn into_iter(self) -> Self::IntoIter {
        DisjointSubsetsIter {
            access: self.access.clone(),
            subsets: self.subsets,
            labels: self.labels,
            end: self.range_len + self.range_start_idx,
            current_idx: self.range_start_idx,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let producer_len = self.range_len;
        assert!(index < producer_len);
        let global_subset_idx = self.range_start_idx + index;

        let producer_left = DisjointSubsetsProducer {
            access: self.access.clone(),
            subsets: self.subsets,
            labels: self.labels,
            range_start_idx: self.range_start_idx,
            range_len: index,
        };

        let producer_right = DisjointSubsetsProducer {
            access: self.access,
            subsets: self.subsets,
            labels: self.labels,
            range_start_idx: global_subset_idx,
            range_len: producer_len - index,
        };

        (producer_left, producer_right)
    }
}

struct DisjointSubsetsIter<'a, Access> {
    access: Access,
    subsets: &'a NestedVec<usize>,
    labels: &'a [usize],
    // end is an index one-past the end of the iterator
    end: usize,
    // The current index that the iterator is at
    current_idx: usize,
}

impl<'a, Access> Debug for DisjointSubsetsIter<'a, Access> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DisjointSubsetsIter")
            // .field("access", &"<not debuggable>")
            // .field("subsets", &self.subsets)
            .field("end", &self.end)
            .field("current_idx", &self.current_idx)
            .finish()
    }
}

impl<'a, Access: Clone> Iterator for DisjointSubsetsIter<'a, Access> {
    type Item = SubsetAccess<'a, Access>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx < self.end {
            let access = SubsetAccess {
                subset_label: *self.labels.get(self.current_idx).unwrap(),
                global_indices: self.subsets.get(self.current_idx).unwrap(),
                access: self.access.clone(),
            };
            self.current_idx += 1;
            Some(access)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.current_idx;
        (len, Some(len))
    }
}

impl<'a, Access: Clone> ExactSizeIterator for DisjointSubsetsIter<'a, Access> {}

impl<'a, Access: Clone> DoubleEndedIterator for DisjointSubsetsIter<'a, Access> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end > self.current_idx {
            let subset_index = self.end - 1;
            let access = SubsetAccess {
                subset_label: *self.labels.get(subset_index).unwrap(),
                global_indices: self.subsets.get(subset_index).unwrap(),
                access: self.access.clone(),
            };
            self.end -= 1;
            Some(access)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DisjointSubsets;
    use super::DisjointSubsetsIter;
    use super::ParallelIndexedCollection;
    use fenris_nested_vec::NestedVec;
    use proptest::collection::{btree_set, vec};
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    #[test]
    fn test_disjoint_subsets_iter() {
        let subsets_vec = vec![vec![4, 5], vec![1, 2, 3], vec![6, 0]];
        let subset_labels = vec![0, 1, 2];
        let subsets = NestedVec::from(&subsets_vec);

        // Forward iteration only
        {
            // Range is over all subsets
            let mut data = vec![10, 11, 12, 13, 14, 15, 16];
            let data_slice = data.as_mut_slice();

            let access = unsafe { data_slice.create_access() };

            let mut iter = DisjointSubsetsIter {
                access,
                subsets: &subsets,
                labels: &subset_labels,
                end: subsets.len(),
                current_idx: 0,
            };

            assert_eq!(iter.len(), 3);
            let subset_access = iter.next().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[0].as_slice());
            assert_eq!(iter.len(), 2);
            let subset_access = iter.next().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[1].as_slice());
            assert_eq!(iter.len(), 1);
            let subset_access = iter.next().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[2].as_slice());
            assert_eq!(iter.len(), 0);
            assert!(iter.next().is_none());
        }

        // Forward iteration only
        {
            // Range is over subset
            let mut data = vec![10, 11, 12, 13, 14, 15, 16];
            let data_slice = data.as_mut_slice();

            let access = unsafe { data_slice.create_access() };

            let mut iter = DisjointSubsetsIter {
                access,
                subsets: &subsets,
                labels: &subset_labels,
                end: subsets.len(),
                current_idx: 1,
            };

            assert_eq!(iter.len(), 2);
            let subset_access = iter.next().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[1].as_slice());
            assert_eq!(iter.len(), 1);
            let subset_access = iter.next().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[2].as_slice());
            assert_eq!(iter.len(), 0);
            assert!(iter.next().is_none());
        }

        // Backward iteration only
        {
            // Range is over subset
            let mut data = vec![10, 11, 12, 13, 14, 15, 16];
            let data_slice = data.as_mut_slice();

            let access = unsafe { data_slice.create_access() };

            let mut iter = DisjointSubsetsIter {
                access,
                subsets: &subsets,
                labels: &subset_labels,
                end: subsets.len(),
                current_idx: 0,
            };

            assert_eq!(iter.len(), 3);
            let subset_access = iter.next_back().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[2].as_slice());
            assert_eq!(iter.len(), 2);
            let subset_access = iter.next_back().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[1].as_slice());
            assert_eq!(iter.len(), 1);
            let subset_access = iter.next_back().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[0].as_slice());
            assert_eq!(iter.len(), 0);
            assert!(iter.next().is_none());
        }

        // Backward iteration only
        {
            // Range is over subset
            let mut data = vec![10, 11, 12, 13, 14, 15, 16];
            let data_slice = data.as_mut_slice();

            let access = unsafe { data_slice.create_access() };

            let mut iter = DisjointSubsetsIter {
                access,
                subsets: &subsets,
                labels: &subset_labels,
                end: subsets.len() - 1,
                current_idx: 0,
            };

            assert_eq!(iter.len(), 2);
            let subset_access = iter.next_back().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[1].as_slice());
            assert_eq!(iter.len(), 1);
            let subset_access = iter.next_back().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[0].as_slice());
            assert_eq!(iter.len(), 0);
            assert!(iter.next().is_none());
        }

        // Forward and backward iteration
        {
            // Range is over subset
            let mut data = vec![10, 11, 12, 13, 14, 15, 16];
            let data_slice = data.as_mut_slice();

            let access = unsafe { data_slice.create_access() };

            let mut iter = DisjointSubsetsIter {
                access,
                subsets: &subsets,
                labels: &subset_labels,
                end: subsets.len(),
                current_idx: 0,
            };

            assert_eq!(iter.len(), 3);
            let subset_access = iter.next().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[0].as_slice());
            assert_eq!(iter.len(), 2);
            let subset_access = iter.next_back().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[2].as_slice());
            assert_eq!(iter.len(), 1);
            let subset_access = iter.next().unwrap();
            assert_eq!(subset_access.global_indices(), subsets_vec[1].as_slice());
            assert_eq!(iter.len(), 0);
            assert!(iter.next_back().is_none());
            assert!(iter.next().is_none());
            assert!(iter.next().is_none());
            assert!(iter.next_back().is_none());
            assert!(iter.next_back().is_none());
            assert!(iter.next().is_none());
        }
    }

    #[test]
    fn test_parallel() {
        // TODO: Fixed seed
        let mut rng = StdRng::seed_from_u64(458340234234);

        let mut unique_indices: Vec<_> = (0..100000).collect();
        unique_indices.shuffle(&mut rng);

        let chunks: Vec<_> = unique_indices
            .chunks(10)
            .map(|chunk| chunk.to_vec())
            .collect();

        let labels = (0..chunks.len()).collect();

        let disjoint_subsets = DisjointSubsets::try_from_disjoint_subsets(&chunks, labels).unwrap();

        let mut output_par = vec![0; unique_indices.len()];
        disjoint_subsets
            .subsets_par_iter(output_par.as_mut_slice())
            .zip_eq(&chunks)
            // Try to ensure that rayon actually uses multiple threads, otherwise it might
            // decide to run it all sequentially
            .with_max_len(1)
            .for_each(|(mut subset_access, chunk)| {
                assert_eq!(subset_access.global_indices(), chunk.as_slice());
                for i in 0..chunk.len() {
                    *subset_access.get_mut(i) += 1;
                }
            });

        let mut output_seq = vec![0; unique_indices.len()];
        chunks.iter().for_each(|chunk| {
            for i in 0..chunk.len() {
                output_seq[chunk[i]] += 1;
            }
        });

        let expected_output = vec![1; unique_indices.len()];
        assert_eq!(output_seq, expected_output);
        assert_eq!(output_par, expected_output);
    }

    // TODO: Test the strategy itself!
    // TODO: Our current strategy also enforces that the subsets have no duplicate indices,
    // which is explicitly allowed by our algorithms, so we should include this too
    fn disjoint_subsets_strategy() -> impl Strategy<Value = NestedVec<usize>> {
        let max_num_integers = 20usize;
        (0..max_num_integers)
            .prop_flat_map(|n| Just((0..n).collect::<Vec<_>>()))
            .prop_shuffle()
            .prop_flat_map(|integers| {
                let n = integers.len();
                let num_splits = 0..=n;
                let split_indices = vec(0..n, num_splits);
                (Just(integers), split_indices)
            })
            .prop_map(|(integers, mut split_indices)| {
                let mut subsets = Vec::with_capacity(split_indices.len() + 1);
                split_indices.push(0);
                split_indices.push(integers.len());
                split_indices.sort_unstable();
                for window in split_indices.windows(2) {
                    let idx = window[0];
                    let idx_next = window[1];
                    subsets.push(integers[idx..idx_next].to_vec());
                }
                NestedVec::from(&subsets)
            })
    }

    fn overlapping_subsets_strategy() -> impl Strategy<Value = NestedVec<usize>> {
        // Given a set of overlapping subsets, add the same index to multiple subsets,
        // thereby ensuring that the subsets are no longer disjoint
        let max_index = 20usize;
        disjoint_subsets_strategy()
            .prop_filter("Must have more than 1 subset", |subsets| subsets.len() > 1)
            .prop_flat_map(move |subsets| {
                let insertion_index = 0..max_index;
                let subset_index_strategy = btree_set(0..subsets.len(), 2..=subsets.len());
                (Just(subsets), subset_index_strategy, insertion_index)
            })
            .prop_map(|(subsets, subset_indices, insertion_index)| {
                let mut subsets: Vec<Vec<_>> = subsets.into();
                let num_subsets = subsets.len();
                for subset_idx in subset_indices {
                    subsets[subset_idx % num_subsets].push(insertion_index);
                }
                NestedVec::from(subsets)
            })
    }

    proptest! {
        #[test]
        fn can_create_from_disjoint_subsets(
            disjoint_subsets in disjoint_subsets_strategy()
        ) {
            let labels = (0 .. disjoint_subsets.len()).collect();
            let disjoint = DisjointSubsets::try_from_disjoint_subsets(disjoint_subsets, labels);
            dbg!(&disjoint);
            prop_assert!(disjoint.is_ok());
        }

        #[test]
        fn refuses_to_create_from_overlapping_subsets(
            subsets in overlapping_subsets_strategy()
        ) {
            let labels = (0 .. subsets.len()).collect();
            let disjoint = DisjointSubsets::try_from_disjoint_subsets(subsets, labels);
            prop_assert!(disjoint.is_err());
        }
    }
}

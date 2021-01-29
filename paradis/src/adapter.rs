use crate::{ParallelAccess, ParallelStorage};
use std::marker::PhantomData;

/// An adapter that facilitates blocked storage.
///
/// Many problems naturally store data in blocks. As an example, consider a vector `x` consisting
/// of `n` 3-element vectors `x_i`, stored contiguously as
/// `x = vec![x_11, x_12, x_13, x_21, ..., x_n1, x_n2, x_n3]`.
/// In order to consider subsets of indices in the range `0 .. 3*n`, we may instead consider
/// subsets of indices in the range `0 .. n`, each index corresponding to a *block* of `x`.
/// This adapter facilitates these kind of storage patterns by transparently making
/// `DisjointSubsets` create parallel iterators over subsets of *blocks*.
///
/// For illustration, see the below example.
///
/// ```rust
/// use paradis::DisjointSubsets;
/// use paradis::adapter::BlockAdapter;
/// use rayon::iter::ParallelIterator;
///
/// // 7 blocks of 3
/// let mut data = vec![0; 21];
///
/// let subsets = vec![
///     vec![1, 3, 5],
///     vec![0, 2],
///     vec![4, 6]
/// ];
/// let subsets = DisjointSubsets::try_from_disjoint_subsets(&subsets).unwrap();
///
/// let mut adapter = BlockAdapter::with_block_size(data.as_mut_slice(), 3);
/// subsets.subsets_par_iter(&mut adapter)
///     .for_each(|mut subset| {
///         for i in 0 .. subset.len() {
///             // Each subset consists of blocks `subset.len()` blocks
///             let mut block_i = subset.get_mut(i);
///             *block_i.index_mut(0) += 1;
///             *block_i.index_mut(1) += 2;
///             *block_i.index_mut(2) += 3;
///         }
///     });
///
/// assert_eq!(data, vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);
/// ```
///
///
///
///
///
#[derive(Debug)]
pub struct BlockAdapter<'a, Storage: ?Sized> {
    storage: &'a mut Storage,
    block_size: usize,
}

impl<'a, Storage> BlockAdapter<'a, Storage>
where
    Storage: ?Sized + ParallelStorage<'a>,
{
    pub fn with_block_size(storage: &'a mut Storage, block_size: usize) -> Self {
        assert!(block_size > 0);
        assert_eq!(
            storage.len() % block_size,
            0,
            "Storage length must be divisible by block size."
        );
        Self {
            storage,
            block_size,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Block<'a, Access> {
    access: Access,
    start_idx: usize,
    block_size: usize,
    marker: PhantomData<&'a Access>,
}

impl<'a, 'b, Access> Block<'a, Access>
where
    'a: 'b,
    Access: ParallelAccess<'b>,
{
    pub fn len(&self) -> usize {
        self.block_size
    }

    pub fn get(&'b self, index_in_block: usize) -> Option<Access::Record> {
        if index_in_block < self.block_size {
            let global_index = self.start_idx + index_in_block;
            Some(unsafe { self.access.get_unchecked(global_index) })
        } else {
            None
        }
    }

    pub fn index(&'b self, index_in_block: usize) -> Access::Record {
        self.get(index_in_block).expect("Index must be in bounds")
    }
}

#[derive(Copy, Clone)]
pub struct BlockMut<'a, Access> {
    access: Access,
    start_idx: usize,
    block_size: usize,
    marker: PhantomData<&'a mut Access>,
}

impl<'a, 'b, Access> BlockMut<'a, Access>
where
    'a: 'b,
    Access: ParallelAccess<'b>,
{
    pub fn len(&self) -> usize {
        self.block_size
    }

    pub fn get(&'b self, index_in_block: usize) -> Option<Access::Record> {
        if index_in_block < self.block_size {
            let global_index = self.start_idx + index_in_block;
            Some(unsafe { self.access.get_unchecked(global_index) })
        } else {
            None
        }
    }

    pub fn get_mut(&'b mut self, index_in_block: usize) -> Option<Access::RecordMut> {
        if index_in_block < self.block_size {
            let global_index = self.start_idx + index_in_block;
            Some(unsafe { self.access.get_unchecked_mut(global_index) })
        } else {
            None
        }
    }

    pub fn index(&'b self, index_in_block: usize) -> Access::Record {
        self.get(index_in_block).expect("Index must be in bounds")
    }

    pub fn index_mut(&'b mut self, index_in_block: usize) -> Access::RecordMut {
        self.get_mut(index_in_block)
            .expect("Index must be in bounds")
    }
}

#[derive(Copy, Clone)]
pub struct BlockAccess<Access> {
    access: Access,
    block_size: usize,
}

unsafe impl<'a, Access> ParallelAccess<'a> for BlockAccess<Access>
where
    Access: 'a + ParallelAccess<'a>,
{
    type Record = Block<'a, Access>;
    type RecordMut = BlockMut<'a, Access>;

    unsafe fn get_unchecked(&'a self, global_index: usize) -> Self::Record {
        Block {
            access: self.access.clone(),
            start_idx: self.block_size * global_index,
            block_size: self.block_size,
            marker: PhantomData,
        }
    }

    unsafe fn get_unchecked_mut(&'a self, global_index: usize) -> Self::RecordMut {
        BlockMut {
            access: self.access.clone(),
            start_idx: self.block_size * global_index,
            block_size: self.block_size,
            marker: PhantomData,
        }
    }
}

unsafe impl<'a, Storage> ParallelStorage<'a> for BlockAdapter<'a, Storage>
where
    Storage: ?Sized + ParallelStorage<'a>,
{
    type Access = BlockAccess<Storage::Access>;

    fn create_access(&'a mut self) -> Self::Access {
        BlockAccess {
            access: self.storage.create_access(),
            block_size: self.block_size,
        }
    }

    fn len(&self) -> usize {
        self.storage.len() / self.block_size
    }
}

#[cfg(test)]
mod tests {
    use crate::adapter::BlockAdapter;
    use crate::DisjointSubsets;
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    #[test]
    fn blocked_slice_parallel_test() {
        // 7 blocks of 3
        let mut data = vec![0; 21];

        let subsets = vec![vec![1, 3, 5], vec![0, 2], vec![4, 6]];
        let labels = vec![0, 1, 2];
        let subsets = DisjointSubsets::try_from_disjoint_subsets(&subsets, labels).unwrap();

        let mut adapter = BlockAdapter::with_block_size(data.as_mut_slice(), 3);
        subsets
            .subsets_par_iter(&mut adapter)
            .with_min_len(1)
            .for_each(|mut subset| {
                for i in 0..subset.len() {
                    let mut block_i = subset.get_mut(i);
                    *block_i.index_mut(0) += 1;
                    *block_i.index_mut(1) += 2;
                    *block_i.index_mut(2) += 3;
                }
            });

        assert_eq!(
            data,
            vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        );
    }
}

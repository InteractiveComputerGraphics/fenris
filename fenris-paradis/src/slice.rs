use crate::{ParallelIndexedAccess, ParallelIndexedCollection};
use std::marker::PhantomData;
use std::ops::Range;

#[derive(Copy)]
pub struct ParallelSliceAccess<'a, T> {
    ptr: *mut T,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> ParallelSliceAccess<'a, T> {
    /// Construct a subslice for the given range.
    ///
    /// # Safety
    ///
    /// The range must be valid for the slice.
    ///
    /// Multiple threads must not call this function with overlapping ranges without synchronization.
    pub unsafe fn subslice_mut(&self, range: Range<usize>) -> &'a mut [T] {
        let Range { start, end } = range;
        debug_assert!(end >= start);
        let ptr = self.ptr.add(start);
        let len = end - start;
        std::slice::from_raw_parts_mut(ptr, len)
    }
}

impl<'a, T> Clone for ParallelSliceAccess<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            marker: PhantomData,
        }
    }
}

unsafe impl<'a, T: Sync> Sync for ParallelSliceAccess<'a, T> {}
unsafe impl<'a, T: Send> Send for ParallelSliceAccess<'a, T> {}

unsafe impl<'a, 'b, T: 'b + Sync + Send> ParallelIndexedAccess<'b> for ParallelSliceAccess<'a, T>
where
    'a: 'b,
{
    type Record = &'b T;
    type RecordMut = &'b mut T;

    unsafe fn get_unchecked(&self, global_index: usize) -> Self::Record {
        // TODO: This might technically be unsound. Should we use .wrapping_add, or something else?
        &*self.ptr.add(global_index)
    }

    unsafe fn get_unchecked_mut(&self, global_index: usize) -> Self::RecordMut {
        // TODO: This might technically be unsound. Should we use .wrapping_add, or something else?
        &mut *self.ptr.add(global_index)
    }
}

unsafe impl<'a, T: 'a + Sync + Send> ParallelIndexedCollection<'a> for [T] {
    type Access = ParallelSliceAccess<'a, T>;

    unsafe fn create_access(&'a mut self) -> Self::Access {
        ParallelSliceAccess {
            ptr: self.as_mut_ptr(),
            marker: PhantomData,
        }
    }

    fn len(&self) -> usize {
        <[T]>::len(&self)
    }
}

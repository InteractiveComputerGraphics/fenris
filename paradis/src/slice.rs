use crate::{ParallelAccess, ParallelStorage};
use std::marker::PhantomData;

#[derive(Copy)]
pub struct ParallelSliceAccess<'a, T> {
    ptr: *mut T,
    marker: PhantomData<&'a mut T>,
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

unsafe impl<'a, 'b, T: 'b + Sync + Send> ParallelAccess<'b> for ParallelSliceAccess<'a, T>
where
    'a: 'b,
{
    type Record = &'b T;
    type RecordMut = &'b mut T;

    unsafe fn get_unchecked(&self, global_index: usize) -> &T {
        // TODO: This might technically be unsound. Should we use .wrapping_add, or something else?
        &*self.ptr.add(global_index)
    }

    unsafe fn get_unchecked_mut(&self, global_index: usize) -> &mut T {
        // TODO: This might technically be unsound. Should we use .wrapping_add, or something else?
        &mut *self.ptr.add(global_index)
    }
}

unsafe impl<'a, T: 'a + Sync + Send> ParallelStorage<'a> for [T] {
    type Access = ParallelSliceAccess<'a, T>;

    fn create_access(&'a mut self) -> Self::Access {
        ParallelSliceAccess {
            ptr: self.as_mut_ptr(),
            marker: PhantomData,
        }
    }

    fn len(&self) -> usize {
        <[T]>::len(&self)
    }
}

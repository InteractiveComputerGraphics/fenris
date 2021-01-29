use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Debug;
use std::ops::Range;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NestedVec<T> {
    data: Vec<T>,
    offsets_begin: Vec<usize>,
    offsets_end: Vec<usize>,
}

impl<T: Debug> Debug for NestedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> Default for NestedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> NestedVec<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets_begin: Vec::new(),
            offsets_end: Vec::new(),
        }
    }

    /// Return a data structure that can be used for appending single elements to the same array.
    /// When the returned data structure is dropped, the result is equivalent to
    /// adding the array at once with `CompactArrayStorage::push`.
    ///
    /// TODO: Need better name
    pub fn begin_array<'a>(&'a mut self) -> ArrayAppender<'a, T> {
        let initial_count = self.data.len();
        ArrayAppender {
            initial_count,
            data: &mut self.data,
            offsets_begin: &mut self.offsets_begin,
            offsets_end: &mut self.offsets_end,
        }
    }

    pub fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = &'a [T]> {
        (0..self.len()).map(move |i| self.get(i).unwrap())
    }

    pub fn len(&self) -> usize {
        self.offsets_begin.len()
    }

    /// Returns an iterator over all elements inside all arrays.
    pub fn iter_array_elements<'a>(&'a self) -> impl 'a + Iterator<Item = &'a T> {
        self.iter().flatten()
    }

    pub fn total_num_elements(&self) -> usize {
        self.offsets_begin
            .iter()
            .zip(&self.offsets_end)
            .map(|(begin, end)| end - begin)
            .sum()
    }

    pub fn get(&self, index: usize) -> Option<&[T]> {
        let range = self.get_index_range(index)?;
        self.data.get(range)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut [T]> {
        let range = self.get_index_range(index)?;
        self.data.get_mut(range)
    }

    fn get_index_range(&self, index: usize) -> Option<Range<usize>> {
        let begin = *self.offsets_begin.get(index)?;
        let end = *self.offsets_end.get(index)?;
        Some(begin..end)
    }

    pub fn first(&self) -> Option<&[T]> {
        self.get(0)
    }

    pub fn first_mut(&mut self) -> Option<&mut [T]> {
        self.get_mut(0)
    }

    fn get_last_range(&self) -> Option<Range<usize>> {
        let begin = *self.offsets_begin.last()?;
        let end = *self.offsets_end.last()?;
        Some(begin..end)
    }

    pub fn last(&self) -> Option<&[T]> {
        let range = self.get_last_range()?;
        self.data.get(range)
    }

    pub fn last_mut(&mut self) -> Option<&mut [T]> {
        let range = self.get_last_range()?;
        self.data.get_mut(range)
    }

    pub fn clear(&mut self) {
        self.offsets_end.clear();
        self.offsets_begin.clear();
        self.data.clear();
    }
}

#[derive(Debug)]
pub struct ArrayAppender<'a, T> {
    data: &'a mut Vec<T>,
    offsets_begin: &'a mut Vec<usize>,
    offsets_end: &'a mut Vec<usize>,
    initial_count: usize,
}

impl<'a, T> ArrayAppender<'a, T> {
    pub fn push_single(&mut self, element: T) -> &mut Self {
        self.data.push(element);
        self
    }

    pub fn count(&self) -> usize {
        self.data.len() - self.initial_count
    }
}

impl<'a, T> Drop for ArrayAppender<'a, T> {
    fn drop(&mut self) {
        self.offsets_begin.push(self.initial_count);
        self.offsets_end.push(self.data.len());
    }
}

impl<T: Clone> NestedVec<T> {
    pub fn push(&mut self, array: &[T]) {
        self.offsets_begin.push(self.data.len());
        self.data.extend_from_slice(array);
        self.offsets_end.push(self.data.len());
    }
}

impl<'a, T: Clone> From<&'a Vec<Vec<T>>> for NestedVec<T> {
    fn from(nested_vec: &'a Vec<Vec<T>>) -> Self {
        let mut result = Self::new();
        for vec in nested_vec {
            result.push(vec);
        }
        result
    }
}

impl<T: Clone> From<Vec<Vec<T>>> for NestedVec<T> {
    fn from(vec_vec: Vec<Vec<T>>) -> Self {
        Self::from(&vec_vec)
    }
}

impl<'a, T: Clone> From<&'a NestedVec<T>> for Vec<Vec<T>> {
    fn from(nested: &NestedVec<T>) -> Self {
        nested.iter().map(|slice| slice.to_vec()).collect()
    }
}

impl<'a, T: Clone> From<NestedVec<T>> for Vec<Vec<T>> {
    fn from(nested: NestedVec<T>) -> Self {
        Self::from(&nested)
    }
}

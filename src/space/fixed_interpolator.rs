use std::iter::repeat;
use itertools::izip;
use nalgebra::{DefaultAllocator, DimName, DVectorView, OPoint, OVector, U1};
use fenris_traits::allocators::BiDimAllocator;
use fenris_traits::Real;
use nalgebra::allocator::Allocator;
use serde::{Deserialize, Serialize};
use crate::space::{FindClosestElement};

/// Interpolates solution variables onto a fixed set of interpolation points.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedInterpolator<T> {
    // Store the highest node index in supported_nodes, so that we can
    // guarantee that we don't go out of bounds during interpolation.
    max_node_index: Option<usize>,

    // For a set of points X_I and solution variables u, a finite element interpolation can be written
    //  u_h(X_I) = sum_J N_J(X_I) * u_J,
    // where N_J is the basis function associated with node J, u_J is the solution variable
    // associated with node J (basis weight) and u_h is the interpolation solution. Since the basis
    // functions have local support, it suffices to consider nodes J for which X_I lies in the
    // support of N_J.
    // While the above is essentially a matrix-vector multiplication, we want to work with
    // low-dimensional point and vector types. Thus we implement a CSR-like custom format
    // that lets us compactly represent the weights.

    // Offsets into the node support vector. supported_node_offsets[I] gives the
    // index of the first node that is supported on X_I. The length of support_node_offsets is
    // m + 1, where m is the number of interpolation points X_I.
    // This way the number of supported bases for a given point I is given by
    // count = supported_node_offsets[I + 1] - supported_node_offsets[I].
    supported_node_offsets: Vec<usize>,
    node_values: Vec<T>,
    node_indices: Vec<usize>,
}

impl<T> FixedInterpolator<T>
where
    T: Real,
{
    pub fn interpolate<'a, SolutionDim>(&self, u: impl Into<DVectorView<'a, T>>) -> Vec<OVector<T, SolutionDim>>
    where
        SolutionDim: DimName,
        DefaultAllocator: Allocator<T, SolutionDim, U1>,
    {
        let num_sol_vectors = self.supported_node_offsets.len().checked_sub(1).unwrap();
        let mut sol_vectors = vec![OVector::zeros(); num_sol_vectors];
        self.interpolate_into(&mut sol_vectors, u);
        sol_vectors
    }

    pub fn interpolate_into<'a, SolutionDim>(&self, result: &mut [OVector<T, SolutionDim>], u: impl Into<DVectorView<'a, T>>)
    where
        SolutionDim: DimName,
        DefaultAllocator: Allocator<T, SolutionDim, U1>,
    {
        self.interpolate_into_(result, u.into())
    }

    fn interpolate_into_<'a, SolutionDim>(&self, result: &mut [OVector<T, SolutionDim>], u: DVectorView<'a, T>)
        where
            SolutionDim: DimName,
            DefaultAllocator: Allocator<T, SolutionDim, U1>,
    {
        assert_eq!(
            result.len() + 1,
            self.supported_node_offsets.len(),
            "Number of interpolation points must match."
        );
        assert!(
            self.max_node_index.is_none() || SolutionDim::dim() * self.max_node_index.unwrap() < u.len(),
            "Cannot reference degrees of freedom not present in solution variables"
        );

        for i in 0..result.len() {
            let i_support_start = self.supported_node_offsets[i];
            let i_support_end = self.supported_node_offsets[i + 1];
            let node_values = &self.node_values[i_support_start..i_support_end];
            let node_indices = &self.node_indices[i_support_start..i_support_end];

            let mut interpolated = OVector::zeros();
            for (v, j) in izip!(node_values, node_indices) {
                let u_j = u.rows_generic(SolutionDim::dim() * j, SolutionDim::name());
                interpolated += u_j * v.clone();
            }
            result[i] = interpolated;
        }
    }
}

impl<T> FixedInterpolator<T> {
    pub fn from_compressed_values(
        node_values: Vec<T>,
        node_indices: Vec<usize>,
        supported_node_offsets: Vec<usize>
    ) -> Self {
        assert!(
            supported_node_offsets
                .iter()
                .all(|i| *i < node_values.len() + 1),
            "Supported node offsets must be in bounds with respect to supported nodes."
        );
        assert_eq!(node_values.len(), node_indices.len(),
                   "Number of node values and indices must be the same");

        Self {
            max_node_index: node_indices.iter().max().copied(),
            node_values,
            supported_node_offsets,
            node_indices,
        }
    }
}

impl<T: Real> FixedInterpolator<T> {
    /// Creates a new fixed interpolator for the given space and point set.
    ///
    /// Returns `None` if the space does not have any elements, in which case a meaningful
    /// interpolator cannot be constructed.
    pub fn from_space_and_points<Space>(space: &Space, points: &[OPoint<T, Space::GeometryDim>]) -> Self
    where
        Space: FindClosestElement<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        let mut supported_node_offsets = Vec::new();
        let mut node_values = Vec::new();
        let mut node_indices = Vec::new();

        for point in points {
            let point_node_support_begin = node_values.len();
            assert_eq!(point_node_support_begin, node_indices.len());
            supported_node_offsets.push(point_node_support_begin);

            let Some((element_idx, xi)) = space.find_closest_element_and_reference_coords(point)
                // break out of the loop if there are no elements in the space
                else { break };

            let element_node_count = space.element_node_count(element_idx);
            node_values.extend(repeat(T::zero()).take(element_node_count));
            node_indices.extend(repeat(usize::MAX).take(element_node_count));
            space.populate_element_basis(element_idx, &mut node_values[point_node_support_begin..], &xi);
            space.populate_element_nodes(&mut node_indices[point_node_support_begin..], element_idx);
        }

        supported_node_offsets.push(node_values.len());
        assert_eq!(points.len() + 1, supported_node_offsets.len());

        Self::from_compressed_values(node_values, node_indices, supported_node_offsets)
    }
}

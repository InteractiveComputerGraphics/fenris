use crate::space::{FindClosestElement, VolumetricFiniteElementSpace};
use fenris_traits::allocators::BiDimAllocator;
use fenris_traits::Real;
use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::{DVectorView, DefaultAllocator, DimName, Dyn, MatrixView, MatrixViewMut, OMatrix, OPoint, OVector, U1};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::iter::repeat;

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
    node_indices: Vec<usize>,
    node_values: Option<Vec<T>>,
    // Stored by flattening gradients in column-major ordering
    node_gradients: Option<Vec<T>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValuesOrGradients {
    #[default]
    Both,
    OnlyValues,
    OnlyGradients,
}

impl ValuesOrGradients {
    pub fn compute_values(&self) -> bool {
        use ValuesOrGradients::*;
        match self {
            Both | OnlyValues => true,
            OnlyGradients => false,
        }
    }

    pub fn compute_gradients(&self) -> bool {
        use ValuesOrGradients::*;
        match self {
            Both | OnlyGradients => true,
            OnlyValues => false,
        }
    }
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
        let num_interpolation_points = self.supported_node_offsets.len().checked_sub(1).unwrap();
        let mut interpolated_vectors = vec![OVector::zeros(); num_interpolation_points];
        self.interpolate_into(&mut interpolated_vectors, u);
        interpolated_vectors
    }

    pub fn interpolate_into<'a, SolutionDim>(
        &self,
        result: &mut [OVector<T, SolutionDim>],
        u: impl Into<DVectorView<'a, T>>,
    ) where
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
        let Some(node_values) = &self.node_values else { panic!("cannot interpolate without nodal values") };

        for i in 0..result.len() {
            let i_support_start = self.supported_node_offsets[i];
            let i_support_end = self.supported_node_offsets[i + 1];
            let node_values = &node_values[i_support_start..i_support_end];
            let node_indices = &self.node_indices[i_support_start..i_support_end];

            let mut interpolated = OVector::zeros();
            for (v, j) in izip!(node_values, node_indices) {
                let u_j = u.rows_generic(SolutionDim::dim() * j, SolutionDim::name());
                interpolated += u_j * v.clone();
            }
            result[i] = interpolated;
        }
    }

    pub fn interpolate_gradients<'a, GeometryDim, SolutionDim>(
        &self,
        u: impl Into<DVectorView<'a, T>>,
    ) -> Vec<OMatrix<T, GeometryDim, SolutionDim>>
    where
        SolutionDim: DimName,
        GeometryDim: DimName,
        DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
    {
        let num_points = self.supported_node_offsets.len().checked_sub(1).unwrap();
        let mut gradients = vec![OMatrix::<_, GeometryDim, SolutionDim>::zeros(); num_points];
        self.interpolate_gradients_into(&mut gradients, u);
        gradients
    }

    pub fn interpolate_gradients_into<'a, GeometryDim, SolutionDim>(
        &self,
        result: &mut [OMatrix<T, GeometryDim, SolutionDim>],
        u: impl Into<DVectorView<'a, T>>,
    ) where
        SolutionDim: DimName,
        GeometryDim: DimName,
        DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
    {
        self.interpolate_gradients_into_(result, u.into())
    }

    fn interpolate_gradients_into_<'a, SolutionDim, GeometryDim>(
        &self,
        gradients: &mut [OMatrix<T, GeometryDim, SolutionDim>],
        u: DVectorView<'a, T>,
    ) where
        SolutionDim: DimName,
        GeometryDim: DimName,
        DefaultAllocator: BiDimAllocator<T, GeometryDim, SolutionDim>,
    {
        assert_eq!(
            gradients.len() + 1,
            self.supported_node_offsets.len(),
            "Number of interpolation points must match."
        );
        assert!(
            self.max_node_index.is_none() || SolutionDim::dim() * self.max_node_index.unwrap() < u.len(),
            "Cannot reference degrees of freedom not present in solution variables"
        );

        let Some(node_gradients) = &self.node_gradients
            else { panic!("cannot interpolate gradients without nodal gradient values") };

        let gradient_len = GeometryDim::dim();

        for i in 0..gradients.len() {
            let idx_start = self.supported_node_offsets[i];
            let idx_end = self.supported_node_offsets[i + 1];
            let gradients_begin = gradient_len * idx_start;
            let gradients_end = gradient_len * idx_end;
            let node_gradients = MatrixView::from_slice_generic(
                &node_gradients[gradients_begin..gradients_end],
                GeometryDim::name(),
                Dyn(idx_end - idx_start),
            );
            let node_indices = &self.node_indices[idx_start..idx_end];

            let mut interpolated = OMatrix::<T, GeometryDim, SolutionDim>::zeros();
            for (grad_j, j) in izip!(node_gradients.column_iter(), node_indices) {
                let u_j = u.rows_generic(SolutionDim::dim() * j, SolutionDim::name());
                // Outer product += grad_j * u_j.transpose()
                interpolated.ger(T::one(), &grad_j, &u_j, T::one());
            }
            gradients[i] = interpolated;
        }
    }
}

impl<T> FixedInterpolator<T> {
    pub fn from_compressed_values(
        node_values: Option<Vec<T>>,
        node_gradients: Option<Vec<T>>,
        node_indices: Vec<usize>,
        supported_node_offsets: Vec<usize>,
    ) -> Self {
        assert!(
            supported_node_offsets
                .iter()
                .all(|i| *i < node_indices.len() + 1),
            "Supported node offsets must be in bounds with respect to supported nodes."
        );
        if let Some(node_values) = &node_values {
            assert_eq!(
                node_values.len(),
                node_indices.len(),
                "Number of node values and indices must be the same"
            );
        }
        if let Some(node_gradients) = &node_gradients {
            if node_indices.len() != 0 {
                assert_eq!(
                    node_gradients.len() % node_indices.len(),
                    0,
                    "Number of gradient values must be compatible with number of indices"
                );
            } else if node_gradients.len() != 0 {
                panic!("gradient data must be empty if indices are empty");
            }
        }

        Self {
            max_node_index: node_indices.iter().max().copied(),
            node_values,
            supported_node_offsets,
            node_indices,
            node_gradients,
        }
    }
}

impl<T: Real> FixedInterpolator<T> {
    /// Creates a new fixed interpolator for the given space and point set.
    ///
    /// Returns `None` if the space does not have any elements, in which case a meaningful
    /// interpolator cannot be constructed.
    pub fn from_space_and_points<Space>(
        space: &Space,
        points: &[OPoint<T, Space::GeometryDim>],
        what_to_compute: ValuesOrGradients,
    ) -> Self
    where
        // TODO: We currently have to restrict ourselves to volumetric finite element spaces
        // because we allow optionally computing gradients
        Space: FindClosestElement<T> + VolumetricFiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        let mut supported_node_offsets = Vec::new();
        let mut node_values = what_to_compute.compute_values().then(|| Vec::new());
        let mut node_gradients = what_to_compute.compute_gradients().then(|| Vec::new());
        let mut node_indices = Vec::new();

        let mut ref_gradient_buffer_flat = Vec::new();

        let d = Space::GeometryDim::dim();

        for point in points {
            let point_node_support_begin = node_indices.len();
            supported_node_offsets.push(point_node_support_begin);

            let Some((element_idx, xi)) = space.find_closest_element_and_reference_coords(point)
                // break out of the loop if there are no elements in the space
                else { break };

            let element_node_count = space.element_node_count(element_idx);
            node_indices.extend(repeat(usize::MAX).take(element_node_count));
            space.populate_element_nodes(&mut node_indices[point_node_support_begin..], element_idx);

            if let Some(node_values) = &mut node_values {
                debug_assert_eq!(point_node_support_begin, node_values.len());
                node_values.extend(repeat(T::zero()).take(element_node_count));
                space.populate_element_basis(element_idx, &mut node_values[point_node_support_begin..], &xi);
            }

            if let Some(node_gradients) = &mut node_gradients {
                let point_node_gradients_begin = node_gradients.len();
                // First store *reference* gradients in buffers
                ref_gradient_buffer_flat.resize(d * element_node_count, T::zero());
                let mut ref_gradients = MatrixViewMut::from_slice_generic(
                    &mut ref_gradient_buffer_flat,
                    Space::ReferenceDim::name(),
                    Dyn(element_node_count),
                );
                space.populate_element_gradients(element_idx, ref_gradients.as_view_mut(), &xi);
                // Next compute gradients in physical space by transforming by J^{-T}
                // and storing directly in flat gradient storage
                node_gradients.extend(repeat(T::zero()).take(d * element_node_count));
                let mut gradient_buffer = MatrixViewMut::from_slice_generic(
                    &mut node_gradients[point_node_gradients_begin..],
                    Space::GeometryDim::name(),
                    Dyn(element_node_count),
                );
                let jacobian = space.element_reference_jacobian(element_idx, &xi);
                let j_inv_t = jacobian.try_inverse().unwrap().transpose();
                gradient_buffer.gemm(T::one(), &j_inv_t, &ref_gradients, T::zero());
            }
        }

        supported_node_offsets.push(node_indices.len());
        assert_eq!(points.len() + 1, supported_node_offsets.len());

        Self::from_compressed_values(node_values, node_gradients, node_indices, supported_node_offsets)
    }

    /// Same as [`from_space_and_points`], but runs parts of the algorithm in parallel with
    /// `rayon`.
    pub fn from_space_and_points_par<Space>(
        space: &Space,
        points: &[OPoint<T, Space::GeometryDim>],
        what_to_compute: ValuesOrGradients,
    ) -> Self
    where
        // TODO: We currently have to restrict ourselves to volumetric finite element spaces
        // because we allow optionally computing gradients
        Space: FindClosestElement<T> + VolumetricFiniteElementSpace<T> + Sync,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
        OPoint<T, Space::GeometryDim>: Sync + Send,
    {
        let mut supported_node_offsets = Vec::new();
        let mut node_values = what_to_compute.compute_values().then(|| Vec::new());
        let mut node_gradients = what_to_compute.compute_gradients().then(|| Vec::new());
        let mut node_indices = Vec::new();

        let mut ref_gradient_buffer_flat = Vec::new();

        let d = Space::GeometryDim::dim();

        // This is by far the most expensive operation, hence we do this in parallel first
        let mut closest_elements_and_ref_coords = Vec::new();
        points
            .par_iter()
            .map(|point| space.find_closest_element_and_reference_coords(point))
            .collect_into_vec(&mut closest_elements_and_ref_coords);

        // The rest is sequential for now, it's a bit more cumbersome to parallelize
        // (needs some parallel prefix sum etc.)
        for closest_element_and_ref_coord in closest_elements_and_ref_coords {
            let point_node_support_begin = node_indices.len();
            supported_node_offsets.push(point_node_support_begin);

            let Some((element_idx, xi)) = closest_element_and_ref_coord
                // break out of the loop if there are no elements in the space
                else { break };

            let element_node_count = space.element_node_count(element_idx);
            node_indices.extend(repeat(usize::MAX).take(element_node_count));
            space.populate_element_nodes(&mut node_indices[point_node_support_begin..], element_idx);

            if let Some(node_values) = &mut node_values {
                debug_assert_eq!(point_node_support_begin, node_values.len());
                node_values.extend(repeat(T::zero()).take(element_node_count));
                space.populate_element_basis(element_idx, &mut node_values[point_node_support_begin..], &xi);
            }

            if let Some(node_gradients) = &mut node_gradients {
                let point_node_gradients_begin = node_gradients.len();
                // First store *reference* gradients in buffers
                ref_gradient_buffer_flat.resize(d * element_node_count, T::zero());
                let mut ref_gradients = MatrixViewMut::from_slice_generic(
                    &mut ref_gradient_buffer_flat,
                    Space::ReferenceDim::name(),
                    Dyn(element_node_count),
                );
                space.populate_element_gradients(element_idx, ref_gradients.as_view_mut(), &xi);
                // Next compute gradients in physical space by transforming by J^{-T}
                // and storing directly in flat gradient storage
                node_gradients.extend(repeat(T::zero()).take(d * element_node_count));
                let mut gradient_buffer = MatrixViewMut::from_slice_generic(
                    &mut node_gradients[point_node_gradients_begin..],
                    Space::GeometryDim::name(),
                    Dyn(element_node_count),
                );
                let jacobian = space.element_reference_jacobian(element_idx, &xi);
                let j_inv_t = jacobian.try_inverse().unwrap().transpose();
                gradient_buffer.gemm(T::one(), &j_inv_t, &ref_gradients, T::zero());
            }
        }

        supported_node_offsets.push(node_indices.len());
        assert_eq!(points.len() + 1, supported_node_offsets.len());

        Self::from_compressed_values(node_values, node_gradients, node_indices, supported_node_offsets)
    }
}

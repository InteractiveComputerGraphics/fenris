use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::space::{FindClosestElement, VolumetricFiniteElementSpace};
use crate::{Real, SmallDim};
use nalgebra::{DefaultAllocator, DVectorSlice, OMatrix, OPoint, OVector};
use davenport::{define_thread_local_workspace, with_thread_local_workspace};
use itertools::izip;
use crate::assembly::buffers::{BufferUpdate, InterpolationBuffer};

define_thread_local_workspace!(INTERPOLATE_WORKSPACE);

/// A finite element space that admits interpolation at arbitrary points.
pub trait InterpolateFiniteElementSpace<T>: VolumetricFiniteElementSpace<T>
where
    // TODO: Move these methods out of trait? Or blanket impl?
    Self: FindClosestElement<T>,
    // TODO: Ideally we should be able to use Scalar as a bound, but Scalar doesn't have
    // Default, and unfortunately e.g. OPoint<T, D> require Zero for their default
    // instead of T: Default. Should send a PR to nalgebra ...
    T: Real,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    /// Interpolate a quantity, defined by the global interpolation weights, at a set of
    /// arbitrary points.
    ///
    /// The results are stored in the provided buffer.
    ///
    /// If a point is outside the domain of the finite element space, the implementation
    /// should use the closest element to interpolate.
    ///
    /// TODO: Specify exactly what is expected here: evaluate e.g. basis functions etc.
    /// *at* the closest point or extrapolate somehow?
    ///
    /// The results are unspecified if the space has no elements.
    ///
    /// # Panics
    /// Panics if the result buffer is not of the same length as the number of points.
    fn interpolate_at_points<D: SmallDim>(
        &self,
        points: &[OPoint<T, Self::GeometryDim>],
        interpolation_weights: DVectorSlice<T>,
        result_buffer: &mut [OVector<T, D>]
    )
    where
        DefaultAllocator: DimAllocator<T, D>
    {
        assert_eq!(points.len(), result_buffer.len());
        let u = interpolation_weights;
        let d = D::dim();

        with_thread_local_workspace(&INTERPOLATE_WORKSPACE, |buf: &mut InterpolationBuffer<T>| {
            // TODO: Consider rewriting this to group together points that are mapped to the
            // same element, so that we can re-use the work to e.g. gather weights and similar
            for (point, interpolation) in izip!(points, result_buffer.iter_mut()) {
                // Finding the closest element can only "fail" (return None) if there are
                // no elements in the mesh. So if it fails for one, it must fail for all.
                let closest = self.find_closest_element_and_reference_coords(point);
                if let Some((element, ref_coords)) = closest {
                    let mut element_buf = buf.prepare_element_in_space(element, self, u, d);
                    element_buf.update_reference_point(&ref_coords, BufferUpdate::BasisValues);
                    *interpolation = element_buf.interpolate();
                } else {
                    // If we can't even find a closest element, then there are no elements in
                    // the space, in which case we've elected to return zero as the interpolated
                    // value.
                    *interpolation = OVector::<T, D>::zeros();
                }
            }
        })
    }

    fn interpolate_gradient_at_points<SolutionDim: SmallDim>(
        &self,
        points: &[OPoint<T, Self::GeometryDim>],
        interpolation_weights: DVectorSlice<T>,
        result_buffer: &mut [OMatrix<T, Self::GeometryDim, SolutionDim>]
    )
    where
        DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, SolutionDim>
            + BiDimAllocator<T, Self::ReferenceDim, SolutionDim>
    {
        assert_eq!(points.len(), result_buffer.len());
        let u = interpolation_weights;
        let d = SolutionDim::dim();

        with_thread_local_workspace(&INTERPOLATE_WORKSPACE, |buf: &mut InterpolationBuffer<T>| {
            // TODO: Consider rewriting this to group together points that are mapped to the
            // the same element
            for (point, gradient) in izip!(points, result_buffer.iter_mut()) {
                // Finding the closest element can only "fail" (return None) if there are
                // no elements in the mesh. So if it fails for one, it must fail for all.
                let closest = self.find_closest_element_and_reference_coords(point);
                if let Some((element, ref_coords)) = closest {
                    let mut element_buf = buf.prepare_element_in_space(element, self, u, d);
                    element_buf.update_reference_point(&ref_coords, BufferUpdate::BasisGradients);
                    // We need to compute the gradient with respect to physical coordinates
                    let ref_gradient: OMatrix<_, Self::ReferenceDim, SolutionDim> = element_buf.interpolate_gradient();
                    let j = element_buf.element_reference_jacobian();
                    let inv_j_t: OMatrix<_, Self::GeometryDim, Self::ReferenceDim> = j.try_inverse()
                        .expect("TODO: Fix this")
                        .transpose();
                    *gradient = inv_j_t * ref_gradient;
                } else {
                    // If we can't even find a closest element, then there are no elements in
                    // the space, in which case we've elected to return zero as the interpolated
                    // value.
                    *gradient = OMatrix::<T, Self::GeometryDim, SolutionDim>::zeros();
                }
            }
        })
    }
}

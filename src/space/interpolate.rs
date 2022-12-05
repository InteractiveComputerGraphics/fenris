use std::array;
use crate::space::{FindClosestElement, FiniteElementSpace, VolumetricFiniteElementSpace};
use crate::{Real, SmallDim};
use nalgebra::{DefaultAllocator, DVectorSlice, OMatrix, OPoint, OVector};
use davenport::{define_thread_local_workspace, with_thread_local_workspace};
use itertools::izip;
use crate::allocators::TriDimAllocator;
use crate::assembly::buffers::{BufferUpdate, InterpolationBuffer};

/// A finite element space that allows interpolation at arbitrary points.
pub trait InterpolateInSpace<T: Real, SolutionDim: SmallDim>: FiniteElementSpace<T>
where
    DefaultAllocator: TriDimAllocator<T, Self::GeometryDim, Self::ReferenceDim, SolutionDim>,
{
    /// Interpolate a quantity at a single point.
    ///
    /// Same as [`interpolate_at_points`], but provided for convenience. Generally speaking,
    /// it will be more efficient to call [`interpolate_at_points`] if you need to interpolate
    /// at more than one point.
    fn interpolate_at_point(&self,
                            point: &OPoint<T, Self::GeometryDim>,
                            interpolation_weights: DVectorSlice<T>
    ) -> OVector<T, SolutionDim> {
        let mut buffer = [OVector::<_, SolutionDim>::zeros()];
        self.interpolate_at_points(array::from_ref(point), interpolation_weights, &mut buffer);
        let [result] = buffer;
        result
    }

    /// Interpolate a quantity, defined by the global interpolation weights associated with this
    /// finite element space, at a set of arbitrary points.
    ///
    /// Specifically, for each point $\vec x_i$, compute
    /// <div>$$
    /// u_h(\vec x_i) = \sum_I u_I \, N_I(\vec x_i).
    /// $$</div>
    ///
    /// The results are stored in the provided buffer.
    ///
    /// If a point is outside the domain of the finite element space, the closest element is used to
    /// interpolate.
    ///
    /// TODO: Specify exactly what is expected here: evaluate e.g. basis functions etc.
    /// *at* the closest point or extrapolate somehow?
    ///
    /// The results are unspecified if the space has no elements.
    ///
    /// # Panics
    /// An implementation must panic if the result buffer is not of the same length as the
    /// number of points.
    ///
    /// An implementation may also panic if the length of the interpolation weights vector
    /// is not equal to $s n$, where $s$ is the solution dimension and $n$ is the number of
    /// nodes/vertices in the space.
    fn interpolate_at_points(
        &self,
        points: &[OPoint<T, Self::GeometryDim>],
        interpolation_weights: DVectorSlice<T>,
        result_buffer: &mut [OVector<T, SolutionDim>]
    );
}

/// A volumetric finite element space that allows interpolation of gradients at arbitrary points.
pub trait InterpolateGradientInSpace<T: Real, SolutionDim: SmallDim>: VolumetricFiniteElementSpace<T>
where
    DefaultAllocator: TriDimAllocator<T, Self::GeometryDim, Self::ReferenceDim, SolutionDim>,
{
    /// Interpolate the gradient of a quantity at a single point.
    ///
    /// Same as [`interpolate_gradient_at_points`], but provided for convenience. Generally speaking,
    /// it will be more efficient to call [`interpolate_gradient_at_points`] if you need to interpolate
    /// at more than one point.
    fn interpolate_gradient_at_point(
        &self,
        point: &OPoint<T, Self::GeometryDim>,
        interpolation_weights: DVectorSlice<T>,
    ) -> OMatrix<T, Self::GeometryDim, SolutionDim> {
        let mut buffer = [OMatrix::<_, Self::GeometryDim, SolutionDim>::zeros()];
        self.interpolate_gradient_at_points(array::from_ref(point), interpolation_weights, &mut buffer);
        let [result] = buffer;
        result
    }

    /// Interpolate the gradient of a quantity, defined by the global interpolation weights
    /// associated with this finite element space, at a set of arbitrary points.
    ///
    /// Specifically, for each point $\vec x_i$, compute
    /// <div>$$
    /// \nabla u_h(\vec x_i) = \sum_I \nabla N_I(\vec x_i) \otimes u_I.
    /// $$</div>
    ///
    /// The results are stored in the provided buffer.
    ///
    /// If a point is outside the domain of the finite element space, the closest element is used to
    /// interpolate.
    ///
    /// TODO: Specify exactly what is expected here: evaluate e.g. basis functions etc.
    /// *at* the closest point or extrapolate somehow?
    ///
    /// The results are unspecified if the space has no elements.
    ///
    /// # Panics
    /// An implementation must panic if the result buffer is not of the same length as the
    /// number of points.
    ///
    /// An implementation may also panic if the length of the interpolation weights vector
    /// is not equal to $s n$, where $s$ is the solution dimension and $n$ is the number of
    /// nodes/vertices in the space.
    fn interpolate_gradient_at_points(
        &self,
        points: &[OPoint<T, Self::GeometryDim>],
        interpolation_weights: DVectorSlice<T>,
        result_buffer: &mut [OMatrix<T, Self::GeometryDim, SolutionDim>]
    );
}

define_thread_local_workspace!(INTERPOLATE_WORKSPACE);

/// Interpolate a quantity, defined by the global interpolation weights associated with the
/// given finite element space, at a set of arbitrary points.
///
/// Specifically, for each point $\vec x_i$, compute
/// <div>$$
/// u_h(\vec x_i) = \sum_I u_I \, N_I(\vec x_i).
/// $$</div>
///
/// The results are stored in the provided buffer.
///
/// If a point is outside the domain of the finite element space, the closest element is used to
/// interpolate.
///
/// TODO: Specify exactly what is expected here: evaluate e.g. basis functions etc.
/// *at* the closest point or extrapolate somehow?
///
/// The results are unspecified if the space has no elements.
///
/// # Panics
/// Panics if the result buffer is not of the same length as the number of points.
pub fn interpolate_at_points<T, SolutionDim, Space>(
    space: &Space,
    points: &[OPoint<T, Space::GeometryDim>],
    interpolation_weights: DVectorSlice<T>,
    result_buffer: &mut [OVector<T, SolutionDim>]
)
where
    T: Real,
    SolutionDim: SmallDim,
    Space: FindClosestElement<T>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>
{
    assert_eq!(points.len(), result_buffer.len());
    let u = interpolation_weights;
    let d = SolutionDim::dim();
    with_thread_local_workspace(&INTERPOLATE_WORKSPACE, |buf: &mut InterpolationBuffer<T>| {
        // TODO: Consider rewriting this to group together points that are mapped to the
        // same element, so that we can re-use the work to e.g. gather weights and similar
        for (point, interpolation) in izip!(points, result_buffer.iter_mut()) {
            let closest = space.find_closest_element_and_reference_coords(point);
            if let Some((element, ref_coords)) = closest {
                let mut element_buf = buf.prepare_element_in_space(element, space, u, d);
                element_buf.update_reference_point(&ref_coords, BufferUpdate::BasisValues);
                *interpolation = element_buf.interpolate();
            } else {
                // If we can't even find a closest element, then there are no elements in
                // the space, in which case we've elected to return zero as the interpolated
                // value.
                *interpolation = OVector::<T, SolutionDim>::zeros();
            }
        }
    })
}

/// Interpolate the gradient of a quantity, defined by the global interpolation weights
/// associated with the given finite element space, at a set of arbitrary points.
///
/// Specifically, for each point $\vec x_i$, compute
/// <div>$$
/// \nabla u_h(\vec x_i) = \sum_I \nabla N_I(\vec x_i) \otimes u_I.
/// $$</div>
///
/// The results are stored in the provided buffer.
///
/// If a point is outside the domain of the finite element space, the closest element is used to
/// interpolate.
///
/// TODO: Specify exactly what is expected here: evaluate e.g. basis functions etc.
/// *at* the closest point or extrapolate somehow?
///
/// The results are unspecified if the space has no elements.
///
/// # Panics
/// Panics if the result buffer is not of the same length as the number of points.
pub fn interpolate_gradient_at_points<T, SolutionDim, Space>(
    space: &Space,
    points: &[OPoint<T, Space::GeometryDim>],
    interpolation_weights: DVectorSlice<T>,
    result_buffer: &mut [OMatrix<T, Space::GeometryDim, SolutionDim>]
)
where
    T: Real,
    SolutionDim: SmallDim,
    Space: FindClosestElement<T> + VolumetricFiniteElementSpace<T>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>
{
    assert_eq!(points.len(), result_buffer.len());
    let u = interpolation_weights;
    let d = SolutionDim::dim();

    with_thread_local_workspace(&INTERPOLATE_WORKSPACE, |buf: &mut InterpolationBuffer<T>| {
        // TODO: Consider rewriting this to group together points that are mapped to the
        // the same element
        for (point, gradient) in izip!(points, result_buffer.iter_mut()) {
            let closest = space.find_closest_element_and_reference_coords(point);
            if let Some((element, ref_coords)) = closest {
                let mut element_buf = buf.prepare_element_in_space(element, space, u, d);
                element_buf.update_reference_point(&ref_coords, BufferUpdate::BasisGradients);
                // We need to compute the gradient with respect to physical coordinates,
                // so need to transform it by inverse transpose Jacobian matrix
                let ref_gradient = element_buf.interpolate_gradient();
                let j = element_buf.element_reference_jacobian();
                let inv_j_t = j.try_inverse()
                    .expect("TODO: Fix this")
                    .transpose();
                *gradient = inv_j_t * ref_gradient;
            } else {
                // If we can't even find a closest element, then there are no elements in
                // the space, in which case we've elected to return zero as the interpolated
                // value.
                *gradient = OMatrix::<T, Space::GeometryDim, SolutionDim>::zeros();
            }
        }
    })
}

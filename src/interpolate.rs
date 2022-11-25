use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::space::{FiniteElementConnectivity, FiniteElementSpace, GeometricFiniteElementSpace, VolumetricFiniteElementSpace};
use crate::{Real, SmallDim};
use nalgebra::{DefaultAllocator, DimName, DVectorSlice, Dynamic, MatrixSliceMut, OMatrix, OPoint, OVector, Scalar};
use std::marker::PhantomData;
use davenport::{define_thread_local_workspace, with_thread_local_workspace};
use itertools::izip;
use nalgebra::allocator::Allocator;
use rstar::{AABB, Envelope, PointDistance, RTree, RTreeObject};
use rstar::primitives::{GeomWithData};
use fenris_geometry::{AxisAlignedBoundingBox, BoundedGeometry, GeometryCollection};
use crate::assembly::buffers::{BufferUpdate, InterpolationBuffer};

pub enum ClosestPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    InElement(OPoint<T, D>),
    ClosestPoint(OPoint<T, D>),
}

define_thread_local_workspace!(INTERPOLATE_WORKSPACE);

/// A finite element space that admits interpolation at arbitrary points.
pub trait InterpolateFiniteElementSpace<T>: VolumetricFiniteElementSpace<T>
where
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

    fn closest_point_on_element(&self, element_index: usize, p: &OPoint<T, Self::GeometryDim>)
        -> ClosestPoint<T, Self::ReferenceDim>;

    /// Find the closest point on the mesh to the given point, represented as the
    /// index of the closest element and the coordinates in the reference element.
    fn find_closest_element_and_reference_coords(
        &self,
        point: &OPoint<T, Self::GeometryDim>,
    ) -> Option<(usize, OPoint<T, Self::ReferenceDim>)>;
}

struct RTreeAccelerationStructure<D: DimName>
where
    DefaultAllocator: Allocator<f64, D>,
{
    tree: RTree<GeomWithData<RTreeAABB<D>, usize>>,
}


#[derive(Debug, Clone, PartialEq)]
struct RTreePoint<D>(pub OPoint<f64, D>)
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>;

impl<D> rstar::Point for RTreePoint<D>
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>
{
    type Scalar = f64;
    const DIMENSIONS: usize = D::USIZE;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self(OVector::<f64, D>::from_fn(|i, _| generator(i)).into())
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        self.0[index]
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        &mut self.0[index]
    }
}

impl<D: DimName> RTreeObject for RTreeAABB<D>
where
    DefaultAllocator: Allocator<f64, D>
{
    type Envelope = AABB<RTreePoint<D>>;

    fn envelope(&self) -> Self::Envelope {
        let Self(aabb) = self;
        let box_min = aabb.min().clone();
        let box_max = aabb.max().clone();
        AABB::from_corners(RTreePoint(box_min), RTreePoint(box_max))
    }
}

impl<D: DimName> PointDistance for RTreeAABB<D>
where
    DefaultAllocator: Allocator<f64, D>
{
    fn distance_2(&self, point: &RTreePoint<D>) -> <<Self::Envelope as Envelope>::Point as rstar::Point>::Scalar {
        self.0.dist2_to(&point.0)
    }

    fn contains_point(&self, point: &<Self::Envelope as Envelope>::Point) -> bool {
        self.0.contains_point(&point.0)
    }
}

struct RTreeAABB<D: DimName>(pub AxisAlignedBoundingBox<f64, D>)
where
    DefaultAllocator: Allocator<f64, D>;

impl<D: DimName> RTreeAccelerationStructure<D>
where
    DefaultAllocator: Allocator<f64, D>
{
    /// This is intended to be called with already known matching dimensions
    pub fn from_bounding_boxes<T: Real>(boxes: &[AxisAlignedBoundingBox<T, D>]) -> Self
    where
        DefaultAllocator: DimAllocator<T, D>
    {
        let geometries = boxes.iter()
            .enumerate()
            .map(|(i, bounding_box)| {
                // Make bounding box larger than necessary to accommodate
                // possible floating point errors etc.
                let bounding_box = bounding_box.uniformly_scale(T::from_f64(1.01).unwrap());
                let box_min = bounding_box.min().coords.map(|x_i| x_i.to_subset().unwrap());
                let box_max = bounding_box.max().coords.map(|x_i| x_i.to_subset().unwrap());
                let box_f64 = AxisAlignedBoundingBox::new(box_min.into(), box_max.into());
                GeomWithData::new(RTreeAABB(box_f64), i)
            }).collect();
        let tree = RTree::bulk_load(geometries);
        Self { tree }
    }

    pub fn closest_cell_candidates<'a, T: Real>(&'a self, point: &OPoint<T, D>) -> impl 'a + Iterator<Item=usize>
    where
        DefaultAllocator: DimAllocator<T, D>
    {
        let point_f64: OPoint<f64, D> = point.map(|x_i| x_i.to_subset().expect("TODO"));
        let mut iter = self.tree.nearest_neighbor_iter(&RTreePoint(point_f64.clone()))
            .map(|geom| (&geom.geom().0, geom.data))
            .peekable();

        // First find the maximum possible distance to any point in the first AABB
        let d2_max = iter.peek()
            .map(|(aabb, _)| aabb.max_dist2_to(&point_f64))
            .unwrap_or(f64::NAN);
        iter
            // Any subsequent AABB can be excluded if its closest point is larger
            // than the maximum possible distance to any point in the first AABB
            .take_while(move |&(aabb, _)| aabb.dist2_to(&point_f64) <= d2_max)
            .map(|(_, index)| index)
    }
}

pub struct Interpolator<T, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    space: Space,
    tree: RTreeAccelerationStructure<Space::GeometryDim>,
    marker: PhantomData<T>,
}

impl<T, Space> Interpolator<T, Space>
where
    T: Real,
    for<'a> Space: GeometricFiniteElementSpace<'a, T>,
    for<'a> <Space as GeometryCollection<'a>>::Geometry: BoundedGeometry<T, Dimension=Space::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    pub fn from_space(space: Space) -> Self {
        let bounding_boxes: Vec<_> = (0 .. space.num_geometries())
            .map(|i| space.get_geometry(i).unwrap().bounding_box())
            .collect();
        let rtree = RTreeAccelerationStructure::from_bounding_boxes(&bounding_boxes);
        Self {
            space,
            tree: rtree,
            marker: Default::default()
        }
    }
}

impl<T, Space> FiniteElementConnectivity for Interpolator<T, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, nodes: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(nodes, element_index)
    }
}

impl<T, Space> FiniteElementSpace<T> for Interpolator<T, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    type GeometryDim = Space::GeometryDim;
    type ReferenceDim = Space::ReferenceDim;

    fn populate_element_basis(&self, element_index: usize, basis_values: &mut [T], reference_coords: &OPoint<T, Self::ReferenceDim>) {
        self.space.populate_element_basis(element_index, basis_values, reference_coords)
    }

    fn populate_element_gradients(&self, element_index: usize, gradients: MatrixSliceMut<T, Self::ReferenceDim, Dynamic>, reference_coords: &OPoint<T, Self::ReferenceDim>) {
        self.space.populate_element_gradients(element_index, gradients, reference_coords)
    }

    fn element_reference_jacobian(&self, element_index: usize, reference_coords: &OPoint<T, Self::ReferenceDim>) -> OMatrix<T, Self::GeometryDim, Self::ReferenceDim> {
        self.space.element_reference_jacobian(element_index, reference_coords)
    }

    fn map_element_reference_coords(&self, element_index: usize, reference_coords: &OPoint<T, Self::ReferenceDim>) -> OPoint<T, Self::GeometryDim> {
        self.space.map_element_reference_coords(element_index, reference_coords)
    }

    fn diameter(&self, element_index: usize) -> T {
        self.space.diameter(element_index)
    }
}

impl<T, Space> InterpolateFiniteElementSpace<T> for Interpolator<T, Space>
where
    T: Real,
    Space: InterpolateFiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    fn closest_point_on_element(&self,
                                element_index: usize,
                                p: &OPoint<T, Self::GeometryDim>) -> ClosestPoint<T, Self::ReferenceDim> {
        self.space.closest_point_on_element(element_index, p)
    }

    fn find_closest_element_and_reference_coords(&self, point: &OPoint<T, Self::GeometryDim>) -> Option<(usize, OPoint<T, Self::ReferenceDim>)> {
        let mut min_dist2 = None;
        let mut closest_result = None;
        for candidate_element_idx in self.tree.closest_cell_candidates(point) {
            match self.space.closest_point_on_element(candidate_element_idx, point) {
                // Pick the first element that reports that the point is contained in the element
                ClosestPoint::InElement(ref_coords) => return Some((candidate_element_idx, ref_coords)),
                ClosestPoint::ClosestPoint(ref_coords) => {
                    let x = self.space.map_element_reference_coords(candidate_element_idx, &ref_coords);
                    let dist2 = (x - point).norm_squared();

                    let is_min = min_dist2.map(|d2| d2 <= dist2).unwrap_or(true);
                    if is_min {
                        min_dist2 = Some(dist2);
                        closest_result = Some((candidate_element_idx, ref_coords));
                    }
                }
            }
        }
        closest_result
    }
}

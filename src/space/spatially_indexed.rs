use crate::space::{BoundsForElementInSpace, ClosestPointInElementInSpace, FindClosestElement, FiniteElementConnectivity, FiniteElementSpace, interpolate_at_points, interpolate_gradient_at_points, InterpolateGradientInSpace, InterpolateInSpace, VolumetricFiniteElementSpace};
use nalgebra::{DefaultAllocator, DimName, DVectorSlice, Dynamic, MatrixSliceMut, OMatrix, OPoint, OVector, Scalar};
use rstar::{AABB, Envelope, PointDistance, RTree, RTreeObject};
use nalgebra::allocator::Allocator;
use fenris_geometry::AxisAlignedBoundingBox;
use fenris_traits::allocators::{BiDimAllocator, DimAllocator, TriDimAllocator};
use fenris_traits::Real;
use rstar::primitives::GeomWithData;
use std::marker::PhantomData;
use crate::element::ClosestPoint;
use crate::SmallDim;

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

/// Provides accelerated geometry queries for a
/// [finite element space](crate::space::FiniteElementSpace).
///
/// Specifically, given a space that implements [`BoundsForElementInSpace`] and [`ClosestPointInElementInSpace`],
/// `SpatiallyIndexed` wraps the space and provides an implementation of
/// [`FindClosestElement`] on top.
///
/// In addition, `SpatiallyIndexed` provides interpolation of arbitrary points by implementing
/// the [`InterpolateInSpace`] and [`InterpolateGradientInSpace`] finite element space
/// traits.
pub struct SpatiallyIndexed<T, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    space: Space,
    tree: RTreeAccelerationStructure<Space::GeometryDim>,
    marker: PhantomData<T>,
}

impl<T, Space> SpatiallyIndexed<T, Space>
where
    T: Real,
    Space: BoundsForElementInSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    pub fn from_space(space: Space) -> Self {
        let bounding_boxes = space.bounds_for_all_elements();
        let rtree = RTreeAccelerationStructure::from_bounding_boxes(&bounding_boxes);
        Self {
            space,
            tree: rtree,
            marker: Default::default()
        }
    }
}

impl<T, Space> SpatiallyIndexed<T, Space>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
{
    pub fn space(&self) -> &Space {
        &self.space
    }
}

impl<T, Space> FiniteElementConnectivity for SpatiallyIndexed<T, Space>
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

impl<T, Space> FiniteElementSpace<T> for SpatiallyIndexed<T, Space>
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

impl<T, Space> ClosestPointInElementInSpace<T> for SpatiallyIndexed<T, Space>
where
    T: Real,
    Space: ClosestPointInElementInSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    fn closest_point_in_element(&self, element_index: usize, p: &OPoint<T, Self::GeometryDim>) -> ClosestPoint<T, Self::ReferenceDim> {
        self.space.closest_point_in_element(element_index, p)
    }
}

impl<T, Space> BoundsForElementInSpace<T> for SpatiallyIndexed<T, Space>
where
    T: Real,
    Space: BoundsForElementInSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    fn bounds_for_element(&self, element_index: usize) -> AxisAlignedBoundingBox<T, Self::GeometryDim> {
        self.space.bounds_for_element(element_index)
    }
}

impl<T, Space> FindClosestElement<T> for SpatiallyIndexed<T, Space>
where
    T: Real,
    Space: ClosestPointInElementInSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    fn find_closest_element_and_reference_coords(&self, point: &OPoint<T, Self::GeometryDim>) -> Option<(usize, OPoint<T, Self::ReferenceDim>)> {
        let mut min_dist2 = None;
        let mut closest_result = None;
        // TODO: This is inefficient because we could have pruned way more candidates
        // if we used the *actual* current minimum distance to the element
        // to prune further bounding boxes. This suggests that this routine
        // needs to be merged with the implementation of closest_cell_candidates so that
        // we have all the information at hand
        for candidate_element_idx in self.tree.closest_cell_candidates(point) {
            match self.space.closest_point_in_element(candidate_element_idx, point) {
                // Pick the first element that reports that the point is contained in the element
                ClosestPoint::InElement(ref_coords) => return Some((candidate_element_idx, ref_coords)),
                ClosestPoint::ClosestPoint(ref_coords) => {
                    let x = self.space.map_element_reference_coords(candidate_element_idx, &ref_coords);
                    let dist2 = (x - point).norm_squared();

                    let is_min = min_dist2.map(|d2| dist2 <= d2).unwrap_or(true);
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

impl<T, Space, SolutionDim> InterpolateInSpace<T, SolutionDim> for SpatiallyIndexed<T, Space>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: FiniteElementSpace<T> + BoundsForElementInSpace<T> + ClosestPointInElementInSpace<T>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>,
{
    fn interpolate_at_points(&self,
                             points: &[OPoint<T, Self::GeometryDim>],
                             interpolation_weights: DVectorSlice<T>,
                             result_buffer: &mut [OVector<T, SolutionDim>]
    ) {
        interpolate_at_points(self, points, interpolation_weights, result_buffer)
    }
}

impl<T, Space, SolutionDim> InterpolateGradientInSpace<T, SolutionDim> for SpatiallyIndexed<T, Space>
where
    T: Real,
    SolutionDim: SmallDim,
    Space: VolumetricFiniteElementSpace<T> + BoundsForElementInSpace<T> + ClosestPointInElementInSpace<T>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, SolutionDim>,
{
    fn interpolate_gradient_at_points(&self,
                                      points: &[OPoint<T, Self::GeometryDim>],
                                      interpolation_weights: DVectorSlice<T>,
                                      result_buffer: &mut [OMatrix<T, Self::GeometryDim, SolutionDim>]
    ) {
        interpolate_gradient_at_points(self, points, interpolation_weights, result_buffer)
    }
}

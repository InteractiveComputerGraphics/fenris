use crate::allocators::{BiDimAllocator, DimAllocator};
use crate::space::{FiniteElementConnectivity, FiniteElementSpace, GeometricFiniteElementSpace};
use crate::{Real, SmallDim};
use nalgebra::{Const, DefaultAllocator, DimName, Dynamic, MatrixSliceMut, OMatrix, OPoint, OVector, Point, Point2, Point3, Scalar};
use std::array;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::transmute;
use std::ops::Deref;
use davenport::Workspace;
use itertools::izip;
use nalgebra::allocator::Allocator;
use rstar::{AABB, RTree, RTreeObject};
use rstar::primitives::{GeomWithData, Rectangle};
use fenris_geometry::{AxisAlignedBoundingBox, BoundedGeometry, DistanceQuery, GeometryCollection};
use crate::util::{try_transmute_ref, try_transmute_slice};

pub enum ClosestPoint<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    InElement(OPoint<T, D>),
    ClosestPoint(OPoint<T, D>),
}

pub trait InterpolateFiniteElementSpace<T>: FiniteElementSpace<T>
where
    // TODO: Ideally we should be able to use Scalar as a bound, but Scalar doesn't have
    // Default, and unfortunately e.g. OPoint<T, D> require Zero for their default
    // instead of T: Default. Should send a PR to nalgebra ...
    T: Real,
    DefaultAllocator: BiDimAllocator<T, Self::GeometryDim, Self::ReferenceDim>,
{
    // fn interpolate(&self, point: &OPoint<T, Self::GeometryDim>, weights: DVectorSlice<T>) -> OVector<>{
    //     let (element, coords) = self.find_closest_element_and_reference_coords(point);
    //     self.populate_element_basis(element, &mut [])
    // }
    //
    // fn interpolate_gradient(&self, point: &OPoint<T, Self::GeometryDim>, weights: DVectorSlice<T>)
    fn closest_point_on_element(&self, element_index: usize, p: &OPoint<T, Self::GeometryDim>)
        -> ClosestPoint<T, Self::ReferenceDim>;

    /// Find the closest point on the mesh to the given point, represented as the
    /// index of the closest element and the coordinates in the reference element.
    fn find_closest_element_and_reference_coords(
        &self,
        point: &OPoint<T, Self::GeometryDim>,
    ) -> (usize, OPoint<T, Self::ReferenceDim>) {
        let mut result = [(usize::MAX, OPoint::default()); 1];
        self.populate_closest_element_and_reference_coords(array::from_ref(point), &mut result);
        let [result] = result;
        result
    }

    /// Same as [`find_closest_element_and_reference_coords`], but applied to several
    /// points at the same time.
    ///
    /// # Panics
    ///
    /// The method should panic if the input point slice and the output slice
    /// do not have the same length.
    fn populate_closest_element_and_reference_coords(
        &self,
        points: &[OPoint<T, Self::GeometryDim>],
        result: &mut [(usize, OPoint<T, Self::ReferenceDim>)],
    );
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

    fn generate(generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        todo!()
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        todo!()
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        todo!()
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

    pub fn closest_element_candidates<T: Real>(&self, point: &OPoint<T, D>) -> impl Iterator<Item=usize>
    where
        DefaultAllocator: DimAllocator<T, D>
    {
        // let mut iter = self.tree.nearest_neighbor_iter(point);
        // std::iter::from_fn(|| {
        //     if let Some(item) = iter.next() {
        //         Some(item.data)
        //     } else {
        //         None
        //     }
        // })

        std::iter::empty()


        // let mut iter = self.tree.nearest_neighbor_iter(point);
        // if let Some(first) = iter.next() {
        //     let index = first.data;
        //     let rectangle = first.geom();
        //     let max_dist = rectangle.max_dist_squared_to_point(&)
        // }
        // if let Some(first) = iter.peek() {
        //     let
        // }
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
    // workspace: RefCell<Workspace>,
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
        // workspace.try_insert(rtree);
        // match Space::GeometryDim::dim() {
        //     // TODO: Support dimension 1, probably need to send a PR to rstar for this
        //     2 => {
        //         // TODO: Implement a try_insert method on davenport::Workspace?
        //         workspace.get_or_insert_with(|| RTreeAccelerationStructure::<2>::from_bounding_boxes(&bounding_boxes));
        //     },
        //     3 => {
        //         workspace.get_or_insert_with(|| RTreeAccelerationStructure::<3>::from_bounding_boxes(&bounding_boxes));
        //     },
        //     _ => panic!("Unsupported dimension. Currently we only support dimension 2 and 3")
        // }

        Self {
            space,
            // workspace: RefCell::new(workspace),
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
    // for<'a> Space: GeometricFiniteElementSpace<'a, T>,
    // for<'a> <Space as GeometryCollection<'a>>::Geometry: BoundedGeometry<T, Dimension=Space::GeometryDim>,
    Space: InterpolateFiniteElementSpace<T>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>
{
    fn closest_point_on_element(&self,
                                element_index: usize,
                                p: &OPoint<T, Self::GeometryDim>) -> ClosestPoint<T, Self::ReferenceDim> {
        todo!()
    }

    fn populate_closest_element_and_reference_coords(
        &self,
        points: &[OPoint<T, Self::GeometryDim>],
        result: &mut [(usize, OPoint<T, Self::ReferenceDim>)]
    ) {
        assert_eq!(points.len(), result.len());
        for (query_point, (closest_element_idx, ref_coords)) in izip!(points, result) {
            for candidate_element_idx in self.tree.closest_element_candidates(query_point) {

            }
            // TODO: Instead of first collecting all candidates, it *may* be prudent
            // to actually directly query element closest points, since each
            // iteration through rstar's closest points may incur some manner of tree traversal.
            // We currently collect element indices first, because this is the only
            // "dimension-dependent" piece of code right now.
            // TODO: Get rid of this mess. The matching is an unfortunate necessity because
            // we have to use typenum on the nalgebra side, but rtree requires us to
            // give a constant integer for the dimension. These concepts are unfortunately not
            // compatible, in the sense that there appears to be no way to convert
            //
            // match Space::GeometryDim::dim() {
            //     2 => {
            //         let tree: &RTreeAccelerationStructure<2> = workspace.try_get().unwrap();
            //         let point: &Point2<T> = try_transmute_ref(query_point).unwrap();
            //         let rtree_point: [f64; 2] = point.coords.map(|x_i| x_i.to_subset().unwrap())
            //             .into();
            //         elements_buffer.extend(tree.closest_element_candidates(&rtree_point));
            //     },
            //     3 => {
            //         let tree: &RTreeAccelerationStructure<3> = workspace.try_get().unwrap();
            //         let point: &Point3<T> = try_transmute_ref(query_point).unwrap();
            //         let rtree_point: [f64; 3] = point.coords.map(|x_i| x_i.to_subset().unwrap())
            //             .into();
            //         elements_buffer.extend(tree.closest_element_candidates(&rtree_point));
            //     }
            //     _ => todo!("Make this work for other dims? Especially 1D...")
            // }
        }

        todo!()
    }
}

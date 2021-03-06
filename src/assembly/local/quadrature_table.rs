use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{DefaultAllocator, DimName, OPoint, Scalar};
use crate::quadrature::QuadraturePair;
use crate::util::NestedVec;
use crate::SmallDim;
use itertools::izip;
use serde::{Deserialize, Serialize};

/// Lookup table mapping elements to quadrature rules.
pub trait QuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    type Data: Default + Clone;

    fn element_quadrature_size(&self, element_index: usize) -> usize;

    fn populate_element_data(&self, element_index: usize, data: &mut [Self::Data]);

    fn populate_element_quadrature(
        &self,
        element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    );

    fn populate_element_quadrature_and_data(
        &self,
        element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
        data: &mut [Self::Data],
    ) {
        self.populate_element_quadrature(element_index, points, weights);
        self.populate_element_data(element_index, data);
    }
}

/// A quadrature table that keeps a separate quadrature rule per element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneralQuadratureTable<T, GeometryDim, Data = ()>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    #[serde(bound(serialize = "OPoint<T, GeometryDim>: Serialize"))]
    #[serde(bound(deserialize = "OPoint<T, GeometryDim>: Deserialize<'de>"))]
    points: NestedVec<OPoint<T, GeometryDim>>,
    weights: NestedVec<T>,
    data: NestedVec<Data>,
}

impl<T, GeometryDim> GeneralQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_and_weights(points: NestedVec<OPoint<T, GeometryDim>>, weights: NestedVec<T>) -> Self {
        let mut data = NestedVec::new();
        for i in 0..points.len() {
            data.push(&vec![(); points.get(i).unwrap().len()]);
        }
        Self::from_points_weights_and_data(points, weights, data)
    }
}

impl<T, GeometryDim, Data> GeneralQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_weights_and_data(
        points: NestedVec<OPoint<T, GeometryDim>>,
        weights: NestedVec<T>,
        data: NestedVec<Data>,
    ) -> Self {
        assert_eq!(points.len(), weights.len());
        assert_eq!(points.len(), data.len());

        // Ensure that each element has a consistent quadrature rule
        let iter = izip!(points.iter(), weights.iter(), data.iter());
        for (element_index, (element_points, element_weights, element_data)) in iter.enumerate() {
            assert_eq!(
                element_points.len(),
                element_weights.len(),
                "Element {} has mismatched number of points and weights.",
                element_index
            );
            assert_eq!(
                element_points.len(),
                element_data.len(),
                "Element {} has mismatched number of points and data.",
                element_index
            );
        }

        Self { points, weights, data }
    }

    pub fn into_parts(self) -> GeneralQuadratureParts<T, GeometryDim, Data> {
        GeneralQuadratureParts {
            points: self.points,
            weights: self.weights,
            data: self.data,
        }
    }
}

pub struct GeneralQuadratureParts<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub points: NestedVec<OPoint<T, GeometryDim>>,
    pub weights: NestedVec<T>,
    pub data: NestedVec<Data>,
}

impl<T, GeometryDim, Data> QuadratureTable<T, GeometryDim> for GeneralQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: SmallDim,
    Data: Clone + Default,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    type Data = Data;

    fn element_quadrature_size(&self, element_index: usize) -> usize {
        // TODO: Should we rather return results from all these methods? It seems that currently
        // we are just panicking if the quadrature table size doesn't match the number of elements
        // in the finite element space. This seems bad.
        self.weights
            .get(element_index)
            .expect("Element index out of bounds")
            .len()
    }

    fn populate_element_data(&self, element_index: usize, data: &mut [Self::Data]) {
        let data_for_element = self
            .data
            .get(element_index)
            .expect("Element index out of bounds");
        assert_eq!(data_for_element.len(), data.len());
        data.clone_from_slice(&data_for_element);
    }

    fn populate_element_quadrature(
        &self,
        element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    ) {
        let points_for_element = self
            .points
            .get(element_index)
            .expect("Element index out of bounds");
        let weights_for_element = self
            .weights
            .get(element_index)
            .expect("Element index out of bounds");
        assert_eq!(points_for_element.len(), points.len());
        assert_eq!(weights_for_element.len(), weights.len());
        points.clone_from_slice(&points_for_element);
        weights.clone_from_slice(&weights_for_element);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniformQuadratureTable<T, GeometryDim, Data = ()>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    #[serde(bound(serialize = "OPoint<T, GeometryDim>: Serialize"))]
    #[serde(bound(deserialize = "OPoint<T, GeometryDim>: Deserialize<'de>"))]
    points: Vec<OPoint<T, GeometryDim>>,
    weights: Vec<T>,
    data: Vec<Data>,
}

impl<T, GeometryDim> UniformQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_and_weights(points: Vec<OPoint<T, GeometryDim>>, weights: Vec<T>) -> Self {
        let data = vec![(); points.len()];
        Self::from_points_weights_and_data(points, weights, data)
    }

    pub fn from_quadrature(quadrature: QuadraturePair<T, GeometryDim>) -> Self {
        Self::from_quadrature_and_uniform_data(quadrature, ())
    }
}

impl<T, GeometryDim, Data> UniformQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_weights_and_data(points: Vec<OPoint<T, GeometryDim>>, weights: Vec<T>, data: Vec<Data>) -> Self {
        let msg = "Points, weights and data must have the same length.";
        assert_eq!(points.len(), weights.len(), "{}", msg);
        assert_eq!(points.len(), data.len(), "{}", msg);
        Self { points, weights, data }
    }

    pub fn from_quadrature_and_uniform_data(quadrature: QuadraturePair<T, GeometryDim>, data: Data) -> Self
    where
        Data: Clone,
    {
        let (weights, points) = quadrature;
        let data = vec![data; weights.len()];
        Self::from_points_weights_and_data(points, weights, data)
    }

    pub fn with_uniform_data<Data2: Clone>(self, data: Data2) -> UniformQuadratureTable<T, GeometryDim, Data2> {
        UniformQuadratureTable::from_quadrature_and_uniform_data((self.weights, self.points), data)
    }
}

impl<T, GeometryDim, Data> QuadratureTable<T, GeometryDim> for UniformQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: SmallDim,
    Data: Clone + Default,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    type Data = Data;

    fn element_quadrature_size(&self, _element_index: usize) -> usize {
        self.points.len()
    }

    fn populate_element_data(&self, _element_index: usize, data: &mut [Self::Data]) {
        assert_eq!(data.len(), self.data.len());
        data.clone_from_slice(&self.data);
    }

    fn populate_element_quadrature(
        &self,
        _element_index: usize,
        points: &mut [OPoint<T, GeometryDim>],
        weights: &mut [T],
    ) {
        assert_eq!(points.len(), self.points.len());
        assert_eq!(weights.len(), self.weights.len());
        points.clone_from_slice(&self.points);
        weights.clone_from_slice(&self.weights);
    }
}

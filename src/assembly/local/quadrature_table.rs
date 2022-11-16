use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{DefaultAllocator, DimName, OPoint, Scalar};
use crate::quadrature::QuadraturePair;
use crate::util::NestedVec;
use crate::SmallDim;
use itertools::izip;
use nalgebra::{U1, U2, U3};
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

/// Trait alias for a one-dimensional quadrature table.
pub trait QuadratureTable1d<T: Scalar>: QuadratureTable<T, U1> {}

/// Trait alias for a two-dimensional quadrature table.
pub trait QuadratureTable2d<T: Scalar>: QuadratureTable<T, U2> {}

/// Trait alias for a three-dimensional quadrature table.
pub trait QuadratureTable3d<T: Scalar>: QuadratureTable<T, U3> {}

impl<T: Scalar, Table: QuadratureTable<T, U1>> QuadratureTable1d<T> for Table {}
impl<T: Scalar, Table: QuadratureTable<T, U2>> QuadratureTable2d<T> for Table {}
impl<T: Scalar, Table: QuadratureTable<T, U3>> QuadratureTable3d<T> for Table {}

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

fn unit_data_table_for_weights<T>(points: &NestedVec<T>) -> NestedVec<()> {
    let mut data = NestedVec::new();
    for i in 0..points.len() {
        data.push(&vec![(); points.get(i).unwrap().len()]);
    }
    data
}

impl<T, GeometryDim> GeneralQuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    pub fn from_points_and_weights(points: NestedVec<OPoint<T, GeometryDim>>, weights: NestedVec<T>) -> Self {
        let data = unit_data_table_for_weights(&weights);
        Self::from_points_weights_and_data(points, weights, data)
    }
}

/// Checks that the provided quadrature rules are consistent, in the sense that
/// the number of elements for each table is identical, and that each rule has
/// consistent numbers of points, weights and data entries.
fn check_rules_consistency<T, D, Data>(points: &NestedVec<OPoint<T, D>>, weights: &NestedVec<T>, data: &NestedVec<Data>)
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    assert_eq!(
        points.len(),
        weights.len(),
        "Quadrature point and weight tables must have the same number of rules."
    );
    assert_eq!(
        points.len(),
        data.len(),
        "Quadrature point and data tables must have the same number of rules."
    );

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
        check_rules_consistency(&points, &weights, &data);
        Self { points, weights, data }
    }

    pub fn into_parts(self) -> GeneralQuadratureParts<T, GeometryDim, Data> {
        GeneralQuadratureParts {
            points: self.points,
            weights: self.weights,
            data: self.data,
        }
    }

    /// Replaces the data of the quadrature table with the given data.
    pub fn with_data<NewData>(self, data: NestedVec<NewData>) -> GeneralQuadratureTable<T, GeometryDim, NewData> {
        GeneralQuadratureTable {
            points: self.points,
            weights: self.weights,
            data: data,
        }
    }

    /// Replaces the data of the quadrature table by calling the given closure with every quadrature
    /// point in reference coordinates and its element index.
    pub fn with_data_from_fn<NewData>(
        self,
        mut data_fn: impl FnMut(usize, &OPoint<T, GeometryDim>, &Data) -> NewData,
    ) -> GeneralQuadratureTable<T, GeometryDim, NewData> {
        let mut data = NestedVec::new();

        for (element_index, (points, datas)) in self.points.iter().zip(self.data.iter()).enumerate() {
            let mut arr = data.begin_array();

            for (point, data) in points.iter().zip(datas.iter()) {
                arr.push_single(data_fn(element_index, point, data));
            }
        }

        GeneralQuadratureTable {
            points: self.points,
            weights: self.weights,
            data: data,
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

impl<T, GeometryDim, Data> UniformQuadratureTable<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
    Data: Clone,
{
    pub fn to_general(&self, num_elements: usize) -> GeneralQuadratureTable<T, GeometryDim, Data> {
        let mut points = NestedVec::new();
        let mut weights = NestedVec::new();
        let mut data = NestedVec::new();

        for _ in 0..num_elements {
            points.push(&self.points);
            weights.push(&self.weights);
            data.push(&self.data);
        }

        GeneralQuadratureTable::from_points_weights_and_data(points, weights, data)
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

/// A general quadrature table that avoids duplication of identical rules.
///
/// In a nutshell, [`CompactQuadratureTable`] sits in between [`UniformQuadratureTable`]
/// and [`GeneralQuadratureTable`]. Like [`GeneralQuadratureTable`], it can store an arbitrary
/// rule per element, but it uses a layer of indirection so that `M` quadrature rules
/// are associated with `N` elements.
///
/// This can be useful in settings where many elements share the same quadrature data, such
/// as a finite element space with elements of different degrees, or the common case
/// where the elements are the same but different quadrature data is needed in different
/// regions of the mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactQuadratureTable<T, D, Data = ()>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    #[serde(bound(serialize = "OPoint<T, D>: Serialize"))]
    #[serde(bound(deserialize = "OPoint<T, D>: Deserialize<'de>"))]
    points: NestedVec<OPoint<T, D>>,
    weights: NestedVec<T>,
    data: NestedVec<Data>,
    element_to_rule_map: Vec<usize>,
}

impl<T, D> CompactQuadratureTable<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn from_points_weights_and_map(
        points: NestedVec<OPoint<T, D>>,
        weights: NestedVec<T>,
        element_to_rule_map: Vec<usize>,
    ) -> Self {
        let data = unit_data_table_for_weights(&weights);
        Self::from_quadrature_rules_and_map(points, weights, data, element_to_rule_map)
    }
}

impl<T, D, Data> CompactQuadratureTable<T, D, Data>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// Construct a new table from the given quadrature rules and a map from elements
    /// to quadrature rules.
    ///
    /// # Panics
    ///
    /// Panics if `points`, `weights` and `data` are not consistent with each other.
    ///
    /// Panics if the mapping from elements to quadrature rules contains indices that are
    /// out of bounds with respect to the number of quadrature rules.
    pub fn from_quadrature_rules_and_map(
        points: NestedVec<OPoint<T, D>>,
        weights: NestedVec<T>,
        data: NestedVec<Data>,
        element_to_rule_map: Vec<usize>,
    ) -> Self {
        check_rules_consistency(&points, &weights, &data);
        let num_rules = points.len();
        let rule_indices_in_bounds = element_to_rule_map
            .iter()
            .all(|rule_index| rule_index < &num_rules);
        assert!(
            rule_indices_in_bounds,
            "Each rule index must correspond to a provided quadrature rule."
        );
        Self {
            element_to_rule_map,
            points,
            weights,
            data,
        }
    }

    fn rule_index_for_element(&self, element_index: usize) -> usize {
        self.element_to_rule_map[element_index]
    }
}

impl<T, D, Data> QuadratureTable<T, D> for CompactQuadratureTable<T, D, Data>
where
    T: Scalar,
    D: SmallDim,
    Data: Default + Clone,
    DefaultAllocator: Allocator<T, D>,
{
    type Data = Data;

    fn element_quadrature_size(&self, element_index: usize) -> usize {
        let rule_index = self.rule_index_for_element(element_index);
        self.points
            .get(rule_index)
            .expect("Internal error: Rule index out of bounds")
            .len()
    }

    fn populate_element_data(&self, element_index: usize, data: &mut [Self::Data]) {
        let rule_index = self.rule_index_for_element(element_index);
        let data_array = self
            .data
            .get(rule_index)
            .expect("Internal error: Rule index out of bounds");
        assert_eq!(
            data.len(),
            data_array.len(),
            "Length mismatch in data array: Stored quadrature data array has different length than output array."
        );
        data.clone_from_slice(data_array);
    }

    fn populate_element_quadrature(&self, element_index: usize, points: &mut [OPoint<T, D>], weights: &mut [T]) {
        let rule_index = self.rule_index_for_element(element_index);
        let points_array = self
            .points
            .get(rule_index)
            .expect("Internal error: Rule index out of bounds");
        let weights_array = self
            .weights
            .get(rule_index)
            .expect("Internal error: Rule index out of bounds");
        assert_eq!(
            points.len(),
            points_array.len(),
            "Length mismatch in points array: Stored points array has different length than output array."
        );
        assert_eq!(
            weights.len(),
            weights_array.len(),
            "Length mismatch in points array: Stored points array has different length than output array."
        );
        points.clone_from_slice(points_array);
        weights.clone_from_slice(weights_array);
    }
}

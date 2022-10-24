//! Lower level details for refinement abstractions.

use crate::allocators::DimAllocator;
use crate::connectivity::Tri3d2Connectivity;
use crate::mesh::refinement::{InvalidVertexCount, RefineConnectivity, UniformRefinement, VertexRepresentation};
use core::cmp::{max, min};
use core::hash::{Hash, Hasher};
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::DimName;
use nalgebra::OPoint;
use nalgebra::RealField;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VertexLabel(pub usize);

impl VertexRepresentation for VertexLabel {
    fn construct_vertex<T, D>(&self, all_vertices: &[OPoint<T, D>]) -> OPoint<T, D>
    where
        T: RealField,
        D: DimName,
        DefaultAllocator: DimAllocator<T, D>,
    {
        let &Self(vertex_idx) = self;
        all_vertices[vertex_idx].clone()
    }
}

#[derive(Debug, Copy, Clone, Eq)]
pub struct EdgeMidpointLabel(pub [usize; 2]);

impl EdgeMidpointLabel {
    fn canonical_vertex_indices(&self) -> [usize; 2] {
        let &EdgeMidpointLabel([a, b]) = self;
        [min(a, b), max(a, b)]
    }
}

impl VertexRepresentation for EdgeMidpointLabel {
    fn construct_vertex<T, D>(&self, all_vertices: &[OPoint<T, D>]) -> OPoint<T, D>
    where
        T: RealField,
        D: DimName,
        DefaultAllocator: DimAllocator<T, D>,
    {
        let &Self(vertex_indices) = self;
        let [a, b] = vertex_indices.map(|idx| &all_vertices[idx]);
        OPoint::from((&a.coords + &b.coords) / T::from_subset(&2.0))
    }
}

impl PartialEq for EdgeMidpointLabel {
    fn eq(&self, other: &Self) -> bool {
        self.canonical_vertex_indices() == other.canonical_vertex_indices()
    }
}

impl Hash for EdgeMidpointLabel {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_vertex_indices().hash(state)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum VertexOrEdgeMidpointVertex {
    Vertex(VertexLabel),
    EdgeMidpoint(EdgeMidpointLabel),
}

impl From<VertexLabel> for VertexOrEdgeMidpointVertex {
    fn from(label: VertexLabel) -> Self {
        Self::Vertex(label)
    }
}

impl From<EdgeMidpointLabel> for VertexOrEdgeMidpointVertex {
    fn from(label: EdgeMidpointLabel) -> Self {
        Self::EdgeMidpoint(label)
    }
}

impl VertexRepresentation for VertexOrEdgeMidpointVertex {
    fn construct_vertex<T, D>(&self, all_vertices: &[OPoint<T, D>]) -> OPoint<T, D>
    where
        T: RealField,
        D: DimName,
        DefaultAllocator: DimAllocator<T, D>,
    {
        match self {
            Self::Vertex(label) => label.construct_vertex(all_vertices),
            Self::EdgeMidpoint(label) => label.construct_vertex(all_vertices),
        }
    }
}

pub fn edge_midpoint(vertices: [usize; 2]) -> EdgeMidpointLabel {
    EdgeMidpointLabel(vertices)
}

pub fn vertex(vertex: usize) -> VertexLabel {
    VertexLabel(vertex)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntermediateTri3d2([VertexOrEdgeMidpointVertex; 3]);

impl RefineConnectivity<Tri3d2Connectivity> for UniformRefinement {
    type Intermediate = IntermediateTri3d2;
    type OutputConnectivity = Tri3d2Connectivity;
    type VertexLabel = VertexOrEdgeMidpointVertex;

    fn populate_refined_connectivity(
        &self,
        connectivity: &Tri3d2Connectivity,
        intermediates: &mut Vec<Self::Intermediate>,
    ) {
        let &Tri3d2Connectivity([a, b, c]) = connectivity;
        let d = edge_midpoint([a, b]).into();
        let e = edge_midpoint([b, c]).into();
        let f = edge_midpoint([c, a]).into();
        let [a, b, c] = [a, b, c].map(|vertex_idx| vertex(vertex_idx).into());

        intermediates.extend_from_slice(&[
            IntermediateTri3d2([a, d, f]),
            IntermediateTri3d2([d, b, e]),
            IntermediateTri3d2([f, e, c]),
            IntermediateTri3d2([d, e, f]),
        ]);
    }

    fn populate_vertex_labels(&self, intermediate: &Self::Intermediate, labels: &mut Vec<Self::VertexLabel>) {
        labels.extend_from_slice(&intermediate.0);
    }

    fn construct_output_connectivity(
        &self,
        _intermediate: &Self::Intermediate,
        vertex_indices: &[usize],
    ) -> Result<Self::OutputConnectivity, InvalidVertexCount> {
        Ok(Tri3d2Connectivity(
            vertex_indices.try_into().map_err(|_| InvalidVertexCount)?,
        ))
    }
}

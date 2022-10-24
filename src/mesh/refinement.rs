//! Functionality and abstractions for mesh refinement.
//!
//! Currently we only provide uniform refinement for select element types through
//! [`refine_mesh`] and [`UniformRefinement`].
use crate::allocators::DimAllocator;
use crate::connectivity::Connectivity;
use crate::mesh::Mesh;
use nalgebra::{DefaultAllocator, DimName, OPoint, RealField};
use std::collections::HashMap;
use std::hash::Hash;

pub mod detail;

#[derive(Debug, Clone)]
pub struct InvalidVertexCount;

pub trait VertexRepresentation: Clone {
    fn construct_vertex<T, D>(&self, all_vertices: &[OPoint<T, D>]) -> OPoint<T, D>
    where
        T: RealField,
        D: DimName,
        DefaultAllocator: DimAllocator<T, D>;
}

/// Defines a refinement scheme for a given connectivity.
pub trait RefineConnectivity<Connectivity> {
    /// An intermediate connectivity type that holds the necessary information
    /// to represent the connectivity in a manner independent of the index labeling of
    /// new vertices.
    ///
    /// This is generally an internal detail.
    type Intermediate;
    /// The resulting connectivity type after refinement.
    type OutputConnectivity;
    /// The kind of label used to define vertices in a globally consistent fashion.
    type VertexLabel: VertexRepresentation;

    /// For a given connectivity, produce a set of intermediate connectivities that represent
    /// refined connectivities.
    fn populate_refined_connectivity(&self, connectivity: &Connectivity, intermediates: &mut Vec<Self::Intermediate>);

    /// Populate a set of vertex labels associated with an intermediate connectivity.
    fn populate_vertex_labels(&self, intermediate: &Self::Intermediate, labels: &mut Vec<Self::VertexLabel>);

    /// Construct the final connectivity from an intermediate given final,
    /// globally consistent vertex indices.
    ///
    /// The implementation must succeed if the number of vertex indices is equal to the
    /// number of vertex labels associated with the intermediate, as defined by
    /// the number of vertex labels produced by
    /// [`populate_vertex_labels`](Self::populate_vertex_labels).
    fn construct_output_connectivity(
        &self,
        intermediate: &Self::Intermediate,
        vertex_indices: &[usize],
    ) -> Result<Self::OutputConnectivity, InvalidVertexCount>;
}

pub struct UniformRefinement;

/// Refine a mesh with the provided refinement scheme.
pub fn refine_mesh<T, D, C, Refinement>(
    mesh: &Mesh<T, D, C>,
    refinement_scheme: Refinement,
) -> Mesh<T, D, Refinement::OutputConnectivity>
where
    T: RealField,
    D: DimName,
    Refinement: RefineConnectivity<C>,
    Refinement::VertexLabel: Eq + Hash,
    DefaultAllocator: DimAllocator<T, D>,
{
    let mut label_to_idx_map = HashMap::new();
    let mut next_vertex_idx = 0;

    let mut new_connectivity = Vec::new();

    // Local buffers
    let mut intermediates = Vec::new();
    let mut vertex_labels = Vec::new();
    let mut new_vertex_indices = Vec::new();
    for connectivity in mesh.connectivity() {
        new_vertex_indices.clear();
        intermediates.clear();
        refinement_scheme.populate_refined_connectivity(&connectivity, &mut intermediates);
        for intermediate in &intermediates {
            vertex_labels.clear();
            new_vertex_indices.clear();
            refinement_scheme.populate_vertex_labels(&intermediate, &mut vertex_labels);
            for label in &vertex_labels {
                let idx = label_to_idx_map.entry(label.clone()).or_insert_with(|| {
                    let idx = next_vertex_idx;
                    next_vertex_idx += 1;
                    idx
                });
                new_vertex_indices.push(*idx);
            }
            let new_cell_connectivity = refinement_scheme
                .construct_output_connectivity(&intermediate, &new_vertex_indices)
                .expect("Must succeed since vertex label count is consistent with vertex index count");
            new_connectivity.push(new_cell_connectivity);
        }
    }

    let mut new_vertices = vec![Default::default(); next_vertex_idx];
    for (label, index) in label_to_idx_map {
        let vertex = label.construct_vertex(mesh.vertices());
        new_vertices[index] = vertex;
    }
    Mesh::from_vertices_and_connectivity(new_vertices, new_connectivity)
}

/// Apply one round of uniform mesh refinement.
///
/// This is a convenience function for `refine_mesh(mesh, UniformRefinement)`.
pub fn refine_uniformly<T, D, C>(mesh: &Mesh<T, D, C>) -> Mesh<T, D, C>
where
    T: RealField,
    D: DimName,
    UniformRefinement: RefineConnectivity<C, OutputConnectivity = C>,
    <UniformRefinement as RefineConnectivity<C>>::VertexLabel: Eq + Hash,
    DefaultAllocator: DimAllocator<T, D>,
{
    refine_mesh(mesh, UniformRefinement)
}

/// Repeatedly applies uniform mesh refinement to the given mesh.
pub fn refine_uniformly_repeat<T, D, C>(mesh: &Mesh<T, D, C>, repeat_times: usize) -> Mesh<T, D, C>
where
    T: RealField,
    D: DimName,
    C: Connectivity,
    UniformRefinement: RefineConnectivity<C, OutputConnectivity = C>,
    <UniformRefinement as RefineConnectivity<C>>::VertexLabel: Eq + Hash,
    DefaultAllocator: DimAllocator<T, D>,
{
    let mut mesh: Mesh<_, _, _> = mesh.clone();
    for _ in 0..repeat_times {
        mesh = refine_uniformly(&mesh);
    }
    mesh
}

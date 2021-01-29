use crate::assembly::CsrParAssembler;
use crate::connectivity::{Connectivity, ConnectivityMut};
use crate::mesh::Mesh;
use core::fmt;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Scalar};
use fenris_sparse::SparsityPattern;
use std::collections::VecDeque;
use std::error::Error;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct MeshPermutation {
    vertex_perm: Permutation,
    connectivity_perm: Permutation,
}

impl MeshPermutation {
    pub fn vertex_permutation(&self) -> &Permutation {
        &self.vertex_perm
    }

    pub fn connectivity_permutation(&self) -> &Permutation {
        &self.connectivity_perm
    }

    pub fn apply<T, D, C>(&self, mesh: &Mesh<T, D, C>) -> Mesh<T, D, C>
    where
        T: Scalar,
        D: DimName,
        C: ConnectivityMut,
        DefaultAllocator: Allocator<T, D>,
    {
        let new_vertices = self.vertex_permutation().apply_to_slice(mesh.vertices());

        // Connectivity is a little more involved: we need to update all "old" vertex indices
        // referenced to new vertices
        let inv_vertex_perm = self.vertex_permutation().inverse();
        let mut new_connectivity = self
            .connectivity_permutation()
            .apply_to_slice(mesh.connectivity());
        for conn in &mut new_connectivity {
            for vertex_idx in conn.vertex_indices_mut() {
                let new_vertex_index = inv_vertex_perm.source_index(*vertex_idx);
                *vertex_idx = new_vertex_index;
            }
        }
        Mesh::from_vertices_and_connectivity(new_vertices, new_connectivity)
    }
}

/// Creates a mesh permutation by computing a Reverse Cuthill-McKee permutation.
pub fn reorder_mesh_par<T, D, C>(mesh: &Mesh<T, D, C>) -> MeshPermutation
where
    T: Scalar,
    D: DimName,
    C: Sync + Connectivity,
    DefaultAllocator: Allocator<T, D>,
    Mesh<T, D, C>: Sync,
{
    let assembler = CsrParAssembler::<i32>::default();

    // Construct the CSR adjacency matrix for the graph represented by the mesh
    let csr_graph = assembler.assemble_pattern(mesh);
    let vertex_perm = reverse_cuthill_mckee(&csr_graph);
    let inv_vertex_perm = vertex_perm.inverse();

    // Reorder connectivity by sorting the connectivities by minimum (permuted) vertex index,
    // which, after vertex reordering, typically has the effect of re-ordering elements
    // so that elements with similar indices reference vertices with similar indices,
    // thereby improving memory locality.
    let mut connectivity_perm: Vec<_> = (0..mesh.connectivity().len()).collect();
    connectivity_perm.sort_by_key(|&connectivity_index| {
        let vertices = mesh.connectivity()[connectivity_index].vertex_indices();
        vertices
            .iter()
            // Need to sort by the *new* (permuted) index of the vertex, not the old one
            .map(|old_vertex_idx| inv_vertex_perm.source_index(*old_vertex_idx))
            .min()
    });
    let connectivity_perm = Permutation::from_vec(connectivity_perm)
        .expect("Internal error: Connectivity permutation must always be valid.");

    MeshPermutation {
        vertex_perm,
        connectivity_perm,
    }
}

/// A representation of an index permutation.
///
/// More precisely, given `n` objects stored contiguously, the permutation internally
/// stores a permutation array `perm` such that for *target index* `i` in `0 .. n`,
/// the corresponding *source index* is given by
///
/// ```ignore
/// target[i] = source[perm[i]]
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    perm: Vec<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct InvalidPermutation {
    marker: PhantomData<()>,
}

impl fmt::Display for InvalidPermutation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid permutation")
    }
}

impl Error for InvalidPermutation {}

impl Permutation {
    pub fn from_vec(perm: Vec<usize>) -> Result<Self, InvalidPermutation> {
        let mut visited = vec![false; perm.len()];
        for &index in &perm {
            if visited[index] {
                return Err(InvalidPermutation {
                    marker: PhantomData,
                });
            } else {
                visited[index] = true;
            }
        }
        Ok(Self { perm })
    }

    pub fn len(&self) -> usize {
        self.perm.len()
    }

    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    pub fn reverse(&mut self) {
        self.perm.reverse()
    }

    pub fn source_index(&self, target_index: usize) -> usize {
        self.perm[target_index]
    }

    pub fn inverse(&self) -> Permutation {
        let mut inverse_perm = vec![std::usize::MAX; self.len()];
        for (target_idx, &source_idx) in self.perm().iter().enumerate() {
            inverse_perm[source_idx] = target_idx;
        }

        // TODO: Consider using an unsafe unchecked method
        Self::from_vec(inverse_perm).unwrap()
    }

    pub fn apply_to_slice<T: Clone>(&self, slice: &[T]) -> Vec<T> {
        assert_eq!(
            slice.len(),
            self.len(),
            "Slice and permutation must have the same size."
        );
        self.perm()
            .iter()
            .map(|source_idx| slice[*source_idx].clone())
            .collect()
    }
}

/// Create a vertex permutation for a sparse symmetric matrix using the Cuthill-McKee algorithm.
pub fn cuthill_mckee(sparsity_pattern: &SparsityPattern) -> Permutation {
    assert_eq!(
        sparsity_pattern.major_dim(),
        sparsity_pattern.minor_dim(),
        "Matrix must be square."
    );

    let adjacent_vertices = |vertex_idx| {
        sparsity_pattern
            .lane(vertex_idx)
            .expect("Vertex must be in bounds")
    };
    let vertex_degree = |vertex_idx| adjacent_vertices(vertex_idx).len();

    let mut queue = VecDeque::new();
    let mut permutation = Vec::with_capacity(sparsity_pattern.major_dim());
    let mut visited = vec![false; sparsity_pattern.major_dim()];

    let mut adjacency_workspace = Vec::new();

    // For matrices with zero rows or block diagonal patterns, the standard CutHill-McKee
    // algorithm would not run to completion, because it assumes that all vertices are connected.
    // To cope with this, we keep re-running the algorithm with a different starting vertex
    // picked from the set of non-visited vertices
    // Note: the current implementation is asymptotically very suboptimal if
    // the sparsity pattern consists of a large number of disjoint components.
    // TODO: Improve this?
    while visited.iter().any(|entry| entry == &false) {
        // Only look for minimum degree among vertices we have not yet visited
        // TODO: Chosing the first minimum degree vertex is not necessarily the best choice.
        // Investigate this!
        let least_degree_vertex = (0..sparsity_pattern.major_dim())
            .filter(|vertex_idx| visited[*vertex_idx] == false)
            .min_by_key(|vertex_idx| adjacent_vertices(*vertex_idx).len());

        if let Some(start_vertex) = least_degree_vertex {
            queue.push_back(start_vertex);
            visited[start_vertex] = true;

            while let Some(vertex) = queue.pop_front() {
                adjacency_workspace.clear();
                adjacency_workspace.extend(adjacent_vertices(vertex));
                adjacency_workspace.sort_unstable_by_key(|idx| vertex_degree(*idx));

                permutation.push(vertex);

                // Cuthill-McKee is essentially just a breadth-first search in which
                // the neighbors are visited in sorted order from lowest to highest
                // vertex degree
                for &adjacent_vertex in &adjacency_workspace {
                    if !visited[adjacent_vertex] {
                        visited[adjacent_vertex] = true;
                        queue.push_back(adjacent_vertex);
                    }
                }
            }
        }
    }

    assert_eq!(
        permutation.len(),
        sparsity_pattern.major_dim(),
        "Internal error: Permutation has invalid length"
    );
    Permutation::from_vec(permutation).expect("Internal error: Constructed permutation is invalid")
}

/// Create a vertex permutation for a sparse symmetric matrix using the Reverse Cuthill-McKee (RCM)
/// algorithm.
pub fn reverse_cuthill_mckee(sparsity_pattern: &SparsityPattern) -> Permutation {
    let mut perm = cuthill_mckee(sparsity_pattern);
    perm.reverse();
    perm
}

use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Point3, RealField, Scalar, Vector3, U3};
use numeric_literals::replace_float_literals;
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;

// use crate::connectivity::Connectivity;
use crate::{compute_polyhedron_volume_from_faces, ConvexPolygon3d, ConvexPolyhedron, HalfSpace, LineSegment3d};
// use crate::mesh::Mesh;
use fenris_nested_vec::NestedVec;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct UndirectedEdge {
    // Indices are always sorted, so that a <= b, for [a, b]
    indices: [usize; 2],
}

impl UndirectedEdge {
    pub fn new(a: usize, b: usize) -> Self {
        Self {
            indices: [min(a, b), max(a, b)],
        }
    }

    pub fn indices(&self) -> &[usize; 2] {
        &self.indices
    }
}

#[derive(Debug)]
pub struct PolyMeshFace<'a, T: Scalar, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    all_vertices: &'a [OPoint<T, D>],
    face_vertex_indices: &'a [usize],
}

impl<'a, T: Scalar> ConvexPolygon3d<'a, T> for PolyMeshFace<'a, T, U3> {
    fn num_vertices(&self) -> usize {
        self.face_vertex_indices.len()
    }

    fn get_vertex(&self, index: usize) -> Option<OPoint<T, U3>> {
        let v = self
            .all_vertices
            .get(*self.face_vertex_indices.get(index)?)
            .expect("Internal error: Vertex must always exist if the local index is valid.");
        Some(v.clone())
    }
}

/// A volumetric polytopal mesh.
///
/// It is assumed that each polytopal cell is convex.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "OPoint<T, D>: Serialize"))]
#[serde(bound(deserialize = "OPoint<T, D>: Deserialize<'de>"))]
pub struct PolyMesh<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    #[serde(bound(serialize = "<DefaultAllocator as Allocator<T, D>>::Buffer: Serialize"))]
    #[serde(bound(deserialize = "<DefaultAllocator as Allocator<T, D>>::Buffer: Deserialize<'de>"))]
    vertices: Vec<OPoint<T, D>>,
    faces: NestedVec<usize>,
    cells: NestedVec<usize>,
}

pub type PolyMesh3d<T> = PolyMesh<T, U3>;

impl<T, D> PolyMesh<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// Creates an empty poly mesh without any vertices, faces or cells.
    pub fn new_empty() -> Self {
        Self::from_poly_data(vec![], NestedVec::new(), NestedVec::new())
    }

    pub fn from_poly_data(vertices: Vec<OPoint<T, D>>, faces: NestedVec<usize>, cells: NestedVec<usize>) -> Self {
        let num_vertices = vertices.len();
        let num_faces = faces.len();

        if faces.iter_array_elements().any(|idx| *idx >= num_vertices) {
            panic!("Vertex index out of bounds in faces description.")
        }

        if cells.iter_array_elements().any(|idx| *idx >= num_faces) {
            panic!("Face index out of bounds in cells description.")
        }

        Self {
            vertices,
            faces: faces,
            cells: cells,
        }
    }

    pub fn vertices(&self) -> &[OPoint<T, D>] {
        &self.vertices
    }

    pub fn vertices_mut(&mut self) -> &mut [OPoint<T, D>] {
        &mut self.vertices
    }

    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn face_vertices<'a>(&'a self, face_idx: usize) -> impl 'a + Iterator<Item = &'a OPoint<T, D>> {
        self.get_face_connectivity(face_idx)
            .into_iter()
            .flatten()
            .map(move |vertex_idx| &self.vertices()[*vertex_idx])
    }

    pub fn face_connectivity_iter<'a>(&'a self) -> impl 'a + Iterator<Item = &'a [usize]> {
        self.faces.iter()
    }

    pub fn cell_connectivity_iter<'a>(&'a self) -> impl 'a + Iterator<Item = &'a [usize]> {
        self.cells.iter()
    }

    pub fn get_face_connectivity(&self, index: usize) -> Option<&[usize]> {
        self.faces.get(index)
    }

    pub fn get_cell_connectivity(&self, index: usize) -> Option<&[usize]> {
        self.cells.get(index)
    }

    pub fn get_face(&self, index: usize) -> Option<PolyMeshFace<T, D>> {
        self.get_face_connectivity(index)
            .map(|face_vertex_indices| PolyMeshFace {
                all_vertices: &self.vertices,
                face_vertex_indices,
            })
    }

    /// Returns a nested array, in which each array i contains the indices of the cells
    /// associated with face i.
    pub fn compute_face_cell_connectivity(&self) -> NestedVec<usize> {
        // TODO: Implement this more efficiently so that we don't construct a
        // Vec<Vec<_>>. Ideally we'd implement facilities for CompactArrayStorage to be able
        // to push directly to existing arrays (by moving the array around in storage if necessary).
        let mut connectivity = vec![Vec::new(); self.num_faces()];

        for (cell_idx, cell) in self.cell_connectivity_iter().enumerate() {
            for face_idx in cell {
                connectivity[*face_idx].push(cell_idx);
            }
        }

        let mut compact = NestedVec::new();
        for face_cell_conn in connectivity {
            compact.push(&face_cell_conn);
        }

        compact
    }

    /// Removes duplicate instances of topologically equivalent faces.
    ///
    /// Two faces are topologically equivalent if the sets of vertices connected by each face
    /// are equivalent.
    pub fn dedup_faces(&mut self) {
        let mut sorted_face_connectivity = NestedVec::new();
        for face in self.face_connectivity_iter() {
            sorted_face_connectivity.push(face);
            sorted_face_connectivity.last_mut().unwrap().sort_unstable();
        }

        let mut new_faces = NestedVec::new();
        let mut connectivity_map = HashMap::new();
        // Store new_indices of faces
        let mut face_relabel = Vec::with_capacity(self.num_faces());
        for i in 0..self.num_faces() {
            let sorted_face_conn = sorted_face_connectivity.get(i).unwrap();
            // TODO: Rewrite using entry API to avoid double lookup
            if let Some(new_face_index) = connectivity_map.get(sorted_face_conn) {
                // We've already encountered a face with equivalent topology,
                // so map this face to the one we've found already
                face_relabel.push(*new_face_index);
            } else {
                // We haven't so far encountered a face with equivalent topology,
                // so we add this face to the new collection of faces.
                let new_index = new_faces.len();
                connectivity_map.insert(sorted_face_conn, new_index);
                new_faces.push(self.get_face_connectivity(i).unwrap());
                face_relabel.push(new_index);
            }
        }

        self.faces = new_faces;

        for i in 0..self.num_cells() {
            let cell = self.cells.get_mut(i).unwrap();

            for face_idx in cell {
                *face_idx = face_relabel[*face_idx];
            }
        }
    }

    /// Returns the indices of the faces which are only referenced by at most one cells.
    pub fn find_boundary_faces(&self) -> Vec<usize> {
        let mut face_occurences = vec![0; self.num_faces()];

        for cell_faces in self.cell_connectivity_iter() {
            for face in cell_faces {
                face_occurences[*face] += 1;
            }
        }

        face_occurences
            .into_iter()
            .enumerate()
            .filter_map(|(face_idx, count)| if count <= 1 { Some(face_idx) } else { None })
            .collect()
    }

    /// Merges multiple meshes into a single instance of `PolyMesh`.
    ///
    /// The mesh vertices, faces and cells are simply relabeled and glued together so that they
    /// form a well-defined PolyMesh. No mesh processing is performed.
    pub fn concatenate<'a>(meshes: impl IntoIterator<Item = &'a PolyMesh<T, D>>) -> Self {
        let meshes = meshes.into_iter();
        let mut vertices = Vec::new();
        let mut faces = NestedVec::new();
        let mut cells = NestedVec::new();

        let mut vertex_offset = 0;
        let mut face_offset = 0;

        for mesh in meshes {
            vertices.extend(mesh.vertices().iter().cloned());

            for face_vertices in mesh.face_connectivity_iter() {
                let mut new_face_vertices = faces.begin_array();
                for vertex_idx in face_vertices {
                    new_face_vertices.push_single(vertex_idx + vertex_offset);
                }
            }

            for cell_faces in mesh.cell_connectivity_iter() {
                let mut new_cell_faces = cells.begin_array();
                for face_idx in cell_faces {
                    new_cell_faces.push_single(face_idx + face_offset);
                }
            }

            vertex_offset += mesh.vertices().len();
            face_offset += mesh.num_faces();
        }

        Self::from_poly_data(vertices, faces, cells)
    }
}

impl<T, D> PolyMesh<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    /// Recursively splits each edge in the mesh the specified number of times
    pub fn split_edges_n_times(&mut self, n_times: usize) {
        for _ in 0..n_times {
            self.split_edges()
        }
    }

    /// Splits the edges of all faces in the mesh by inserting a vertex at the midpoint of each edge
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn split_edges(&mut self) {
        let vertex_offset = self.vertices.len();
        let mut additional_vertices = Vec::new();
        let mut new_faces = NestedVec::new();
        let mut subdivided_edges = HashMap::new();

        for face in self.faces.iter() {
            let edges = face
                .iter()
                .copied()
                .cycle()
                .take(face.len() + 1)
                .tuple_windows();
            let mut new_face = new_faces.begin_array();
            for (v1, v2) in edges {
                let edge = [v1.min(v2), v1.max(v2)];
                let v12 = *subdivided_edges.entry(edge).or_insert_with(|| {
                    let new_vertex_index = vertex_offset + additional_vertices.len();
                    let midpoint = OPoint::from((&self.vertices[v1].coords + &self.vertices[v2].coords) * 0.5);
                    additional_vertices.push(midpoint);
                    new_vertex_index
                });
                new_face.push_single(v1);
                new_face.push_single(v12);
            }
        }

        self.vertices.extend(additional_vertices);
        self.faces = new_faces;
    }
}

impl<T> PolyMesh3d<T>
where
    T: Scalar,
{
    /// Triangulate the polyhedral mesh.
    ///
    /// Note that the algorithm currently only gives non-degenerate results when each cell
    /// is *strictly* convex, in the sense that no two faces of a cell are co-planar.
    ///
    /// TODO: Can we relax the strict convexity restriction without introducing additional
    /// Steiner points into the triangulation? The restriction is explained in the
    /// paper by Max (2000) (see comments in implementation).
    pub fn triangulate(&self) -> Result<PolyMesh3d<T>, Box<dyn Error>> {
        // The implementation here follows the procedure described in
        // Nelson Max (2000), "Consistent Subdivision of Convex Polyhedra into Tetrahedra"
        // The key is as follows: Whenever subdividing a face/cell into triangles/tetrahedra,
        // we do so by connecting the vertex of smallest index to the triangle faces
        // that were obtained from faces not incident to the vertex.

        // First triangulate each face and maintain a map from the original face to the
        // new face indices

        let mut triangulated_faces = NestedVec::new();
        let mut face_map = NestedVec::new();

        for face in self.face_connectivity_iter() {
            if face.len() < 3 {
                return Err(Box::from(
                    "Encountered face with less than 3 vertices,\
                     cannot triangulate.",
                ));
            }

            let mut face_map_array = face_map.begin_array();
            // Pick the vertex with smallest index
            let (min_idx, _) = face
                .iter()
                .enumerate()
                .min_by_key(|(_, v_idx)| *v_idx)
                .unwrap();

            for i in 0..face.len() - 2 {
                let a = face[min_idx];
                let b = face[(i + 1 + min_idx) % face.len()];
                let c = face[(i + 2 + min_idx) % face.len()];
                face_map_array.push_single(triangulated_faces.len());
                triangulated_faces.push(&[a, b, c]);
            }
        }

        let mut tetrahedral_cells = NestedVec::new();
        for cell in self.cell_connectivity_iter() {
            // Ignore empty cells
            if cell.len() > 0 {
                // Pick the vertex of least index in the cell
                let v_idx = cell
                    .iter()
                    .flat_map(|face_idx| {
                        self.get_face_connectivity(*face_idx).expect(
                            "Logic error: Cell references face that \
                             does not exist.",
                        )
                    })
                    .min()
                    .expect("Cell is non-empty");

                // For each original face in the cell that does *not* contain the vertex,
                // create new tetrahedral cells by connecting the chosen vertex
                // to the triangulated faces.
                // It is important to discard the *original* faces with this test, because
                // otherwise we would create degenerate tetrahedral cells, as our
                // chosen vertex would be in the same plane as the 3 vertices of the triangulated
                // face.
                for original_face_idx in cell {
                    let original_face_vertices = self
                        .get_face_connectivity(*original_face_idx)
                        .expect("Logic error: Cell references face that does not exist.");

                    if !original_face_vertices.contains(v_idx) {
                        let triangulated_face_indices = face_map
                            .get(*original_face_idx)
                            .expect("Logic error: Cell references face that does not exist.");

                        for tri_face_idx in triangulated_face_indices {
                            let tri_face_vertices = triangulated_faces.get(*tri_face_idx).unwrap();
                            assert_eq!(tri_face_vertices.len(), 3);

                            // Connect v to face by constructing 3 new triangular faces
                            let a = tri_face_vertices[0];
                            let b = tri_face_vertices[1];
                            let c = tri_face_vertices[2];
                            let v = *v_idx;

                            // Triangular faces denoted by vertices connected,
                            // i.e. abc means triangular face constructed by vertices a, b and c.
                            let abc_idx = *tri_face_idx;
                            let abv_idx = triangulated_faces.len();
                            let bcv_idx = abv_idx + 1;
                            let cav_idx = bcv_idx + 1;

                            // This will cause duplicated faces, but we deduplicate them
                            // as a post-process. TODO: Can we directly construct
                            // faces without duplicating faces in a succinct way?
                            triangulated_faces.push(&[a, b, v]);
                            triangulated_faces.push(&[b, c, v]);
                            triangulated_faces.push(&[c, a, v]);
                            tetrahedral_cells.push(&[abc_idx, abv_idx, bcv_idx, cav_idx]);
                        }
                    }
                }
            }
        }

        let mut new_poly_mesh =
            PolyMesh::from_poly_data(self.vertices().to_vec(), triangulated_faces, tetrahedral_cells);
        new_poly_mesh.dedup_faces();
        Ok(new_poly_mesh)
    }

    pub fn keep_cells(&self, cell_indices: &[usize]) -> Self {
        // Use BTreeSets so that the relative order of the indices are kept
        let keep_faces: BTreeSet<_> = cell_indices
            .iter()
            .flat_map(|cell_idx| {
                self.get_cell_connectivity(*cell_idx)
                    .expect("All cell indices must be in bounds")
            })
            .copied()
            .collect();

        let keep_vertices: BTreeSet<_> = keep_faces
            .iter()
            .flat_map(|face_idx| self.get_face_connectivity(*face_idx).unwrap())
            .copied()
            .collect();

        let faces_old_to_new_map: HashMap<_, _> = keep_faces
            .iter()
            .enumerate()
            .map(|(new_idx, old_idx)| (old_idx, new_idx))
            .collect();

        let vertices_old_to_new_map: HashMap<_, _> = keep_vertices
            .iter()
            .enumerate()
            .map(|(new_idx, old_idx)| (*old_idx, new_idx))
            .collect();

        let new_vertices = keep_vertices
            .iter()
            .map(|old_vertex_idx| self.vertices()[*old_vertex_idx].clone())
            .collect();

        let mut new_faces = NestedVec::new();
        for old_face_idx in &keep_faces {
            let old_face_vertices = self.get_face_connectivity(*old_face_idx).unwrap();
            let mut new_face_vertices = new_faces.begin_array();

            for old_vertex_idx in old_face_vertices {
                let new_vertex_idx = vertices_old_to_new_map.get(old_vertex_idx).unwrap();
                new_face_vertices.push_single(*new_vertex_idx);
            }
        }

        let mut new_cells = NestedVec::new();
        for old_cell_idx in cell_indices {
            let old_cell_faces = self.get_cell_connectivity(*old_cell_idx).unwrap();
            let mut new_cell_faces = new_cells.begin_array();

            for old_face_idx in old_cell_faces {
                let new_face_idx = faces_old_to_new_map.get(old_face_idx).unwrap();
                new_cell_faces.push_single(*new_face_idx);
            }
        }

        Self::from_poly_data(new_vertices, new_faces, new_cells)
    }
}

/// Marks vertices according to whether or not they are contained in the half space.
///
/// More precisely, given N vertices, a vector of N boolean values is returned.
/// If element `i` is `true`, then vertex `i` is contained in the half space.
fn mark_vertices<T: RealField>(vertices: &[Point3<T>], half_space: &HalfSpace<T>) -> Vec<bool> {
    vertices
        .iter()
        .map(|v| half_space.contains_point(v))
        .collect()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Classification {
    Inside,
    Outside,
    Cut,
}

//fn classify_face(face_vertices: &[usize], vertex_table: &[bool]) -> Classification {
//    let num_outside_vertices = face_vertices.iter().map(|i| vertex_table[*i]).count();
//
//    if num_outside_vertices == 0 {
//        Classification::Inside
//    } else if num_outside_vertices == face_vertices.len() {
//        Classification::Outside
//    } else {
//        Classification::Cut
//    }
//}

fn is_intersection_vertex(vertex_edge_representation: &UndirectedEdge) -> bool {
    let [a, b] = vertex_edge_representation.indices();
    a != b
}

impl<T> PolyMesh3d<T>
where
    T: RealField,
{
    pub fn translate(&mut self, translation: &Vector3<T>) {
        for v in self.vertices_mut() {
            *v += translation;
        }
    }

    pub fn translated(mut self, translation: &Vector3<T>) -> Self {
        self.translate(translation);
        self
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn compute_volume(&self) -> T {
        let boundary_faces = self.find_boundary_faces();
        let face_iter = boundary_faces
            .iter()
            .map(|face_idx| self.get_face(*face_idx).unwrap());

        compute_polyhedron_volume_from_faces(face_iter)
    }

    pub fn intersect_convex_polyhedron<'a>(&self, polyhedron: &impl ConvexPolyhedron<'a, T>) -> Self {
        let mut mesh = self.clone();
        for i in 0..polyhedron.num_faces() {
            let face = polyhedron.get_face(i).unwrap();
            if let Some(half_space) = face.compute_half_space() {
                mesh = mesh.intersect_half_space(&half_space);
            }
        }
        mesh
    }

    pub fn intersect_half_space(&self, half_space: &HalfSpace<T>) -> Self {
        // Our approach will be to first build up the complete description of faces
        // by representing vertices as undirected edges between two vertices a and b.
        // The physical vertex is defined to be the closest point to the half space on the
        // line segment connecting a and b.
        // In particular, if a == b, then the vertex is a vertex in the original mesh.
        // If a != b, then the vertex is an intersection point between an edge and the half space.

        // A vertex is a member of the half space if the half space contains the vertex.
        let vertex_half_space_membership = mark_vertices(&self.vertices, half_space);

        let mut new_faces = NestedVec::new();
        let mut face_classifications = Vec::with_capacity(self.num_faces());

        for face_vertices in self.faces.iter() {
            let mut new_face_vertices = new_faces.begin_array();

            let mut classification = Classification::Inside;

            let face_vertices = face_vertices.iter().chain(face_vertices.first()).copied();
            for (a, b) in face_vertices.tuple_windows() {
                let a_is_inside = vertex_half_space_membership[a];
                let b_is_inside = vertex_half_space_membership[b];

                if a_is_inside {
                    new_face_vertices.push_single(UndirectedEdge::new(a, a));
                }

                if a_is_inside != b_is_inside {
                    // Edge is cut
                    new_face_vertices.push_single(UndirectedEdge::new(a, b));
                    classification = Classification::Cut;
                }
            }

            // Only vertices which are somehow inside of the half space get added.
            // Thus, if we added no vertices, the face is entirely outside.
            if new_face_vertices.count() == 0 {
                classification = Classification::Outside;
            }

            face_classifications.push(classification);
        }

        #[derive(Debug)]
        struct IntersectionEdge {
            //            face_idx: usize,
            a: UndirectedEdge,
            b: UndirectedEdge,
        }

        let mut intersection_edges = Vec::new();

        let mut new_cells = NestedVec::new();
        for cell_faces in self.cells.iter() {
            intersection_edges.clear();
            let mut new_cell_faces = new_cells.begin_array();

            for face_idx in cell_faces {
                let face_classification = face_classifications[*face_idx];
                match face_classification {
                    Classification::Inside => {
                        new_cell_faces.push_single(*face_idx);
                    }
                    Classification::Outside => {}
                    Classification::Cut => {
                        new_cell_faces.push_single(*face_idx);
                        let face_vertices = new_faces
                            .get(*face_idx)
                            .expect("Invalid face index referenced in cell.");
                        let face_vertices = face_vertices.iter().chain(face_vertices.first());
                        for (a, b) in face_vertices.tuple_windows() {
                            // We're looking for edges that connect two intersection vertices
                            // (i.e. new vertices that result from the intersection with the plane).
                            if is_intersection_vertex(a) && is_intersection_vertex(b) {
                                intersection_edges.push(IntersectionEdge {
                                    //                                    face_idx: *face_idx,
                                    a: *a,
                                    b: *b,
                                });
                            }
                        }
                    }
                }
            }

            // At this point we know which edges that are involved in creating new faces.
            // In order to connect them together, we pick a random one and then start
            // stringing them together as long as this is possible. For each such
            // separate sequence, we generate a new face. Under non-degenerate situations,
            // only one such face should get created.
            while let Some(start_edge) = intersection_edges.pop() {
                // TODO: Are we guaranteed that we don't accidentally pick another intersection
                // edge from a face we have already visited? I *think* that this is the case,
                // but I am not sure. We *might* have to additionally keep track of
                // faces that we have picked?

                let new_face_index = new_faces.len();
                let mut new_face_vertices = new_faces.begin_array();

                new_face_vertices.push_single(start_edge.a);
                let mut next_vertex = start_edge.b;

                while let Some(next_local_idx) = intersection_edges
                    .iter()
                    .position(|edge| edge.a == next_vertex || edge.b == next_vertex)
                {
                    let next_edge = intersection_edges.swap_remove(next_local_idx);
                    if next_edge.a == next_vertex {
                        new_face_vertices.push_single(next_edge.a);
                        next_vertex = next_edge.b;
                    } else {
                        new_face_vertices.push_single(next_edge.b);
                        next_vertex = next_edge.a;
                    }
                }

                new_cell_faces.push_single(new_face_index);
            }
        }

        // At this point, we have fully constructed all faces and cells,
        // but vertices are all represented in the point-on-edge representation.
        // Moreover, we need to remove unused vertices, empty faces (faces that were
        // removed but kept as empty sets in order to retain stable indexing) and cells.
        let vertex_label_map = generate_edge_repr_vertex_labels(
            new_faces
                .iter()
                .flat_map(|face_vertices| face_vertices)
                .copied(),
        );

        // Compute new vertex coordinates where necessary
        let mut final_vertices = vec![Point3::origin(); vertex_label_map.len()];
        for (edge_rep, new_vertex_idx) in &vertex_label_map {
            let [a, b] = *edge_rep.indices();
            let vertex_coords = if a == b {
                self.vertices[a]
            } else {
                let v_a = self.vertices[a];
                let v_b = self.vertices[b];
                let segment = LineSegment3d::from_end_points(v_a, v_b);
                segment.closest_point_to_plane(&half_space.plane())
            };
            final_vertices[*new_vertex_idx] = vertex_coords;
        }

        // Convert faces from edge representation to new indices,
        // and remove and remap empty faces
        let (final_faces, face_label_map) = relabel_face_edge_representations(&new_faces, &vertex_label_map);

        // TODO: If we're a little more clever earlier on, we wouldn't have to
        // allocate a whole new storage here
        let mut final_cells = NestedVec::new();
        for cell_faces in new_cells.iter() {
            if !cell_faces.is_empty() {
                let mut new_cell_faces = final_cells.begin_array();
                for cell_face in cell_faces {
                    new_cell_faces.push_single(
                        *face_label_map
                            .get(cell_face)
                            .expect("Logic error: Face index is not in map"),
                    );
                }
            }
        }

        PolyMesh3d::from_poly_data(final_vertices, final_faces, final_cells)
    }
}

fn generate_edge_repr_vertex_labels(
    vertices_in_edge_repr: impl IntoIterator<Item = UndirectedEdge>,
) -> BTreeMap<UndirectedEdge, usize> {
    let mut map = BTreeMap::new();
    let mut iter = vertices_in_edge_repr.into_iter();

    let mut next_available_index = 0;
    while let Some(vertex) = iter.next() {
        map.entry(vertex).or_insert_with(|| {
            let idx = next_available_index;
            next_available_index += 1;
            idx
        });
    }

    map
}

/// Computes the standard index representation given edge representations of a collection
/// of faces and a label mapping for vertices.
///
/// Empty faces are removed as part of the process. A mapping is returned which
/// serves to map "old" face indices to new indices.
fn relabel_face_edge_representations(
    faces: &NestedVec<UndirectedEdge>,
    vertex_label_map: &BTreeMap<UndirectedEdge, usize>,
) -> (NestedVec<usize>, HashMap<usize, usize>) {
    let mut new_faces = NestedVec::new();
    let mut face_label_map = HashMap::new();
    let mut next_available_index = 0;
    {
        for (i, face_vertices) in faces.iter().enumerate() {
            if !face_vertices.is_empty() {
                let mut new_face_vertices = new_faces.begin_array();
                for vertex_edge_rep in face_vertices {
                    new_face_vertices.push_single(
                        *vertex_label_map
                            .get(vertex_edge_rep)
                            .expect("Logic error: Label map must map all relevant vertices."),
                    );
                }

                face_label_map.insert(i, next_available_index);
                next_available_index += 1;
            }
        }
    }

    (new_faces, face_label_map)
}

impl<T> Display for PolyMesh3d<T>
where
    T: Scalar + Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PolyMesh3d")?;
        writeln!(f, "  Vertices:")?;
        for v in self.vertices() {
            writeln!(f, "    {}", v)?;
        }
        writeln!(f)?;

        writeln!(f, "  Face vertex indices:")?;
        for (i, vertex_indices) in self.face_connectivity_iter().enumerate() {
            write!(f, "    {}: ", i)?;
            if let Some(first) = vertex_indices.first() {
                write!(f, "{}", first)?;
            }
            for idx in vertex_indices.iter().skip(1) {
                write!(f, ", {}", idx)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        writeln!(f, "  Cell face indices:")?;
        for (i, face_indices) in self.cell_connectivity_iter().enumerate() {
            write!(f, "    {}: ", i)?;
            if let Some(first) = face_indices.first() {
                write!(f, "{}", first)?;
            }
            for idx in face_indices.iter().skip(1) {
                write!(f, ", {}", idx)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

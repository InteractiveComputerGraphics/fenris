use crate::connectivity::{
    Connectivity, ConnectivityMut, Hex20Connectivity, Hex27Connectivity, Hex8Connectivity, Quad4d2Connectivity,
    Quad9d2Connectivity, Tet10Connectivity, Tet4Connectivity, Tri3d2Connectivity, Tri6d2Connectivity,
};
use crate::element::{ElementConnectivity, FiniteElement};
use crate::mesh::{HexMesh, Mesh, Mesh2d, Mesh3d, Tet4Mesh};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Point2, Point3, RealField, Scalar, U3};

use crate::geometry::polymesh::{PolyMesh, PolyMesh3d};
use crate::geometry::{OrientationTestResult, Triangle};
use fenris_nested_vec::NestedVec;
use itertools::{izip, Itertools};
use numeric_literals::replace_float_literals;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::error::Error;

pub trait RefineFrom<T, D, Connectivity>: ConnectivityMut
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    // TODO: Avoid allocating so much memory for small containers
    /// Return a refined connectivity and populate the provided containers with
    /// child indices and parent connectivity information.
    ///
    /// TODO: Explain how this works
    fn refine(
        connectivity: &Connectivity,
        mesh_vertices: &[OPoint<T, D>],
        vertices: &mut Vec<OPoint<T, D>>,
        child_indices: &mut Vec<usize>,
        parents: &mut NestedVec<usize>,
    ) -> Self;
}

impl<T> RefineFrom<T, U3, Tet4Connectivity> for Tet10Connectivity
where
    T: RealField,
    DefaultAllocator: Allocator<T, U3>,
{
    fn refine(
        connectivity: &Tet4Connectivity,
        mesh_vertices: &[Point3<T>],
        vertices: &mut Vec<Point3<T>>,
        child_indices: &mut Vec<usize>,
        parents: &mut NestedVec<usize>,
    ) -> Self {
        let global_indices = connectivity.vertex_indices();

        // Add vertex nodes
        for v_idx in global_indices {
            parents.push(&[*v_idx]);
            child_indices.push(0);
            vertices.push(mesh_vertices[*v_idx].clone());
        }

        let mut add_edge_node = |v_local_begin, v_local_end| {
            let v_global_begin = global_indices[v_local_begin];
            let v_global_end = global_indices[v_local_end];
            parents.push(&[v_global_begin, v_global_end]);
            child_indices.push(0);
            let midpoint = mesh_vertices[v_global_begin]
                .coords
                .lerp(&mesh_vertices[v_global_end].coords, T::from_f64(0.5).unwrap());
            vertices.push(midpoint.into());
        };

        add_edge_node(0, 1);
        add_edge_node(1, 2);
        add_edge_node(0, 2);
        add_edge_node(0, 3);
        add_edge_node(2, 3);
        add_edge_node(1, 3);

        Tet10Connectivity([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    }
}

impl<T> RefineFrom<T, U3, Hex8Connectivity> for Hex27Connectivity
where
    T: RealField,
    DefaultAllocator: Allocator<T, U3>,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn refine(
        connectivity: &Hex8Connectivity,
        mesh_vertices: &[Point3<T>],
        vertices: &mut Vec<Point3<T>>,
        child_indices: &mut Vec<usize>,
        parents: &mut NestedVec<usize>,
    ) -> Self {
        let global_indices = connectivity.vertex_indices();

        // Add vertex nodes
        for v_idx in global_indices {
            parents.push(&[*v_idx]);
            child_indices.push(0);
            vertices.push(mesh_vertices[*v_idx].clone());
        }

        let mut add_edge_node = |v_local_begin, v_local_end| {
            let v_global_begin = global_indices[v_local_begin];
            let v_global_end = global_indices[v_local_end];
            parents.push(&[v_global_begin, v_global_end]);
            child_indices.push(0);
            let midpoint = mesh_vertices[v_global_begin]
                .coords
                .lerp(&mesh_vertices[v_global_end].coords, 0.5);
            vertices.push(midpoint.into());
        };

        add_edge_node(0, 1);
        add_edge_node(0, 3);
        add_edge_node(0, 4);
        add_edge_node(1, 2);
        add_edge_node(1, 5);
        add_edge_node(2, 3);
        add_edge_node(2, 6);
        add_edge_node(3, 7);
        add_edge_node(4, 5);
        add_edge_node(4, 7);
        add_edge_node(5, 6);
        add_edge_node(6, 7);

        // Use element to map reference coords to physical coords
        let element = connectivity.element(mesh_vertices).unwrap();
        let mut add_face_node = |faces: &[usize], reference_vertex| {
            let vertex = element.map_reference_coords(&reference_vertex);
            child_indices.push(0);
            vertices.push(vertex.into());

            let mut array = parents.begin_array();
            for local_vertex_index in faces {
                let global_vertex_index = global_indices[*local_vertex_index];
                array.push_single(global_vertex_index);
            }
        };

        add_face_node(&[0, 1, 2, 3], Point3::new(0.0, 0.0, -1.0));
        add_face_node(&[0, 1, 4, 5], Point3::new(0.0, -1.0, 0.0));
        add_face_node(&[0, 3, 4, 7], Point3::new(-1.0, 0.0, 0.0));
        add_face_node(&[1, 2, 5, 6], Point3::new(1.0, 0.0, 0.0));
        add_face_node(&[2, 3, 6, 7], Point3::new(0.0, 1.0, 0.0));
        add_face_node(&[4, 5, 6, 7], Point3::new(0.0, 0.0, 1.0));

        // Add center node
        {
            let reference_vertex = Point3::origin();
            let vertex = element.map_reference_coords(&reference_vertex);
            parents.push(global_indices);
            child_indices.push(0);
            vertices.push(vertex.into());
        }

        // TODO: This looks a bit silly
        Hex27Connectivity([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        ])
    }
}

impl<T> RefineFrom<T, U3, Hex8Connectivity> for Hex20Connectivity
where
    T: RealField,
    DefaultAllocator: Allocator<T, U3>,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn refine(
        connectivity: &Hex8Connectivity,
        mesh_vertices: &[Point3<T>],
        vertices: &mut Vec<Point3<T>>,
        child_indices: &mut Vec<usize>,
        parents: &mut NestedVec<usize>,
    ) -> Self {
        let global_indices = connectivity.vertex_indices();

        // Add vertex nodes
        for v_idx in global_indices {
            parents.push(&[*v_idx]);
            child_indices.push(0);
            vertices.push(mesh_vertices[*v_idx].clone());
        }

        let mut add_edge_node = |v_local_begin, v_local_end| {
            let v_global_begin = global_indices[v_local_begin];
            let v_global_end = global_indices[v_local_end];
            parents.push(&[v_global_begin, v_global_end]);
            child_indices.push(0);
            let midpoint = mesh_vertices[v_global_begin]
                .coords
                .lerp(&mesh_vertices[v_global_end].coords, 0.5);
            vertices.push(midpoint.into());
        };

        add_edge_node(0, 1);
        add_edge_node(0, 3);
        add_edge_node(0, 4);
        add_edge_node(1, 2);
        add_edge_node(1, 5);
        add_edge_node(2, 3);
        add_edge_node(2, 6);
        add_edge_node(3, 7);
        add_edge_node(4, 5);
        add_edge_node(4, 7);
        add_edge_node(5, 6);
        add_edge_node(6, 7);

        // TODO: This looks a bit silly
        Hex20Connectivity([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    }
}

/// Stop-gap solution for generalizing mesh conversion.
///
/// TODO: Remove this trait and use RefineFrom directly? Though this is not directly possible
/// due to conflicting impls. At least find a better name though!
pub trait FromTemp<T> {
    fn from(obj: T) -> Self;
}

impl<'a, T, D, C, CNew> FromTemp<&'a Mesh<T, D, C>> for Mesh<T, D, CNew>
where
    T: RealField,
    D: DimName,
    C: Connectivity,
    CNew: RefineFrom<T, D, C>,
    DefaultAllocator: Allocator<T, D>,
{
    fn from(mesh: &'a Mesh<T, D, C>) -> Self {
        // Workspaces are used to hold only per-connectivity information,
        // which is later transformed to global data
        let mut child_indices_workspace = Vec::new();
        let mut parents_workspace = NestedVec::new();
        let mut vertices_workspace = Vec::new();

        // Global intermediate data
        let mut child_indices = Vec::new();
        let mut parents = NestedVec::new();
        let mut intermediate_vertices = Vec::new();

        let mut new_connectivity = Vec::new();

        // First construct intermediate data. This means that we basically just lay out the
        // data from each local connectivity refinement linearly. Then in the next step,
        // we label vertices, making sure to label equivalent vertices with the same index,
        // before finally reconstructing connectivity.
        for conn in mesh.connectivity() {
            child_indices_workspace.clear();
            parents_workspace.clear();
            vertices_workspace.clear();

            let mut new_conn = CNew::refine(
                conn,
                mesh.vertices(),
                &mut vertices_workspace,
                &mut child_indices_workspace,
                &mut parents_workspace,
            );

            assert_eq!(
                child_indices_workspace.len(),
                parents_workspace.len(),
                "Invalid RefineFrom implementation: \
                       Number of child indices and parent groups must be equal."
            );
            assert_eq!(
                child_indices_workspace.len(),
                vertices_workspace.len(),
                "Invalid RefineFrom implementation: \
                       Number of child indices and vertices must be equal."
            );

            let intermediate_vertex_index_offset = child_indices.len();

            intermediate_vertices.extend_from_slice(&vertices_workspace);
            child_indices.extend_from_slice(&child_indices_workspace);
            for parent_group in parents_workspace.iter() {
                parents.push(parent_group);
                // TODO: Sort here or in impl? Might as well do it here I guess?
                parents.last_mut().unwrap().sort_unstable();
            }

            // Vertex indices are local with respect to the returned new vertices.
            // By adding the offset, the connectivity holds global intermediate indices
            for v_idx in new_conn.vertex_indices_mut() {
                *v_idx += intermediate_vertex_index_offset;
            }

            new_connectivity.push(new_conn);
        }

        // Map (child index, parents) to final vertex index
        let mut vertex_label_map = FxHashMap::default();
        let mut final_vertices = Vec::new();
        let mut next_available_vertex_index = 0;

        // Rewrite connectivity and label vertices, making sure to collect
        // vertices with the same child index and parents under the same label
        for conn in &mut new_connectivity {
            for vertex_index in conn.vertex_indices_mut() {
                let vertex_parents = parents.get(*vertex_index).unwrap();
                let vertex_child_index = child_indices[*vertex_index];

                // TODO: Avoid double lookup
                let key = (vertex_child_index, vertex_parents);
                let final_vertex_index = if vertex_label_map.contains_key(&key) {
                    *vertex_label_map.get(&key).unwrap()
                } else {
                    let vertex = intermediate_vertices[*vertex_index].clone();
                    final_vertices.push(vertex);

                    let final_index = next_available_vertex_index;
                    vertex_label_map.insert((vertex_child_index, vertex_parents), final_index);
                    next_available_vertex_index += 1;
                    final_index
                };

                *vertex_index = final_vertex_index;
            }
        }

        Mesh::from_vertices_and_connectivity(final_vertices, new_connectivity)
    }
}

impl<T> From<Mesh2d<T, Tri3d2Connectivity>> for Mesh2d<T, Tri6d2Connectivity>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn from(initial_mesh: Mesh2d<T, Tri3d2Connectivity>) -> Self {
        let mut vertices = initial_mesh.vertices().to_vec();

        // Holds edges on which vertices should be inserted
        let mut edge_vertex_index_map = HashMap::new();

        let mut new_connectivity = Vec::new();

        for connectivity in initial_mesh.connectivity() {
            // TODO: Find a nicer way to write this
            let vertex_indices = connectivity.vertex_indices();
            let num_vertices = vertex_indices.len();
            let edges = vertex_indices
                .iter()
                .cycle()
                .take(num_vertices + 1)
                .tuple_windows();

            // Add nodal vertices
            let mut tri6_vertex_indices = [0usize; 6];
            for (i, index) in vertex_indices.iter().enumerate() {
                tri6_vertex_indices[i] = *index;
            }

            // Add vertices that are midpoints on edges
            for ((a, b), vertex_index) in izip!(edges, &mut tri6_vertex_indices[3..]) {
                // Sort the tuple so that edges are uniquely described
                let edge = (a.min(b), a.max(b));

                let index = edge_vertex_index_map.entry(edge).or_insert_with(|| {
                    let new_vertex_index = vertices.len();
                    let (v_a, v_b) = (vertices[*a], vertices[*b]);
                    let midpoint = Point2::from((v_a.coords + v_b.coords) / 2.0);
                    vertices.push(midpoint);
                    new_vertex_index
                });

                *vertex_index = *index;
            }

            // Finally add the new p-refined connectivity
            new_connectivity.push(Tri6d2Connectivity(tri6_vertex_indices));
        }

        Mesh2d::from_vertices_and_connectivity(vertices, new_connectivity)
    }
}

impl<T> From<Mesh2d<T, Quad4d2Connectivity>> for Mesh2d<T, Quad9d2Connectivity>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn from(initial_mesh: Mesh2d<T, Quad4d2Connectivity>) -> Self {
        let mut vertices = initial_mesh.vertices().to_vec();

        // Holds edges on which vertices should be inserted
        let mut edge_vertex_index_map = HashMap::new();

        let mut new_connectivity = Vec::new();

        for connectivity in initial_mesh.connectivity() {
            // TODO: Find a nicer way to write this
            let vertex_indices = connectivity.vertex_indices();
            let num_vertices = vertex_indices.len();
            let edges = vertex_indices
                .iter()
                .cycle()
                .take(num_vertices + 1)
                .tuple_windows();

            // Add nodal vertices
            let mut quad9_vertex_indices = [0usize; 9];
            for (i, index) in vertex_indices.iter().enumerate() {
                quad9_vertex_indices[i] = *index;
            }

            // Add vertices that are midpoints on edges
            for ((a, b), vertex_index) in izip!(edges, &mut quad9_vertex_indices[4..]) {
                // Sort the tuple so that edges are uniquely described
                let edge = (a.min(b), a.max(b));

                let index = edge_vertex_index_map.entry(edge).or_insert_with(|| {
                    let new_vertex_index = vertices.len();
                    let (v_a, v_b) = (vertices[*a], vertices[*b]);
                    let midpoint = Point2::from((v_a.coords + v_b.coords) / 2.0);
                    vertices.push(midpoint);
                    new_vertex_index
                });

                *vertex_index = *index;
            }

            // Add the midpoint of the cell
            let element = connectivity.element(initial_mesh.vertices()).unwrap();
            let midpoint = Point2::from(element.map_reference_coords(&Point2::origin()));
            quad9_vertex_indices[8] = vertices.len();
            vertices.push(midpoint);

            // Finally add the new p-refined connectivity
            new_connectivity.push(Quad9d2Connectivity(quad9_vertex_indices));
        }

        Mesh2d::from_vertices_and_connectivity(vertices, new_connectivity)
    }
}

impl<'a, T> From<&'a Mesh3d<T, Tet4Connectivity>> for Mesh3d<T, Tet10Connectivity>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn from(initial_mesh: &'a Mesh3d<T, Tet4Connectivity>) -> Self {
        <Self as FromTemp<_>>::from(initial_mesh)
    }
}

impl<'a, T> From<&'a Mesh3d<T, Tet10Connectivity>> for Mesh3d<T, Tet4Connectivity>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn from(initial_mesh: &'a Mesh3d<T, Tet10Connectivity>) -> Self {
        let mut new_connectivity = Vec::with_capacity(initial_mesh.connectivity().len());

        for conn in initial_mesh.connectivity() {
            let tet4_conn = Tet4Connectivity::from(conn);
            new_connectivity.push(tet4_conn);
        }

        let tet4_mesh = Mesh::from_vertices_and_connectivity(initial_mesh.vertices().to_vec(), new_connectivity);
        tet4_mesh.keep_cells(&((0..tet4_mesh.connectivity().len()).collect::<Vec<_>>()))
    }
}

impl<'a, T> From<&'a Mesh3d<T, Hex8Connectivity>> for Mesh3d<T, Hex27Connectivity>
where
    T: RealField,
{
    fn from(initial_mesh: &'a Mesh3d<T, Hex8Connectivity>) -> Self {
        <Self as FromTemp<_>>::from(initial_mesh)
    }
}

impl<'a, T> From<&'a Mesh3d<T, Hex8Connectivity>> for Mesh3d<T, Hex20Connectivity>
where
    T: RealField,
{
    fn from(initial_mesh: &'a Mesh3d<T, Hex8Connectivity>) -> Self {
        <Self as FromTemp<_>>::from(initial_mesh)
    }
}

impl<'a, T> From<&'a HexMesh<T>> for Tet4Mesh<T>
where
    T: RealField,
{
    fn from(hex_mesh: &'a HexMesh<T>) -> Self {
        // TODO: Provide a "direct" method that does not rely on triangulation
        // (then we could also reduce the `RealField` bound to `Scalar`)
        let poly_mesh = PolyMesh3d::from(hex_mesh)
            .triangulate()
            .expect("Must be able to triangulate hex mesh");
        Tet4Mesh::try_from(&poly_mesh).expect("Must be able to convert triangulated mesh into TetMesh")
    }
}

impl<'a, T, D, C> From<&'a Mesh<T, D, C>> for PolyMesh<T, D>
where
    T: Scalar,
    D: DimName,
    // TODO: Should somehow ensure that the face connectivity describes
    // a counter-clockwise oriented polygon
    C: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
    fn from(mesh: &'a Mesh<T, D, C>) -> Self {
        let vertices = mesh.vertices().to_vec();

        // We are able to query each cell for its faces, but these faces are only described
        // in terms of the vertices they connect. We consider any two faces with the same
        // set of vertex indices to be the same face. We must moreover be careful to
        // always preserve the order of the vertex indices for whatever face we choose
        // to add, as they need to remain counter-clockwise. This also preserves the correct
        // normal direction for boundary faces.

        let mut sorted_connectivities = NestedVec::new();
        for c in mesh.connectivity() {
            for i in 0..c.num_faces() {
                let face_conn = c.get_face_connectivity(i).unwrap();
                sorted_connectivities.push(face_conn.vertex_indices());
                sorted_connectivities.last_mut().unwrap().sort_unstable();
            }
        }

        // Map face connectivities to face indices
        let mut conn_map = HashMap::new();
        let mut faces = NestedVec::new();
        let mut cells = NestedVec::new();

        let mut global_cell_face_index = 0;
        for c in mesh.connectivity() {
            let mut cell_array = cells.begin_array();
            for i in 0..c.num_faces() {
                let face_conn = c.get_face_connectivity(i).unwrap();
                let sorted_conn = sorted_connectivities.get(global_cell_face_index).unwrap();

                // TODO: Use entry to avoid double lookup
                let face_idx = if let Some(face_idx) = conn_map.get(sorted_conn) {
                    *face_idx
                } else {
                    let face_idx = faces.len();
                    faces.push(face_conn.vertex_indices());
                    conn_map.insert(sorted_conn, face_idx);
                    face_idx
                };

                cell_array.push_single(face_idx);
                global_cell_face_index += 1;
            }
        }

        assert_eq!(cells.len(), mesh.connectivity().len());

        PolyMesh::from_poly_data(vertices, faces, cells)
    }
}

impl<'a, T> TryFrom<&'a PolyMesh3d<T>> for Tet4Mesh<T>
where
    T: RealField,
{
    // TODO: Proper Error type
    type Error = Box<dyn Error>;

    fn try_from(poly_mesh: &'a PolyMesh3d<T>) -> Result<Self, Self::Error> {
        let mut connectivity = Vec::new();

        let get_face = |idx| {
            poly_mesh
                .get_face_connectivity(idx)
                .expect("Logic error: Cell references non-existent face.")
        };
        let get_vertex = |idx| {
            poly_mesh
                .vertices()
                .get(idx)
                .expect("Logic error: Face references non-existent vertex.")
        };

        for (cell_idx, cell) in poly_mesh.cell_connectivity_iter().enumerate() {
            if cell.len() == 4 {
                // We construct a tetrahedron by taking the first face
                // (which becomes the base of the tetrahedron) and then
                // finding the remaining apex vertex which is not referenced by the other faces,
                // and finally connecting the base vertices to the apex.
                let mut base_vertices = [0; 3];
                let base_face = get_face(cell[0]);
                base_vertices.clone_from_slice(base_face);

                // Each remaining face should consist of two vertices from the base face,
                // and one vertex not in the base face
                let apex = get_face(cell[1])
                    .iter()
                    .filter(|idx| !base_vertices.contains(idx))
                    .next()
                    .ok_or_else(|| {
                        format!(
                            "Failure to convert: \
                             Detected several faces with the same set of vertices in cell {}.",
                            cell_idx
                        )
                    })?;

                for i in 1..4 {
                    let face = get_face(cell[i]);
                    let has_no_extra_vertices = face
                        .iter()
                        .all(|idx| base_vertices.contains(idx) || apex == idx);

                    if !has_no_extra_vertices {
                        return Err(Box::from(format!(
                            "Failure to convert: The faces of cell {} do not form a \
                             tetrahedral cell.",
                            cell_idx
                        )));
                    }
                }

                let base_tri = Triangle([
                    *get_vertex(base_vertices[0]),
                    *get_vertex(base_vertices[1]),
                    *get_vertex(base_vertices[2]),
                ]);
                let apex_vertex = *get_vertex(*apex);

                // If the apex is "below" the triangle, flip the normal of the triangle
                // by swapping some of its vertices
                if base_tri.point_orientation(&apex_vertex) == OrientationTestResult::Negative {
                    base_vertices.swap(0, 1);
                }

                // Now we know that the apex is "above" the triangle, in the sense that it is
                // on the "non-negative" side of the triangle with respect to the normal.
                // Then it only remains to connect the base to the apex.
                let mut tet4_vertex_indices = [0; 4];
                tet4_vertex_indices[0..3].clone_from_slice(&base_vertices);
                tet4_vertex_indices[3] = *apex;
                connectivity.push(Tet4Connectivity(tet4_vertex_indices));
            } else {
                return Err(Box::from("Failure to convert: Detected non-tetrahedral cell."));
            }
        }

        Ok(Mesh::from_vertices_and_connectivity(
            poly_mesh.vertices().to_vec(),
            connectivity,
        ))
    }
}

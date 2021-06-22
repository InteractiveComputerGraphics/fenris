use fenris::connectivity::{
    CellConnectivity, Connectivity, Quad9d2Connectivity, Tri3d2Connectivity,
};
use fenris::geometry::polymesh::PolyMesh;
use fenris::geometry::{Orientation, Triangle};
use fenris::mesh::procedural::{
    create_rectangular_uniform_hex_mesh, create_rectangular_uniform_quad_mesh_2d,
    create_unit_square_uniform_quad_mesh_2d,
};
use fenris::mesh::{Mesh, Mesh2d};
use fenris::proptest::rectangular_uniform_mesh_strategy;
use itertools::{equal, sorted, Itertools};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Point2, Scalar, Vector2};
use proptest::collection::vec;
use proptest::prelude::*;
use std::cmp::max;

#[test]
fn quad4_find_boundary_faces() {
    // Single quad
    {
        let mesh = create_unit_square_uniform_quad_mesh_2d::<f64>(1);
        let boundary_faces = mesh.find_boundary_faces();

        let cells: Vec<_> = boundary_faces
            .iter()
            .cloned()
            .map(|(_, cell, _)| cell)
            .collect();
        let mut local_indices: Vec<_> = boundary_faces
            .iter()
            .cloned()
            .map(|(_, _, idx)| idx)
            .collect();
        local_indices.sort();

        assert_eq!(cells, [0, 0, 0, 0]);
        assert_eq!(local_indices, [0, 1, 2, 3]);
    }
}

#[test]
fn quad9_find_boundary_vertices() {
    {
        // Single element
        let mesh: Mesh2d<f64, Quad9d2Connectivity> =
            create_unit_square_uniform_quad_mesh_2d(1).into();
        let boundary_vertex_indices = mesh.find_boundary_vertices();

        assert_eq!(boundary_vertex_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    {
        // Two elements
        let vertices = vec![
            Point2::new(0.0, 0.0), // 0
            Point2::new(1.0, 0.0), // 1
            Point2::new(2.0, 0.0), // 2
            Point2::new(0.0, 1.0), // 3
            Point2::new(1.0, 1.0), // 4
            Point2::new(0.0, 1.0), // 5
            Point2::new(0.5, 0.0), // 6
            Point2::new(1.5, 0.0), // 7
            Point2::new(2.0, 0.5), // 8
            Point2::new(1.5, 0.5), // 9
            Point2::new(1.0, 0.5), // 10
            Point2::new(0.5, 0.5), // 11
            Point2::new(0.0, 0.5), // 12
            Point2::new(0.5, 1.0), // 13
            Point2::new(1.5, 1.0), // 14
        ];
        let connectivity = vec![
            Quad9d2Connectivity([0, 1, 4, 5, 6, 10, 13, 12, 11]),
            Quad9d2Connectivity([1, 2, 3, 4, 7, 8, 14, 10, 9]),
        ];
        let mesh = Mesh2d::from_vertices_and_connectivity(vertices, connectivity);
        let boundary_vertex_indices = mesh.find_boundary_vertices();

        assert_eq!(
            boundary_vertex_indices,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]
        );
    }
}

#[test]
fn winding_order() {
    let a = Point2::new(2.0, 1.0);
    let b = Point2::new(3.0, 2.0);
    let c = Point2::new(0.0, 3.0);

    use Orientation::{Clockwise, Counterclockwise};

    assert_eq!(Triangle([a, b, c]).orientation(), Clockwise);
    assert_eq!(Triangle([b, c, a]).orientation(), Clockwise);
    assert_eq!(Triangle([c, a, b]).orientation(), Clockwise);
    assert_eq!(Triangle([a, c, b]).orientation(), Counterclockwise);
    assert_eq!(Triangle([b, a, c]).orientation(), Counterclockwise);
    assert_eq!(Triangle([c, b, a]).orientation(), Counterclockwise);

    let mut triangle = Triangle([a, b, c]);
    triangle.swap_vertices(0, 1);
    assert_eq!(triangle.orientation(), Counterclockwise);
    triangle.swap_vertices(2, 1);
    assert_eq!(triangle.orientation(), Clockwise);
    triangle.swap_vertices(2, 0);
    assert_eq!(triangle.orientation(), Counterclockwise);

    // No-op swaps don't change winding
    triangle.swap_vertices(0, 0);
    assert_eq!(triangle.orientation(), Counterclockwise);
    triangle.swap_vertices(1, 1);
    assert_eq!(triangle.orientation(), Counterclockwise);
    triangle.swap_vertices(2, 2);
    assert_eq!(triangle.orientation(), Counterclockwise);
}

fn verify_connected_poly_mesh<T, D, C>(original_mesh: &Mesh<T, D, C>, poly_mesh: &PolyMesh<T, D>)
where
    T: Scalar,
    D: DimName,
    C: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
    assert_eq!(original_mesh.vertices(), poly_mesh.vertices());
    assert_eq!(original_mesh.connectivity().len(), poly_mesh.num_cells());

    let face_cell_connectivity = poly_mesh.compute_face_cell_connectivity();
    assert!(
        face_cell_connectivity
            .iter()
            .all(|cells| [1, 2].contains(&cells.len())),
        "Every face must be connected to either 1 or 2 cells."
    );

    let mut poly_faces_referenced = vec![false; poly_mesh.num_faces()];

    // Verify that every cell face is contained exactly once in face connectivities of the polymesh
    for cell in original_mesh.connectivity() {
        let num_cell_faces = cell.num_faces();
        for i in 0..num_cell_faces {
            let face = cell.get_face_connectivity(i).unwrap();

            let topologically_equivalent_poly_faces: Vec<_> = poly_mesh
                .face_connectivity_iter()
                .enumerate()
                .filter(|(_, vertex_indices)| {
                    equal(
                        sorted(vertex_indices.to_vec()),
                        sorted(face.vertex_indices().to_vec()),
                    )
                })
                .map(|(i, _)| i)
                .collect();

            assert_eq!(
                topologically_equivalent_poly_faces.len(),
                1,
                "Every cell face must be present exactly once in the polymesh face connectivity"
            );

            let referenced_face_idx = topologically_equivalent_poly_faces.first().unwrap();
            poly_faces_referenced[*referenced_face_idx] = true;
        }
    }

    assert!(
        poly_faces_referenced.iter().all(|referenced| *referenced),
        "All faces in poly mesh must be equivalent to at least one local cell face \
         in the original mesh."
    );
}

#[test]
fn convert_mesh_to_poly_mesh() {
    {
        // Single element 2D triangle mesh

        // The exact coordinates should not matter as the conversion is an entirely
        // topological conversion
        let vertices = vec![Point2::<i32>::origin(); 3];
        let connectivity = vec![Tri3d2Connectivity([0, 1, 2])];
        let mesh = Mesh::from_vertices_and_connectivity(vertices.clone(), connectivity);
        let polymesh = PolyMesh::from(&mesh);

        assert_eq!(polymesh.vertices(), vertices.as_slice());
        assert_eq!(polymesh.num_cells(), 1);
        assert_eq!(polymesh.num_faces(), 3);

        let cell0_sorted_indices = polymesh
            .get_cell_connectivity(0)
            .unwrap()
            .iter()
            .cloned()
            .sorted();
        assert_eq!(cell0_sorted_indices.as_slice(), &[0, 1, 2]);

        for face_conn in &[[0, 1], [1, 2], [2, 0]] {
            let count_equal = polymesh
                .face_connectivity_iter()
                .filter(|poly_face_conn| poly_face_conn == face_conn)
                .count();
            assert_eq!(count_equal, 1);
        }

        verify_connected_poly_mesh(&mesh, &polymesh);
    }

    {
        // Two element 2D triangle mesh

        // The exact coordinates should not matter as the conversion is an entirely
        // topological conversion
        let vertices = vec![Point2::<i32>::origin(); 4];
        let connectivity = vec![Tri3d2Connectivity([0, 1, 2]), Tri3d2Connectivity([2, 3, 0])];
        let mesh = Mesh::from_vertices_and_connectivity(vertices.clone(), connectivity);
        let polymesh = PolyMesh::from(&mesh);

        assert_eq!(polymesh.vertices(), vertices.as_slice());
        assert_eq!(polymesh.num_cells(), 2);
        assert_eq!(polymesh.num_faces(), 5);

        verify_connected_poly_mesh(&mesh, &polymesh);
    }

    {
        // 2D quad mesh
        let mesh = create_rectangular_uniform_quad_mesh_2d(1.0, 3, 4, 1, &Vector2::zeros());
        let polymesh = PolyMesh::from(&mesh);

        verify_connected_poly_mesh(&mesh, &polymesh);
    }

    {
        // 2D triangle mesh
        let mesh = create_rectangular_uniform_quad_mesh_2d(1.0, 3, 4, 1, &Vector2::zeros());
        let mesh = mesh.split_into_triangles();
        let polymesh = PolyMesh::from(&mesh);

        verify_connected_poly_mesh(&mesh, &polymesh);
    }

    {
        // 3D hex mesh, single hexahedron
        let hex_mesh = create_rectangular_uniform_hex_mesh(1.0, 1, 1, 1, 1);
        let polymesh = PolyMesh::from(&hex_mesh);

        verify_connected_poly_mesh(&hex_mesh, &polymesh);
    }

    {
        // 3D hex mesh
        let hex_mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 1, 1);
        let polymesh = PolyMesh::from(&hex_mesh);

        verify_connected_poly_mesh(&hex_mesh, &polymesh);
    }

    {
        // 3D hex mesh, bigger/more complex
        let hex_mesh = create_rectangular_uniform_hex_mesh(1.0, 3, 2, 4, 1);
        let polymesh = PolyMesh::from(&hex_mesh);

        verify_connected_poly_mesh(&hex_mesh, &polymesh);
    }
}

proptest! {
    #[test]
    fn keep_cells_correctly_preserves_cells_for_quad_mesh(
        (mesh, cell_indices) in rectangular_uniform_mesh_strategy(1.0, 4)
            // For each mesh, generate a strategy that outputs the mesh itself
            // and a collection of cell indices which are in bounds with respect to the mesh
            .prop_flat_map(
                |mesh| {
                    let vec_size_range = (0, max(1, 2 * mesh.connectivity().len()));
                    let vec_element_strategy = 0..max(1, mesh.connectivity().len());
                    (Just(mesh), vec(vec_element_strategy, vec_size_range))
                }
            )
    ) {
        // For every mesh and an appropriate selection of cells to keep,
        // it must hold that the kept cells are equivalent in the sense that
        // the points that define the cell are the same.
        let new_mesh = mesh.keep_cells(&cell_indices);

        let kept_quads_from_old_mesh: Vec<_> = cell_indices
            .iter()
            .cloned()
            .map(|old_cell_index| mesh.connectivity()[old_cell_index])
            .map(|connectivity| connectivity.cell(mesh.vertices()).unwrap())
            .collect();

        let kept_quads_from_new_mesh: Vec<_> = (0 .. new_mesh.connectivity().len())
            .into_iter()
            .map(|new_cell_index| new_mesh.connectivity()[new_cell_index])
            .map(|connectivity| connectivity.cell(new_mesh.vertices()).unwrap())
            .collect();

        prop_assert_eq!(kept_quads_from_old_mesh, kept_quads_from_new_mesh);
    }
}

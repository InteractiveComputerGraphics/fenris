use fenris::geometry::polymesh::PolyMesh3d;
use fenris::geometry::{ConvexPolyhedron, HalfSpace, Tetrahedron};
use fenris::mesh::Tet4Mesh;
use fenris::procedural::create_rectangular_uniform_hex_mesh;
use fenris::vtk::write_vtk;
use nalgebra::{Point3, Unit, Vector3};
use nested_vec::NestedVec;
use std::convert::TryFrom;
use std::error::Error;

fn tetrahedron_vertices() -> [Point3<f64>; 4] {
    [
        Point3::new(0.1, 0.1, 0.1),
        Point3::new(0.9, 0.1, 0.1),
        Point3::new(0.1, 0.9, 0.1),
        Point3::new(0.1, 0.1, 0.9),
    ]
}

fn tetrahedron() -> PolyMesh3d<f64> {
    let vertices = tetrahedron_vertices().to_vec();
    let faces = vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![0, 3, 2]];
    let cells = vec![vec![0, 1, 2, 3]];
    PolyMesh3d::from_poly_data(vertices, NestedVec::from(&faces), NestedVec::from(&cells))
}

fn intersect_meshes_with_single_half_space() -> Result<(), Box<dyn Error>> {
    let cube = create_rectangular_uniform_hex_mesh(1.0, 2, 1, 1, 1);
    let cube = PolyMesh3d::from(&cube);

    // Intersect meshes with half space
    let meshes = vec![("cube", cube), ("tetrahedron", tetrahedron())];

    for (name, mesh) in meshes {
        let half_space = HalfSpace::from_point_and_normal(
            Point3::new(0.0, 0.0, 0.3),
            Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
        );

        let intersection = mesh.intersect_half_space(&half_space);

        let base_path = "data/polymesh_intersection/halfspace";
        let original_mesh_file_name = format!("{}/{}.vtk", base_path, name);
        let intersection_file_name = format!("{}/{}_intersection.vtk", base_path, name);

        write_vtk(&mesh, original_mesh_file_name, "polymesh intersection")?;
        write_vtk(
            &intersection,
            intersection_file_name,
            "polymesh intersection",
        )?;

        println!("Original mesh: {}", mesh);
        println!("Intersection: {}", intersection);
    }
    Ok(())
}

fn intersect_polyhedron_with_mesh<'a>(
    mesh_name: &str,
    polyhedron_name: &str,
    polyhedron: &impl ConvexPolyhedron<'a, f64>,
    mesh: &PolyMesh3d<f64>,
) -> Result<(), Box<dyn Error>> {
    let intersection = mesh.intersect_convex_polyhedron(polyhedron);

    let base_path = "data/polymesh_intersection/polyhedra";
    let original_mesh_file_name = format!("{}/{}.vtk", base_path, mesh_name);
    let intersection_file_name = format!(
        "{}/{}_{}_intersection.vtk",
        base_path, mesh_name, polyhedron_name
    );
    let tet_mesh_file_name = format!(
        "{}/{}_{}_intersection_tet_mesh.vtk",
        base_path, mesh_name, polyhedron_name
    );

    write_vtk(mesh, original_mesh_file_name, "polymesh intersection")?;
    write_vtk(
        &intersection,
        intersection_file_name,
        "polymesh intersection",
    )?;

    println!("Original mesh: {}", mesh);
    println!("Intersection: {}", intersection);

    let tet_mesh = Tet4Mesh::try_from(&intersection.triangulate()?)?;
    write_vtk(&tet_mesh, tet_mesh_file_name, "tet mesh")?;

    Ok(())
}

fn intersect_meshes_with_polyhedra() -> Result<(), Box<dyn Error>> {
    let cube = create_rectangular_uniform_hex_mesh(1.0, 1, 1, 1, 2);
    let cube = PolyMesh3d::from(&cube);

    // Intersect meshes with half space
    let meshes = vec![("cube", cube), ("tetrahedron", tetrahedron())];

    {
        let tet = Tetrahedron::from_vertices(tetrahedron_vertices());
        let name = "tet";
        for (mesh_name, mesh) in &meshes {
            intersect_polyhedron_with_mesh(mesh_name, name, &tet, mesh)?;
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    intersect_meshes_with_single_half_space()?;
    intersect_meshes_with_polyhedra()?;

    Ok(())
}

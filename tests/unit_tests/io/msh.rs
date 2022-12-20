use crate::export_mesh_vtk;
use fenris::connectivity::{
    Hex27Connectivity, Hex8Connectivity, Quad4d2Connectivity, Quad9d2Connectivity, Tet10Connectivity, Tet4Connectivity,
    Tri3d2Connectivity, Tri3d3Connectivity, Tri6d2Connectivity,
};
use fenris::io::msh::load_msh_from_file;
use insta::assert_debug_snapshot;
use nalgebra::{U2, U3};

#[test]
fn load_msh_sphere_tet4_large() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U3, Tet4Connectivity, _>("assets/meshes/sphere_tet4_593.msh")?;

    assert_eq!(mesh.vertices().len(), 183);
    assert_eq!(mesh.connectivity().len(), 593);

    export_mesh_vtk("io_msh", "load_msh_sphere_tet4_large", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_rect_tri3d2_large() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U2, Tri3d2Connectivity, _>("assets/meshes/rectangle_tri3_110.msh")?;

    assert_eq!(mesh.vertices().len(), 70);
    assert_eq!(mesh.connectivity().len(), 110);

    export_mesh_vtk("io_msh", "load_msh_rect_tri3d2_large", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_rect_tri3d3_large() -> eyre::Result<()> {
    // Loading a 2D triangle mesh to a 3D triangle mesh should work
    let mesh = load_msh_from_file::<f64, U3, Tri3d3Connectivity, _>("assets/meshes/rectangle_tri3_110.msh")?;

    assert_eq!(mesh.vertices().len(), 70);
    assert_eq!(mesh.connectivity().len(), 110);

    export_mesh_vtk("io_msh", "load_msh_rect_tri3d3_large", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_square_quad4d2_large() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U2, Quad4d2Connectivity, _>("assets/meshes/square_quad4_79.msh")?;

    assert_eq!(mesh.vertices().len(), 96);
    assert_eq!(mesh.connectivity().len(), 79);

    export_mesh_vtk("io_msh", "load_msh_square_quad4d2_large", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_square_tri3d2() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U2, Tri3d2Connectivity, _>("assets/meshes/square_tri3_4.msh")?;

    assert_eq!(mesh.vertices().len(), 5);
    assert_eq!(mesh.connectivity().len(), 4);

    export_mesh_vtk("io_msh", "load_msh_square_tri3d2", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_square_tri6d2() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U2, Tri6d2Connectivity, _>("assets/meshes/square_tri6_4.msh")?;

    assert_eq!(mesh.vertices().len(), 13);
    assert_eq!(mesh.connectivity().len(), 4);

    export_mesh_vtk("io_msh", "load_msh_square_tri6d2", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_square_quad4d2() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U2, Quad4d2Connectivity, _>("assets/meshes/square_quad4_4.msh")?;

    assert_eq!(mesh.vertices().len(), 9);
    assert_eq!(mesh.connectivity().len(), 4);

    export_mesh_vtk("io_msh", "load_msh_square_quad4d2", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_square_quad9d2() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U2, Quad9d2Connectivity, _>("assets/meshes/square_quad9_4.msh")?;

    assert_eq!(mesh.vertices().len(), 25);
    assert_eq!(mesh.connectivity().len(), 4);

    export_mesh_vtk("io_msh", "load_msh_square_quad9d2", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_cube_tet4() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U3, Tet4Connectivity, _>("assets/meshes/cube_tet4_24.msh")?;

    assert_eq!(mesh.vertices().len(), 14);
    assert_eq!(mesh.connectivity().len(), 24);

    export_mesh_vtk("io_msh", "load_msh_cube_tet4", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_cube_tet10() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U3, Tet10Connectivity, _>("assets/meshes/cube_tet10_24.msh")?;

    assert_eq!(mesh.vertices().len(), 63);
    assert_eq!(mesh.connectivity().len(), 24);

    export_mesh_vtk("io_msh", "load_msh_cube_tet10", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_cube_hex8() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U3, Hex8Connectivity, _>("assets/meshes/cube_hex8_8.msh")?;

    assert_eq!(mesh.vertices().len(), 27);
    assert_eq!(mesh.connectivity().len(), 8);

    export_mesh_vtk("io_msh", "load_msh_cube_hex8", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

#[test]
fn load_msh_cube_hex27() -> eyre::Result<()> {
    let mesh = load_msh_from_file::<f64, U3, Hex27Connectivity, _>("assets/meshes/cube_hex27_8.msh")?;

    assert_eq!(mesh.vertices().len(), 125);
    assert_eq!(mesh.connectivity().len(), 8);

    export_mesh_vtk("io_msh", "load_msh_cube_hex27", &mesh);
    assert_debug_snapshot!(mesh);

    Ok(())
}

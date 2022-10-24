use fenris::connectivity::Tri3d2Connectivity;
use fenris::io::vtk::FiniteElementMeshDataSetBuilder;
use fenris::mesh::refinement::{refine_uniformly, refine_uniformly_repeat};
use fenris::mesh::{Mesh, TriangleMesh2d};
use insta::assert_debug_snapshot;
use nalgebra::point;
use std::path::Path;

fn export_mesh_vtk(test_name: &str, file_stem: &str, mesh: &TriangleMesh2d<f64>) {
    let output_path = Path::new("data/unit_tests/")
        .join(test_name)
        .join(format!("{file_stem}.vtu"));
    FiniteElementMeshDataSetBuilder::from_mesh(mesh)
        .try_export(output_path)
        .expect("Export failure is a test failure")
}

#[test]
fn uniform_refinement_tri3d2() {
    let mesh = {
        let vertices = vec![
            point![0.0, 0.0],
            point![1.0, 0.0],
            point![2.0, -1.0],
            point![2.5, 1.5],
            point![1.2, 1.0],
            point![0.0, 1.3],
        ];
        let cells = vec![
            Tri3d2Connectivity([0, 1, 5]),
            Tri3d2Connectivity([1, 2, 3]),
            Tri3d2Connectivity([3, 4, 1]),
            Tri3d2Connectivity([1, 4, 5]),
        ];
        Mesh::from_vertices_and_connectivity(vertices, cells)
    };
    let refined_once = refine_uniformly(&mesh);
    let refined0 = refine_uniformly_repeat(&mesh, 0);
    let refined1 = refine_uniformly_repeat(&mesh, 1);
    let refined2 = refine_uniformly_repeat(&mesh, 2);
    export_mesh_vtk("uniform_refinement_tri3d2", "mesh", &mesh);
    export_mesh_vtk("uniform_refinement_tri3d2", "refined_once", &refined_once);
    export_mesh_vtk("uniform_refinement_tri3d2", "refined0", &refined0);
    export_mesh_vtk("uniform_refinement_tri3d2", "refined1", &refined1);
    export_mesh_vtk("uniform_refinement_tri3d2", "refined2", &refined2);
    assert_debug_snapshot!(refined_once);
    assert_debug_snapshot!(refined0);
    assert_debug_snapshot!(refined1);
    assert_debug_snapshot!(refined2);
}

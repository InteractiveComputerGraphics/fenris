use fenris::io::vtk::{FiniteElementMeshDataSetBuilder, VtkCellConnectivity};
use fenris::mesh::Mesh;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName};
use std::path::Path;

mod unit_tests;

fn export_mesh_vtk<D, C>(test_name: &str, file_stem: &str, mesh: &Mesh<f64, D, C>)
where
    D: DimName,
    DefaultAllocator: Allocator<f64, D>,
    C: VtkCellConnectivity,
{
    let output_path = Path::new("data/unit_tests/")
        .join(test_name)
        // Use .vtk instead of .vtu because there's currently a bug which prevents
        // correctly exporting empty meshes to .vtu
        .join(format!("{file_stem}.vtk"));
    FiniteElementMeshDataSetBuilder::from_mesh(mesh)
        .try_export(output_path)
        .expect("Export failure is a test failure")
}

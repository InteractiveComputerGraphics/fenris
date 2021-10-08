use fenris_geometry::polymesh::{PolyMesh, PolyMesh3d};
use fenris_nested_vec::NestedVec;

use nalgebra::Point3;

fn create_single_tetrahedron_polymesh() -> PolyMesh3d<f64> {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ];
    let faces = NestedVec::from(&vec![
        // TODO: This isn't consistent with respect to winding order etc.
        // We need to introduce the concept of half faces or something similar to
        // make this stuff consistent per cell
        vec![0, 1, 2],
        vec![0, 1, 3],
        vec![1, 2, 3],
        vec![2, 0, 3],
    ]);
    let cells = NestedVec::from(&vec![vec![0, 1, 2, 3]]);
    PolyMesh::from_poly_data(vertices, faces, cells)
}

#[test]
fn triangulate_single_tetrahedron_is_unchanged() {
    let mesh = create_single_tetrahedron_polymesh();

    let triangulated = mesh.triangulate().unwrap();

    assert_eq!(triangulated.num_cells(), 1);
    assert_eq!(triangulated.num_faces(), 4);

    // TODO: Further tests!
}

#[test]
fn keep_cells() {
    {
        // Single tetrahedron
        let mesh = create_single_tetrahedron_polymesh();

        // Keep no cells, should give empty mesh
        {
            let kept = mesh.keep_cells(&[]);
            assert_eq!(kept.vertices().len(), 0);
            assert_eq!(kept.num_faces(), 0);
            assert_eq!(kept.num_cells(), 0);
        }

        // Keep cell 0, should give unchanged mesh back
        {
            let kept = mesh.keep_cells(&[0]);
            assert_eq!(mesh, kept);
        }
    }
}

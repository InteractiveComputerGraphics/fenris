use fenris::geometry::polymesh::PolyMesh3d;
use fenris::mesh::procedural::create_rectangular_uniform_hex_mesh;

use itertools::iproduct;
use matrixcompare::assert_scalar_eq;

#[test]
fn compute_volume() {
    {
        // Single cube, multiple resolutions
        let unit_lengths = [1.0, 0.5, 1.5];
        let nx = [1, 2, 3];
        let ny = [1, 2, 3];
        let nz = [1, 2, 3];
        let resolutions = [1, 2];

        for (u, nx, ny, nz, res) in iproduct!(&unit_lengths, &nx, &ny, &nz, &resolutions) {
            let cube = create_rectangular_uniform_hex_mesh(*u, *nx, *ny, *nz, *res);
            let cube = PolyMesh3d::from(&cube);
            let expected_volume: f64 = u * u * u * (nx * ny * nz) as f64;
            dbg!(u, nx, ny, nz, res);
            assert_scalar_eq!(
                cube.compute_volume(),
                expected_volume,
                comp = abs,
                tol = 1e-12
            );
        }
    }
}

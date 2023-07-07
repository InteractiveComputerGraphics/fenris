use fenris::assembly::global::assemble_scalar;
use fenris::connectivity::Connectivity;
use fenris::element::{ElementConnectivity, FiniteElement, SurfaceFiniteElement};
use fenris::integrate::{dependency::NoDeps, FnFunction, UFunction};
use fenris::integrate::{integrate_over_element, volume_form, ElementIntegralAssemblerBuilder};
use fenris::io::vtk::FiniteElementMeshDataSetBuilder;
use fenris::mesh::procedural::{create_rectangular_uniform_hex_mesh, create_rectangular_uniform_tet_mesh};
use fenris::quadrature::CanonicalMassQuadrature;
use fenris::quadrature::Quadrature;
use fenris::util::global_vector_from_point_fn;
use fenris_geometry::AxisAlignedBoundingBox3d;
use matrixcompare::prop_assert_scalar_eq;
use nalgebra::coordinates::XYZ;
use nalgebra::{dvector, vector, Point3, Vector1, Vector3, Vector4, U1};
use proptest::prelude::*;
use std::path::PathBuf;

#[test]
fn rectangular_uniform_tet_mesh_basics() {
    let base_path = PathBuf::from("data/unit_tests/mesh/procedural");
    for res in [1, 2] {
        let mesh = create_rectangular_uniform_tet_mesh(1.0, 1, 1, 1, res);
        FiniteElementMeshDataSetBuilder::from_mesh(&mesh)
            .try_export(base_path.join(format!("basic_uniform_{res}.vtk")))
            .unwrap();
        // The idea here is not to inspect the actual debug output but instead to
        // look at the VTK files and check that they look reasonable
        insta::assert_debug_snapshot!(format!("mesh_{res}"), mesh);
    }
}

fn empty_tet_mesh_params() -> impl Strategy<Value = [usize; 4]> {
    let strategy = prop_oneof![Just(0), 0usize..3];
    [strategy.clone(), strategy.clone(), strategy.clone(), strategy]
        .prop_filter("At least one of the parameters must be zero", |params: &[usize; 4]| {
            params.iter().product::<usize>() == 0
        })
}

#[derive(Debug)]
struct RectangularUniformTetMeshParams {
    unit_length: f64,
    units: [usize; 3],
    resolution: usize,
}

impl Arbitrary for RectangularUniformTetMeshParams {
    type Parameters = ();

    fn arbitrary_with(_args: ()) -> Self::Strategy {
        let units = [1usize..2, 1..2, 1..2];
        (0.1f64..10.0, units, 1..3usize)
            .prop_map(|(unit_length, units, resolution)| Self {
                unit_length,
                units,
                resolution,
            })
            .boxed()
    }

    type Strategy = BoxedStrategy<Self>;
}

proptest! {
    #[test]
    fn rectangular_uniform_tet_mesh_empty([units_x, units_y, units_z, cells_per_unit] in empty_tet_mesh_params()) {
        let mesh = create_rectangular_uniform_tet_mesh(1.0, units_x, units_y, units_z, cells_per_unit);
        prop_assert!(mesh.vertices().is_empty());
        prop_assert!(mesh.connectivity().is_empty());
    }

    #[test]
    fn rectangular_uniform_tet_mesh_polynomial_integral(params: RectangularUniformTetMeshParams) {
        // Integrate a function f(x, u), where u = u(x) is a linear function
        // (so that both linear tets and trilinear hex elements are able to represent it exactly)
        // and f overall is linear.
        let u = |p: &Point3<_>| vector![4.0 * p.x + 2.0 * p.y - 2.0 * p.z + 3.0];
        let f = FnFunction::new(|p: &Point3<_>, u: &Vector1<_>| vector![-3.0 * p.x + 5.0 * p.y - 2.0 * p.z + 5.0 - 3.0 * u[0]]);

        let RectangularUniformTetMeshParams { unit_length, units, resolution } = params;
        let [nx, ny, nz] = units;
        let tet_mesh = create_rectangular_uniform_tet_mesh(unit_length, nx, ny, nz, resolution);
        let tet_quadrature = tet_mesh.canonical_mass_quadrature();
        let u_tet = global_vector_from_point_fn(tet_mesh.vertices(), u);

        // Use minimal element hex mesh to compute reference value on
        let hex_mesh = create_rectangular_uniform_hex_mesh(unit_length, nx, ny, nz, 1);
        let hex_quadrature = hex_mesh.canonical_mass_quadrature();
        let u_hex = global_vector_from_point_fn(hex_mesh.vertices(), u);

        let hex_assembler = ElementIntegralAssemblerBuilder::new()
            .with_quadrature_table(&hex_quadrature)
            .with_space(&hex_mesh)
            .with_integrand(f.clone())
            .with_interpolation_weights(&u_hex)
            .build_integrator();
        let hex_result = assemble_scalar(&hex_assembler).unwrap();

        let tet_assembler = ElementIntegralAssemblerBuilder::new()
            .with_quadrature_table(&tet_quadrature)
            .with_space(&tet_mesh)
            .with_integrand(f)
            .with_interpolation_weights(&u_tet)
            .build_integrator();
        let tet_result = assemble_scalar(&tet_assembler).unwrap();

        prop_assert_scalar_eq!(tet_result, hex_result, comp = abs, tol = hex_result.abs() * 1e-12);
    }

    #[test]
    fn rectangular_uniform_tet_mesh_geometric_invariants(params: RectangularUniformTetMeshParams) {
        let RectangularUniformTetMeshParams { unit_length, units, resolution } = params;
        let [nx, ny, nz] = units;
        let tet_mesh = create_rectangular_uniform_tet_mesh(unit_length, nx, ny, nz, resolution);

        let extents = [nx, ny, nz].map(|n| (n as f64) * unit_length);
        let aabb = AxisAlignedBoundingBox3d::from_points(tet_mesh.vertices()).unwrap();
        prop_assert_eq!(aabb.min(), &Point3::origin());
        prop_assert_eq!(aabb.max(), &Point3::from(extents));

        for connectivity in tet_mesh.connectivity() {
            let volume_element = connectivity.element(tet_mesh.vertices()).unwrap();
            // Jacobian is constant so can pick any reference point
            let j_det = volume_element.reference_jacobian(&Point3::origin()).determinant();
            prop_assert!(j_det > 0.0, "element is inverted");
        }
    }

    #[test]
    fn rectangular_uniform_tet_mesh_element_wise_divergence(params: RectangularUniformTetMeshParams) {
        // We check that the divergence theorem applies to each element. This implicitly
        // is intended to verify that all elements have correct orientation. This is however
        // somewhat redundant given that checking for inversion already accomplishes the same thing.
        // However it acts as an additional sanity check that the implementations for the
        // normal computation and face connectivity are sane

        let RectangularUniformTetMeshParams { unit_length, units, resolution } = params;
        let [nx, ny, nz] = units;
        let tet_mesh = create_rectangular_uniform_tet_mesh(unit_length, nx, ny, nz, resolution);
        let tet_quadrature = fenris::quadrature::total_order::tetrahedron(1).unwrap();
        let tri_quadrature = fenris::quadrature::total_order::triangle(2).unwrap();

        let div_f = |p: &Point3<f64>| {
            let XYZ { x, y, z } = *p.coords;
            vector![
                (6.0 * x + 2.0)
                + (4.0 * y + 3.0 - 3.0 * x)
                + (2.0 * z - 3.0 - 3.0 * x)
            ]
        };
        let div_f = FnFunction::new(div_f)
            .with_dependencies::<NoDeps<U1>>();

        let f = |p: &Point3<f64>| {
            let XYZ { x, y, z } = *p.coords;
            vector![
                3.0 * x.powi(2) + 2.0 * x + 4.0 * y * z - 3.0,
                2.0 * y.powi(2) + 3.0 * y - 3.0 * x * y + 2.0,
                z.powi(2) - 3.0 * z - 3.0 * x * z + x * y - 4.0
            ]
        };

        for connectivity in tet_mesh.connectivity() {
            let volume_element = connectivity.element(tet_mesh.vertices()).unwrap();
            // Interpolation weights are irrelevant here since our function does not depend on u,
            // just use zero weights.
            let u = Vector4::repeat(0.0);
            let div_integral = integrate_over_element(
                &div_f,
                &volume_element,
                &tet_quadrature,
                &u,
                &mut Default::default())[0];

            let mut flux_integral = 0.0;
            for face_idx in 0 .. connectivity.num_faces() {
                let face_conn = connectivity.get_face_connectivity(face_idx).unwrap();
                let face_element = face_conn.element(tet_mesh.vertices()).unwrap();

                for (w, xi, _) in tri_quadrature.iter() {
                    let n = face_element.normal(xi);
                    let x = face_element.map_reference_coords(xi);
                    let j = face_element.reference_jacobian(xi);
                    flux_integral += w * volume_form(&j) * n.dot(&f(&x));
                }
            }
            let tol = 1e-12 * div_integral.abs();
            prop_assert_scalar_eq!(flux_integral, div_integral, comp = abs, tol = tol);
        }
    }
}

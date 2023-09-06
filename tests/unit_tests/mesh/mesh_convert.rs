use matrixcompare::assert_scalar_eq;
use nalgebra::coordinates::XYZ;
use nalgebra::{Point3, U1, Vector1};
use fenris::assembly::global::assemble_scalar;
use fenris::assembly::local::UniformQuadratureTable;
use fenris::error::estimate_L2_error;
use fenris::integrate::dependency::NoDeps;
use fenris::integrate::{ElementIntegralAssemblerBuilder, FnFunction};
use fenris::mesh::procedural::create_rectangular_uniform_tet_mesh;
use fenris::mesh::Tet20Mesh;
use fenris::quadrature::total_order;
use fenris::util::global_vector_from_point_fn;

#[test]
fn tet20_from_tet4_can_represent_cubic_polynomial() {
    let res = 1;

    // TODO: Use different lengths in different dims
    let mesh_tet4 = create_rectangular_uniform_tet_mesh(0.5, 2, 1, 3, res);
    let mesh_tet20 = Tet20Mesh::from(&mesh_tet4);

    let polynomial = |p: &Point3<f64>| -> f64 {
        let XYZ { x, y, z } = *p.coords;
        2.0 * x.powi(3)
            - 3.0 * y.powi(3)
            + 1.0 * z.powi(3)
            - 2.0 * x * y * z
            + 3.0 * x.powi(2)
            - 4.0 * y.powi(2)
            + 1.0 * z.powi(2)
            + 2.0 * x * y
            - 3.0 * x * z
            + 1.0 * y * z
            + 4.0 * x
            + 3.0 * y
            - 2.0 * z
            + 1.0
    };
    let f = |x: &Point3<f64>| -> Vector1<f64> { Vector1::new(polynomial(x)) };

    // Nodal interpolation onto Tet20 mesh
    let u_tet20 = global_vector_from_point_fn(mesh_tet20.vertices(), f);
    let u_tet4 = global_vector_from_point_fn(mesh_tet4.vertices(), f);

    // Need order 6 since we're computing squared error of cubic polynomial
    let qtable = UniformQuadratureTable::from_quadrature(total_order::tetrahedron(6).unwrap());
    let error = estimate_L2_error(&mesh_tet20, &f, &u_tet20, &qtable).unwrap();

    let exact_assembler = ElementIntegralAssemblerBuilder::new()
        .with_quadrature_table(&qtable)
        // Use tet4 mesh for good measure, so that it's evaluated somewhat differently
        .with_space(&mesh_tet4)
        // Integration weights are not used but with the current API we must
        // still supply something
        .with_interpolation_weights(&u_tet4)
        .with_integrand(FnFunction::new(f).with_dependencies::<NoDeps<U1>>())
        .build_integrator();
    let exact_integral = assemble_scalar(&exact_assembler).unwrap();

    let tet20_assembler = ElementIntegralAssemblerBuilder::new()
        .with_quadrature_table(&qtable)
        .with_space(&mesh_tet20)
        .with_interpolation_weights(&u_tet20)
        .with_integrand(FnFunction::new(|_x: &Point3<f64>, u: &Vector1<f64>| u.clone()))
        .build_integrator();
    let tet20_integral = assemble_scalar(&tet20_assembler).unwrap();

    let tol = exact_integral * 1e-12;
    assert_scalar_eq!(tet20_integral, exact_integral, comp = abs, tol = tol);
    assert_scalar_eq!(error, 0.0, comp = abs, tol = tol);
}
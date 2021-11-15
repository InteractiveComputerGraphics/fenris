use fenris::assembly::global::{CsrAssembler, VectorAssembler};
use fenris::assembly::local::{Density, ElementMassAssembler, ElementSourceAssemblerBuilder, UniformQuadratureTable};
use fenris::mesh::procedural::create_unit_square_uniform_quad_mesh_2d;
use fenris::mesh::QuadMesh2d;
use fenris::nalgebra;
use fenris::nalgebra::{vector, DVector};
use fenris::quadrature;
use fenris_solid::GravitySource;
use matrixcompare::assert_matrix_eq;
use std::iter::repeat;

#[test]
fn gravity_source_agrees_with_mass_matrix_vector_product_quad4() {
    // Test that the gravity force f_g = M g,
    // where g = [ 0.0, -9.81, 0.0, -9.81, ... ]
    // (when using the same quadrature rule or otherwise an exact quadrature)
    let mesh: QuadMesh2d<f64> = create_unit_square_uniform_quad_mesh_2d(4);
    let quadrature = quadrature::tensor::quadrilateral_gauss(2);
    let mass_quadrature = UniformQuadratureTable::from_quadrature_and_uniform_data(quadrature, Density(2.0));

    let mass_assembler = ElementMassAssembler::with_solution_dim(2)
        .with_quadrature_table(&mass_quadrature)
        .with_space(&mesh);
    let mass_matrix = CsrAssembler::default().assemble(&mass_assembler).unwrap();

    let gravity_source = GravitySource::from_acceleration(vector![0.0, -9.81]);
    let gravity_assembler = ElementSourceAssemblerBuilder::new()
        .with_source(&gravity_source)
        .with_quadrature_table(&mass_quadrature)
        .with_finite_element_space(&mesh)
        .build();
    let f_gravity = VectorAssembler::default()
        .assemble_vector(&gravity_assembler)
        .unwrap();

    let num_nodes = mesh.vertices().len();
    let g = DVector::from_iterator(2 * num_nodes, repeat([0.0, -9.81]).take(num_nodes).flatten());
    let mg = mass_matrix * &g;
    assert_matrix_eq!(f_gravity, mg, comp = float);
}

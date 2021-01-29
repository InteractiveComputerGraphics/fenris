//! Tests for CG applied to finite element matrices.
//!
//! **This test is currently commented out because fenris-solid is not yet available.
//! TODO: Re-enable/re-write this test once we have this functionality available again

// use fenris::model::NodalModel3d;
// use fenris::nalgebra::DVector;
// use fenris::procedural::create_rectangular_uniform_hex_mesh;
// use fenris::quadrature::hex_quadrature_strength_5;
// use fenris_solid::materials::{LinearElasticMaterial, YoungPoisson};
// use fenris_solid::ElasticityModel;
// use fenris_sparse::cg::{CgWorkspace, ConjugateGradient, RelativeResidualCriterion};
// use fenris_sparse::CsrMatrix;
// use std::ops::Add;
// use util::assert_approx_matrix_eq;
//
// #[test]
// fn cg_linear_elasticity_dynamic_regular_grid() {
//     let mesh = create_rectangular_uniform_hex_mesh(1.0f64, 1, 1, 1, 5);
//     let model = NodalModel3d::from_mesh_and_quadrature(mesh, hex_quadrature_strength_5());
//
//     // Use linear elastic material since it is guaranteed to give positive semi-definite
//     // stiffness matrices
//     let material = LinearElasticMaterial::from(YoungPoisson {
//         young: 1e4,
//         poisson: 0.48,
//     });
//
//     let u = DVector::zeros(model.ndof());
//     let mass = model.assemble_mass(1.0).to_csr(Add::add);
//     let mut stiffness =
//         CsrMatrix::from_pattern_and_values(mass.sparsity_pattern(), vec![0.0; mass.nnz()]);
//     model.assemble_stiffness_into(&mut stiffness, &u, &material);
//
//     let dt = 0.1;
//     let system_matrix = mass + stiffness * (dt * dt);
//
//     let solutions = vec![
//         DVector::repeat(model.ndof(), 1.0),
//         DVector::repeat(model.ndof(), 2.0),
//         DVector::repeat(model.ndof(), 3.0),
//     ];
//
//     // Use a workspace and solve several systems to test that the internal state of the workspace
//     // does not impact the solutions
//     let mut cg_workspace = CgWorkspace::default();
//
//     for x0 in solutions {
//         let b = &system_matrix * &x0;
//
//         let mut x_workspace = DVector::repeat(model.ndof(), 0.0);
//         let output_workspace = ConjugateGradient::with_workspace(&mut cg_workspace)
//             .with_operator(&system_matrix)
//             .with_stopping_criterion(RelativeResidualCriterion::new(1e-8))
//             .solve_with_guess(&b, &mut x_workspace)
//             .unwrap();
//
//         let mut x_no_workspace = DVector::repeat(model.ndof(), 0.0);
//         let output_no_workspace = ConjugateGradient::new()
//             .with_operator(&system_matrix)
//             .with_stopping_criterion(RelativeResidualCriterion::new(1e-8))
//             .solve_with_guess(&b, &mut x_no_workspace)
//             .unwrap();
//
//         // Results should be exactly the same regardless of whether we use a workspace or not.
//         assert_eq!(x_no_workspace, x_workspace);
//         assert_eq!(
//             output_workspace.num_iterations,
//             output_no_workspace.num_iterations
//         );
//         let x = x_no_workspace;
//         assert_approx_matrix_eq!(&x, &x0, abstol = 1e-7);
//
//         println!(
//             "Matrix size: {}x{} ({} nnz)",
//             system_matrix.nrows(),
//             system_matrix.ncols(),
//             system_matrix.nnz()
//         );
//         println!("Num CG iterations: {}", output_no_workspace.num_iterations);
//
//         let system_diagonal = DVector::from_iterator(
//             model.ndof(),
//             system_matrix.diag_iter().map(|d_i| d_i.recip()),
//         );
//         let diagonal_preconditioner = CsrMatrix::from_diagonal(&system_diagonal);
//         let mut x = DVector::repeat(model.ndof(), 0.0);
//         let output_preconditioned = ConjugateGradient::new()
//             .with_operator(&system_matrix)
//             .with_preconditioner(&diagonal_preconditioner)
//             .with_stopping_criterion(RelativeResidualCriterion::new(1e-8))
//             .solve_with_guess(&b, &mut x)
//             .unwrap();
//         assert_approx_matrix_eq!(&x, &x0, abstol = 1e-7);
//
//         println!(
//             "Num CG iterations (diagonal precond): {}",
//             output_preconditioned.num_iterations
//         );
//
//         // For these particular matrices we expect the diagonal preconditioner to
//         // yield less iterations
//         assert!(output_preconditioned.num_iterations < output_workspace.num_iterations);
//     }
// }

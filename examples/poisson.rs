//! This example is broken at the moment, and we'll reimplement a better version in the future.

fn main() {}

// mod poisson_common;
// use poisson_common::*;
//
// use fenris::assembly::{
//     apply_homogeneous_dirichlet_bc_csr, apply_homogeneous_dirichlet_bc_rhs,
//     assemble_generalized_mass, assemble_generalized_stiffness, UniformQuadratureTable,
// };
// use fenris::element::ElementConnectivity;
// use fenris::error::estimate_element_L2_error_squared;
// use fenris::mesh::Mesh2d;
// use fenris::procedural::create_rectangular_uniform_quad_mesh_2d;
// use fenris::quadrature::{quad_quadrature_strength_11, quad_quadrature_strength_5, Quadrature2d};
// use nalgebra::storage::Storage;
// use nalgebra::{
//     DVector, DefaultAllocator, DimNameMul, RealField, Scalar, Vector1, Vector2, U1, U2,
// };
// use std::error::Error;
//
// use fenris::allocators::FiniteElementMatrixAllocator;
// use fenris::connectivity::{CellConnectivity, Connectivity};
// use itertools::izip;
// use mkl_corrode::dss;
// use mkl_corrode::dss::Definiteness;
// use mkl_corrode::dss::MatrixStructure::Symmetric;
// use sparse::CsrMatrix;
// use std::ops::Add;
//
// pub struct TestProblem<T, Solution, Rhs, Connectivity>
// where
//     T: Scalar,
// {
//     pub solution: Solution,
//     pub rhs: Rhs,
//     pub mesh: Mesh2d<T, Connectivity>,
// }
//
// pub struct PoissonSystem<T>
// where
//     T: Scalar,
// {
//     pub mass_matrix: CsrMatrix<T>,
//     pub stiffness_matrix: CsrMatrix<T>,
//     pub rhs: DVector<T>,
// }
//
// #[allow(non_snake_case)]
// pub fn build_poisson_system<T, Solution, Rhs, Conn>(
//     problem: &TestProblem<T, Solution, Rhs, Conn>,
// ) -> Result<PoissonSystem<T>, Box<dyn Error>>
// where
//     T: RealField + mkl_corrode::SupportedScalar,
//     Solution: Fn(T, T) -> T,
//     Rhs: Fn(T, T) -> T,
//     Conn: ElementConnectivity<T, GeometryDim = U2, ReferenceDim = U2> + CellConnectivity<T, U2>,
//     Conn::FaceConnectivity: Connectivity,
//     DefaultAllocator: FiniteElementMatrixAllocator<T, U1, U2>,
// {
//     let f = &problem.rhs;
//     let vertices = problem.mesh.vertices();
//     let connectivity = problem.mesh.connectivity();
//
//     let operator = PoissonEllipticOperator;
//     let u = DVector::zeros(vertices.len());
//     let quadrature = quad_quadrature_strength_5();
//     let qtable = UniformQuadratureTable(&quadrature);
//
//     // Assemble system matrix and right hand side
//     let mut A = assemble_generalized_stiffness(vertices, connectivity, &operator, &u, &qtable)
//         .to_csr(Add::add);
//     let mut M = assemble_generalized_mass::<_, U1, _, _>(vertices, connectivity, T::one(), &qtable)
//         .to_csr(Add::add);
//     let f_nodes = DVector::from_iterator(vertices.len(), vertices.iter().map(|v| f(v.x, v.y)));
//     let mut rhs = &M * &f_nodes;
//
//     // Apply BC
//     let boundary_vertices = problem.mesh.find_boundary_vertices();
//     apply_homogeneous_dirichlet_bc_csr::<_, U1>(&mut A, &boundary_vertices);
//     apply_homogeneous_dirichlet_bc_csr::<_, U1>(&mut M, &boundary_vertices);
//     apply_homogeneous_dirichlet_bc_rhs(&mut rhs, &boundary_vertices, 1);
//
//     Ok(PoissonSystem {
//         mass_matrix: M,
//         stiffness_matrix: A,
//         rhs,
//     })
// }
//
// #[allow(non_snake_case)]
// pub fn solve_2d_test_problem<T, Solution, Rhs, Conn>(
//     problem: &TestProblem<T, Solution, Rhs, Conn>,
// ) -> Result<DVector<T>, Box<dyn Error>>
// where
//     T: RealField + mkl_corrode::SupportedScalar,
//     Solution: Fn(T, T) -> T,
//     Rhs: Fn(T, T) -> T,
//     Conn: ElementConnectivity<T, GeometryDim = U2, ReferenceDim = U2> + CellConnectivity<T, U2>,
//     Conn::FaceConnectivity: Connectivity,
//     DefaultAllocator: FiniteElementMatrixAllocator<T, U1, U2>,
// {
//     let system = build_poisson_system(problem)?;
//
//     let A = system.stiffness_matrix;
//     let rhs = system.rhs;
//
//     let A_dss = dss::SparseMatrix::try_convert_from_csr(
//         A.row_offsets(),
//         A.column_indices(),
//         A.values(),
//         Symmetric,
//     )?;
//     let options = dss::SolverOptions::default().parallel_reorder(true);
//     let mut solver =
//         dss::Solver::try_factor_with_opts(&A_dss, Definiteness::PositiveDefinite, &options)?;
//     let solution = solver.solve(rhs.data.as_slice()).unwrap();
//
//     Ok(DVector::from_vec(solution))
// }
//
// pub fn estimate_l2_error<T, Connectivity>(
//     mesh: &Mesh2d<T, Connectivity>,
//     u_h: &DVector<T>,
//     u_exact: impl Fn(T, T) -> T,
//     quadrature: impl Quadrature2d<T>,
// ) -> T
// where
//     T: RealField,
//     Connectivity:
//         ElementConnectivity<T, GeometryDim = U2, ReferenceDim = U2> + CellConnectivity<T, U2>,
//     DefaultAllocator: FiniteElementMatrixAllocator<T, U1, U2>,
// {
//     // In the below, we compute the error in the same mesh as was used for computation.
//     // This should generally be fine for getting an approximate error measure, provided
//     // that the quadrature is of high order compared to the elements used,
//     // so that the integration error is much smaller than the discretization error.
//     let mut l2_error_squared = T::zero();
//     let elements = mesh
//         .connectivity()
//         .iter()
//         .map(|conn| conn.element(mesh.vertices()).unwrap());
//     for (element, conn) in izip!(elements, mesh.connectivity()) {
//         let u_h_weights = conn.element_variables(u_h);
//         l2_error_squared += estimate_element_L2_error_squared(
//             &element,
//             |p, _| Vector1::new(u_exact(p.x, p.y)),
//             &u_h_weights,
//             &quadrature,
//         );
//     }
//     let l2_error = l2_error_squared.sqrt();
//     l2_error
// }
//
// #[allow(non_snake_case)]
// fn main() -> Result<(), Box<dyn Error>> {
//     use std::f64::consts::PI;
//     let sin = |x| f64::sin(x);
//     //    let cos = |x| f64::cos(x);
//     let u_exact = |x, y| sin(PI * x) * sin(PI * y);
//     //    let u_grad_exact = |x, y| Vector2::new(PI * cos(PI * x) * sin(PI * y),
//     //                                           PI * sin(PI * x) * cos(PI * y));
//     let f = |x, y| 2.0 * PI * PI * u_exact(x, y);
//
//     let resolutions = vec![1, 2, 3, 4, 5, 6, 7, 14, 28];
//
//     let error_quadrature = quad_quadrature_strength_11();
//     for res in resolutions {
//         let problem = TestProblem {
//             solution: u_exact,
//             rhs: f,
//             mesh: create_rectangular_uniform_quad_mesh_2d(1.0, 2, 2, res, &Vector2::new(-1.0, 1.0)),
//         };
//         let u_h = solve_2d_test_problem(&problem)?;
//         let l2_error = estimate_l2_error(&problem.mesh, &u_h, u_exact, &error_quadrature);
//         println!("L2 error: {}", l2_error);
//     }
//
//     Ok(())
// }

//! Use method of manufactured solutions on a 2D Poisson problem to verify convergence.
//!
//! The problem is:
//!   - Delta u = f,
//! where Delta = nabla^2 is the Laplace operator.
use std::f64::consts::PI;
use std::ops::Deref;

use eyre::eyre;
use nalgebra::{Dynamic, MatrixSliceMN, UniformNorm};

use fenris::quadrature;
use fenris::assembly::global::{apply_homogeneous_dirichlet_bc_csr, apply_homogeneous_dirichlet_bc_rhs, CsrAssembler, gather_global_to_local, SerialVectorAssembler};
use fenris::assembly::local::{ElementEllipticAssemblerBuilder, ElementSourceAssemblerBuilder, SourceFunction, UniformQuadratureTable};
use fenris::assembly::operators::{LaplaceOperator, Operator};
use fenris::connectivity::Connectivity;
use fenris::element::ElementConnectivity;
use fenris::error::{ErrorWorkspace, estimate_element_L2_error_squared};
use fenris::io::vtk::FiniteElementMeshDataSetBuilder;
use fenris::mesh::QuadMesh2d;
use fenris::nalgebra::{DMatrix, DVector, Point, Point2, U1, U2, Vector1, Vector2, VectorN};
use fenris::nalgebra::coordinates::XY;
use fenris::nalgebra_sparse::CsrMatrix;
use fenris::procedural::create_unit_square_uniform_quad_mesh_2d;

fn sin(x: f64) -> f64 { x.sin() }
// fn cos(x: f64) -> f64 { x.cos() }

// Exact solution
fn u_exact(x: &Point2<f64>) -> f64 {
    let &XY { x, y } = x.coords.deref();
    sin(PI * x) * sin(PI * y)
}

// fn u_exact_grad(x: &Point2<f64>) -> Vector2<f64> {
//     let &XY { x, y } = x.coords.deref();
//     let u_x = - PI * cos(PI * x) * sin(PI * y);
//     let u_y = - PI * sin(PI * x) * cos(PI * y);
//     Vector2::new(u_x, u_y)
// }

fn f(x: &Point2<f64>) -> f64 {
    // Derived from f = - Del u = - u_xx - u_yy
    - 2.0 * PI * PI * u_exact(x)
}

#[derive(Debug)]
pub struct PoissonProblemSourceFunction;

impl Operator for PoissonProblemSourceFunction {
    type SolutionDim = U1;
    type Parameters = ();
}

impl SourceFunction<f64, U2> for PoissonProblemSourceFunction {
    fn evaluate(&self, coords: &Point<f64, U2>, _data: &Self::Parameters) -> VectorN<f64, Self::SolutionDim> {
        Vector1::new(f(coords))
    }
}

/// This is a generalized version of the poisson2d example
fn assemble_linear_system(mesh: &QuadMesh2d<f64>) -> eyre::Result<(CsrMatrix<f64>, DVector<f64>)> {
    let (weights, points) = quadrature::tensor::quadrilateral_gauss(2);
    let quadrature = UniformQuadratureTable::from_points_and_weights(points, weights);

    // TODO: This isn't actually needed. Get rid of it by introducing a separate trait
    // for linear contractions
    let u = DVector::<f64>::zeros(mesh.vertices().len());

    let vector_assembler = SerialVectorAssembler::<f64>::default();
    let matrix_assembler = CsrAssembler::default();

    let laplace_assembler = ElementEllipticAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        .with_operator(&LaplaceOperator)
        .with_quadrature_table(&quadrature)
        .with_u(&u)
        .build();

    let mut a_global = matrix_assembler.assemble(&laplace_assembler)?;

    let source_assembler = ElementSourceAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        // TODO: Use better quadrature
        .with_quadrature_table(&quadrature)
        .with_source(&PoissonProblemSourceFunction)
        .build();

    let mut b_global = vector_assembler.assemble_vector(&source_assembler)?;

    // We want to have a Dirichlet boundary for |x| == 1. To account for slight numerical errors,
    // we determine the indices of the Dirichlet nodes by extracting those node indices
    // which satisfy x < eps, for some small epsilon.
    let dirichlet_nodes: Vec<_> = mesh
        .vertices()
        .iter()
        .enumerate()
        // TODO: Clean this up a bit
        .filter_map(|(idx, x)| ((x.coords - Vector2::new(0.5, 0.5)).apply_norm(&UniformNorm) > 0.4999).then(|| idx))
        .collect();

    apply_homogeneous_dirichlet_bc_csr(&mut a_global, &dirichlet_nodes, 1);
    apply_homogeneous_dirichlet_bc_rhs(&mut b_global, &dirichlet_nodes, 1);

    Ok((a_global, b_global))
}

fn solve_linear_system(matrix: &CsrMatrix<f64>, rhs: &DVector<f64>) -> eyre::Result<DVector<f64>> {
    // TODO: Use sparse solver
    let matrix = DMatrix::from(matrix);
    // The discrete Laplace operator is positive definite (given appropriate boundary conditions),
    // so we can use a Cholesky factorization
    let cholesky = matrix
        .cholesky()
        .ok_or_else(|| eyre!("Failed to solve linear system"))?;
    Ok(cholesky.solve(rhs))
}

#[test]
fn poisson_2d_quad4() {
    let resolutions = [2, 4, 8, 16, 32];

    for &cells_per_dim in &resolutions {
        let mesh = create_unit_square_uniform_quad_mesh_2d(cells_per_dim);

        let (a, b) = assemble_linear_system(&mesh).unwrap();
        let u_h = solve_linear_system(&a, &b).unwrap();

        // TODO: Clean all this up
        let mut error_workspace = ErrorWorkspace::default();
        let error_quadrature = quadrature::tensor::quadrilateral_gauss(4);
        let l2_error = mesh.connectivity()
            .iter()
            .map(|conn| {
                let num_nodes = conn.vertex_indices().len();
                let mut u_local = DVector::zeros(num_nodes);
                gather_global_to_local(&u_h, &mut u_local, conn.vertex_indices(), 1);
                let element = conn.element(mesh.vertices()).unwrap();
                let u_h_element = MatrixSliceMN::from_slice_generic(&u_local.as_slice(), U1, Dynamic::new(num_nodes));
                estimate_element_L2_error_squared(&mut error_workspace, &element, |x, _| Vector1::new(u_exact(x)), u_h_element, &error_quadrature)
            })
            .sum::<f64>()
            .sqrt();

        println!("L2 error: {}", l2_error);

        FiniteElementMeshDataSetBuilder::from_mesh(&mesh)
            .with_title(format!("Poisson 2D FEM Res {}", cells_per_dim))
            .with_point_scalar_attributes("u_h", u_h.as_slice())
            .try_export(format!("data/convergence_tests/poisson_2d_mms/poisson2d_mms_approx_res_{}.vtu", cells_per_dim))
            .unwrap();

        // Evaluate u_exact at mesh vertices
        let u_exact_vector: Vec<_> = mesh.vertices()
                .iter()
                .map(|x| u_exact(x))
                .collect();

        FiniteElementMeshDataSetBuilder::from_mesh(&mesh)
            .with_title(format!("Poisson 2D FEM Exact solution Res {}", cells_per_dim))
            .with_point_scalar_attributes("u_exact", &u_exact_vector)
            .try_export(format!("data/convergence_tests/poisson_2d_mms/poisson2d_mms_exact_res_{}.vtu", cells_per_dim))
            .unwrap();
    }
}
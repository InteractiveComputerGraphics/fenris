use eyre::eyre;
use nalgebra::Vector1;

use fenris::assembly::global::{
    apply_homogeneous_dirichlet_bc_csr, apply_homogeneous_dirichlet_bc_rhs, CsrAssembler,
    SerialVectorAssembler,
};
use fenris::assembly::local::{
    ElementEllipticAssemblerBuilder, ElementSourceAssemblerBuilder, SourceFunction,
    UniformQuadratureTable,
};
use fenris::assembly::operators::{LaplaceOperator, Operator};
use fenris::io::vtk::FiniteElementMeshDataSetBuilder;
use fenris::mesh::QuadMesh2d;
use fenris::nalgebra::{DMatrix, DVector, Point2, U1, U2};
use fenris::nalgebra_sparse::CsrMatrix;
use fenris::mesh::procedural::create_unit_square_uniform_quad_mesh_2d;
use fenris::quadrature;

fn main() -> eyre::Result<()> {
    // TODO: Make it easy to construct triangle meshes as well.
    // Need to make it easy to convert between different meshes, such as Quad2d -> Tri2d
    let mesh: QuadMesh2d<f64> = create_unit_square_uniform_quad_mesh_2d(4);

    let (a, b) = assemble_linear_system(&mesh)?;
    let u = solve_linear_system(&a, &b)?;

    FiniteElementMeshDataSetBuilder::from_mesh(&mesh)
        .with_title("Poisson 2D")
        .with_point_scalar_attributes("u", u.as_slice())
        .try_export("poisson2d.vtu")?;

    Ok(())
}

fn assemble_linear_system(mesh: &QuadMesh2d<f64>) -> eyre::Result<(CsrMatrix<f64>, DVector<f64>)> {
    // A quadrature table is responsible for providing each element with a quadrature rule.
    // Since we want the same quadrature rule per element, we use a uniform quadrature table.
    let (weights, points) = quadrature::tensor::quadrilateral_gauss(2);
    let quadrature = UniformQuadratureTable::from_points_and_weights(points, weights);

    // TODO: This isn't actually needed. Get rid of it by introducing a separate trait
    // for linear contractions
    let u = DVector::<f64>::zeros(mesh.vertices().len());

    // Set up global assemblers. These are responsible for adding element matrix/vector entries from
    // each element into the global matrix/vector
    let vector_assembler = SerialVectorAssembler::<f64>::default();
    let matrix_assembler = CsrAssembler::default();

    // Set up local matrix assembler for the Laplace operator. This assembler is responsible
    // for building the local element matrices corresponding to each element
    let laplace_assembler = ElementEllipticAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        .with_operator(&LaplaceOperator)
        .with_quadrature_table(&quadrature)
        // TODO: If the operator is linear, u is not actually needed...
        // How to reflect this in the API?
        .with_u(&u)
        .build();

    let mut a_global = matrix_assembler.assemble(&laplace_assembler)?;

    // Set up a local vector assembler for the "source term", which corresponds to the weak form
    // term (f, v), where v is a test function and f is the source function `f` in the
    // Poisson equation delta u = f.
    let source_assembler = ElementSourceAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        .with_quadrature_table(&quadrature)
        .with_source(&PoissonProblemSourceFunction)
        .build();

    let mut b_global = vector_assembler.assemble_vector(&source_assembler)?;

    // We want to have a Dirichlet boundary for x == 0. To account for slight numerical errors,
    // we determine the indices of the Dirichlet nodes by extracting those node indices
    // which satisfy x < eps, for some small epsilon.
    let dirichlet_nodes: Vec<_> = mesh
        .vertices()
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| (v.x < 1e-12).then(|| idx))
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

/// Represents the source function `f` in the poisson equation - Delta u = f.
struct PoissonProblemSourceFunction;

impl Operator for PoissonProblemSourceFunction {
    /// f maps R^2 to R^1 (U1)
    type SolutionDim = U1;
    type Parameters = ();
}

impl SourceFunction<f64, U2> for PoissonProblemSourceFunction {
    fn evaluate(&self, _coords: &Point2<f64>, _data: &Self::Parameters) -> Vector1<f64> {
        // TODO: Use a more interesting function than a constant function
        Vector1::new(1.0)
    }
}

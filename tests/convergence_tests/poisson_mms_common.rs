use eyre::eyre;
use fenris::allocators::{DimAllocator, TriDimAllocator};
use fenris::assembly::global::{
    apply_homogeneous_dirichlet_bc_csr, apply_homogeneous_dirichlet_bc_rhs, color_nodes, CsrAssembler, CsrParAssembler,
    VectorAssembler, VectorParAssembler,
};
use fenris::assembly::local::{
    ElementEllipticAssemblerBuilder, ElementSourceAssemblerBuilder, SourceFunction, UniformQuadratureTable,
};
use fenris::assembly::operators::LaplaceOperator;
use fenris::element::ElementConnectivity;
use fenris::error::{estimate_H1_seminorm_error, estimate_L2_error};
use fenris::io::vtk::{FiniteElementMeshDataSetBuilder, VtkCellConnectivity};
use fenris::mesh::Mesh;
use fenris::nalgebra::{DVector, DefaultAllocator, DimName, Dynamic, OPoint, UniformNorm, Vector1, U1};
use fenris::nalgebra_sparse::CsrMatrix;
use fenris::quadrature::QuadraturePair;
use fenris::SmallDim;
use itertools::izip;
use matrixcompare::assert_matrix_eq;
use nalgebra::allocator::Allocator;
use nalgebra::OVector;
use nalgebra_sparse::factorization::CscCholesky;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::PathBuf;

/// For serializing to JSON for subsequent analysis/plots
#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct ErrorSummary {
    pub element_name: String,
    pub L2_errors: Vec<f64>,
    pub H1_seminorm_errors: Vec<f64>,
    /// Resolutions here measured in floating-point cell size, e.g. for quads, each cell is `h x h`,
    /// where `h` is the resolution.
    pub resolutions: Vec<f64>,
}

pub fn assert_summary_is_close_to_reference(summary: &ErrorSummary, reference: &ErrorSummary) {
    assert_eq!(
        summary.element_name, reference.element_name,
        "Element names are not identical"
    );
    assert_eq!(
        summary.resolutions, reference.resolutions,
        "Resolutions are not identical"
    );
    assert_eq!(summary.L2_errors.len(), reference.L2_errors.len());
    assert_eq!(summary.H1_seminorm_errors.len(), reference.H1_seminorm_errors.len());

    for (e1, e2) in izip!(&summary.L2_errors, &reference.L2_errors) {
        let rel_error = (e1 - e2).abs() / e2.abs();
        if rel_error > 0.01 {
            panic!("L2 error deviates by more than 1% compared to expected error.");
        }
    }

    for (e1, e2) in izip!(&summary.H1_seminorm_errors, &reference.H1_seminorm_errors) {
        let rel_error = (e1 - e2).abs() / e2.abs();
        if rel_error > 0.01 {
            panic!("H1 seminorm error deviates by more than 1% compared to expected error.");
        }
    }
}

/// This is a generalized version of the poisson2d example
pub fn assemble_linear_system<C, D, Source>(
    mesh: &Mesh<f64, C::GeometryDim, C>,
    quadrature: QuadraturePair<f64, C::GeometryDim>,
    poisson_source_function: &Source,
) -> eyre::Result<(CsrMatrix<f64>, DVector<f64>)>
where
    D: SmallDim,
    C: ElementConnectivity<f64, GeometryDim = D, ReferenceDim = D> + Sync,
    Source: SourceFunction<f64, D, SolutionDim = U1, Parameters = ()> + Sync,
    DefaultAllocator: DimAllocator<f64, D>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Sync,
{
    let (weights, points) = quadrature;
    let quadrature = UniformQuadratureTable::from_points_and_weights(points, weights);
    //
    // TODO: This isn't actually needed. Get rid of it by introducing a separate trait
    // for linear contractions
    let u = DVector::<f64>::zeros(mesh.vertices().len());

    // Node coloring for test of parallel assembler
    let colors = color_nodes(mesh);

    let vector_assembler = VectorAssembler::default();
    let matrix_assembler = CsrAssembler::default();

    let source_assembler = ElementSourceAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        // TODO: Use better quadrature
        .with_quadrature_table(&quadrature)
        .with_source(poisson_source_function)
        .build();

    let mut b_global = vector_assembler.assemble_vector(&source_assembler)?;

    // Compare with parallel vector assembler
    {
        let par_b_global = VectorParAssembler::default().assemble_vector(&colors, &source_assembler)?;
        assert_matrix_eq!(b_global, par_b_global, comp = float);
    }

    let laplace_assembler = ElementEllipticAssemblerBuilder::new()
        .with_finite_element_space(mesh)
        .with_operator(&LaplaceOperator)
        .with_quadrature_table(&quadrature)
        .with_u(&u)
        .build();

    let mut a_global = matrix_assembler.assemble(&laplace_assembler)?;

    // Compare with parallel CSR assembler
    {
        let par_a_global = CsrParAssembler::default().assemble(&colors, &laplace_assembler)?;
        assert_matrix_eq!(a_global, par_a_global, comp = float);
    }

    // We want to have a Dirichlet boundary for |x| == 1. To account for slight numerical errors,
    // we determine the indices of the Dirichlet nodes by extracting those node indices
    // which satisfy x < eps, for some small epsilon.
    let dirichlet_nodes: Vec<_> = mesh
        .vertices()
        .iter()
        .enumerate()
        // TODO: Clean this up a bit
        .filter_map(|(idx, x)| {
            ((&x.coords - OVector::<f64, D>::repeat(0.5)).apply_norm(&UniformNorm) > 0.4999).then(|| idx)
        })
        .collect();

    apply_homogeneous_dirichlet_bc_csr(&mut a_global, &dirichlet_nodes, 1);
    apply_homogeneous_dirichlet_bc_rhs(&mut b_global, &dirichlet_nodes, 1);

    Ok((a_global, b_global))
}

pub fn solve_linear_system(matrix: &CsrMatrix<f64>, rhs: &DVector<f64>) -> eyre::Result<DVector<f64>> {
    // The discrete Laplace operator is positive definite (given appropriate boundary conditions),
    // so we can use a Cholesky factorization
    let cholesky =
        CscCholesky::factor(&matrix.into()).map_err(|err| eyre!("Failed to solve linear system. Error: {}", err))?;
    // TODO: So apparently `CscCholesky::solve` only works with dynamic matrices. Should support
    // any kind of matrix, especially vectors (DVector in particular)!
    // Need to make a PR for this
    let u = cholesky.solve(rhs);
    Ok(u.reshape_generic(Dynamic::new(rhs.len()), U1::name()))
}

#[allow(non_snake_case)]
pub struct PoissonSolveResult {
    pub u_h: DVector<f64>,
    pub L2_error: f64,
    pub H1_seminorm_error: f64,
}

#[allow(non_snake_case)]
pub fn solve_poisson<C, D, Source>(
    mesh: &Mesh<f64, D, C>,
    quadrature: QuadraturePair<f64, D>,
    error_quadrature: QuadraturePair<f64, D>,
    poisson_source_function: &Source,
    u_exact: impl Fn(&OPoint<f64, D>) -> f64,
    u_exact_grad: impl Fn(&OPoint<f64, D>) -> OVector<f64, D>,
) -> PoissonSolveResult
where
    C: ElementConnectivity<f64, GeometryDim = D, ReferenceDim = D> + Sync,
    D: SmallDim,
    Source: SourceFunction<f64, D, SolutionDim = U1, Parameters = ()> + Sync,
    // TODO: We should technically only require SmallDimAllocator<_, D>, but Rust gets type
    // inference wrong without this bound...
    DefaultAllocator: TriDimAllocator<f64, D, D, U1>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Sync,
{
    let (a, b) = assemble_linear_system(&mesh, quadrature, poisson_source_function).unwrap();
    let u_h = solve_linear_system(&a, &b).unwrap();

    // Use a relatively high order quadrature for error computations
    let (weights, points) = error_quadrature;
    let error_quadrature = UniformQuadratureTable::from_points_and_weights(points, weights);
    let L2_error = estimate_L2_error(
        mesh,
        |x: &OPoint<f64, D>| Vector1::repeat(u_exact(x)),
        &u_h,
        &error_quadrature,
    )
    .unwrap();
    let H1_seminorm_error = estimate_H1_seminorm_error(mesh, u_exact_grad, &u_h, &error_quadrature).unwrap();

    PoissonSolveResult {
        u_h,
        L2_error,
        H1_seminorm_error,
    }
}

pub fn solve_and_produce_output<C, D, Source>(
    element_name: &str,
    resolutions: &[usize],
    // Produce a mesh for the given resolution
    mesh_producer: impl Fn(usize) -> Mesh<f64, D, C>,
    quadrature: QuadraturePair<f64, D>,
    error_quadrature: QuadraturePair<f64, D>,
    poisson_source_function: &Source,
    u_exact: impl Fn(&OPoint<f64, D>) -> f64,
    u_exact_grad: impl Fn(&OPoint<f64, D>) -> OVector<f64, D>,
) where
    C: VtkCellConnectivity + ElementConnectivity<f64, GeometryDim = D, ReferenceDim = D> + Sync,
    D: SmallDim,
    Source: SourceFunction<f64, D, SolutionDim = U1, Parameters = ()> + Sync,
    // TODO: We should technically only require SmallDimAllocator<_, D>, but Rust gets type
    // inference wrong without this bound...
    DefaultAllocator: TriDimAllocator<f64, D, D, U1>,
    <DefaultAllocator as Allocator<f64, D>>::Buffer: Sync,
{
    let element_name_file_component = element_name.to_ascii_lowercase();

    let mut summary = ErrorSummary {
        element_name: element_name.to_string(),
        L2_errors: vec![],
        H1_seminorm_errors: vec![],
        resolutions: vec![],
    };

    let d = D::dim();
    let base_path = PathBuf::from(format!("data/convergence_tests/poisson_{}d_mms", d));

    for &resolution in resolutions {
        let mesh = mesh_producer(resolution);
        let result = solve_poisson(
            &mesh,
            quadrature.clone(),
            error_quadrature.clone(),
            poisson_source_function,
            &u_exact,
            &u_exact_grad,
        );

        // Resolution measures number of cells per unit-length, and the unit square is one unit
        // long.
        let h = 1.0 / resolution as f64;
        summary.resolutions.push(h);
        summary.L2_errors.push(result.L2_error);
        summary.H1_seminorm_errors.push(result.H1_seminorm_error);

        FiniteElementMeshDataSetBuilder::from_mesh(&mesh)
            .with_title(format!("Poisson {}D FEM {} Res {}", D::dim(), element_name, resolution))
            .with_point_scalar_attributes("u_h", result.u_h.as_slice())
            .try_export(base_path.join(format!(
                "poisson{}d_mms_approx_{}_res_{}.vtu",
                d, element_name_file_component, resolution
            )))
            .unwrap();

        // Evaluate u_exact at mesh vertices
        let u_exact_vector: Vec<_> = mesh.vertices().iter().map(|x| u_exact(x)).collect();

        FiniteElementMeshDataSetBuilder::from_mesh(&mesh)
            .with_title(format!(
                "Poisson {}D FEM {} Exact solution Res {}",
                d, element_name, resolution
            ))
            .with_point_scalar_attributes("u_exact", &u_exact_vector)
            .try_export(base_path.join(format!(
                "poisson{}d_mms_exact_{}_res_{}.vtu",
                d, element_name_file_component, resolution
            )))
            .unwrap();
    }

    let summary_path = base_path.join(format!(
        "poisson{}d_mms_{}_summary.json",
        d, element_name_file_component
    ));
    {
        let mut summary_file = File::create(&summary_path).unwrap();
        serde_json::to_writer_pretty(&mut summary_file, &summary).expect("Failed to write JSON output to directory");
    }

    // Load summary containing reference values
    let reference_summary: ErrorSummary = {
        let reference_summary_path = format!(
            "tests/convergence_tests/reference_values/poisson{}d_mms_{}_summary.json",
            d, element_name_file_component
        );
        let summary_file = File::open(&reference_summary_path).expect(&format!(
            "Failed to open reference error summary for element {}",
            element_name
        ));
        serde_json::from_reader(&summary_file).expect(&format!(
            "Failed to deserialize reference summary for element {}",
            element_name
        ))
    };

    assert_summary_is_close_to_reference(&summary, &reference_summary);
}

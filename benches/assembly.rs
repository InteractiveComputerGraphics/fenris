use criterion::{criterion_group, criterion_main, Criterion};
use fenris::assembly::global::CsrAssembler;
use fenris::assembly::local::{ElementEllipticAssemblerBuilder, QuadratureTable};
use fenris::assembly::operators::LaplaceOperator;
use fenris::element::ElementConnectivity;
use fenris::mesh::procedural::create_unit_box_uniform_tet_mesh_3d;
use fenris::mesh::Mesh;
use fenris::quadrature::CanonicalStiffnessQuadrature;
use fenris::SmallDim;
use fenris_traits::allocators::DimAllocator;
use nalgebra::{DVector, DVectorSlice, DefaultAllocator};
use nalgebra_sparse::pattern::SparsityPattern;
use nalgebra_sparse::CsrMatrix;
use std::hint::black_box;
use fenris_solid::MaterialEllipticOperator;
use fenris_solid::materials::{LameParameters, LinearElasticMaterial};

fn assemble_poisson_into_serial<D, C>(
    matrix: &mut CsrMatrix<f64>,
    assembler: &CsrAssembler<f64>,
    u: DVectorSlice<f64>,
    qtable: &impl QuadratureTable<f64, D, Data = ()>,
    mesh: &Mesh<f64, D, C>,
) -> eyre::Result<()>
where
    D: SmallDim,
    C: ElementConnectivity<f64, GeometryDim = D, ReferenceDim = D>,
    DefaultAllocator: DimAllocator<f64, D>,
{
    let element_assembler = ElementEllipticAssemblerBuilder::new()
        .with_u(u)
        .with_finite_element_space(mesh)
        .with_operator(&LaplaceOperator)
        .with_quadrature_table(qtable)
        .build();
    assembler.assemble_into_csr(matrix, &element_assembler)
}

fn assemble_poisson_pattern_serial<D, C>(
    assembler: &CsrAssembler<f64>,
    u: DVectorSlice<f64>,
    qtable: &impl QuadratureTable<f64, D, Data = ()>,
    mesh: &Mesh<f64, D, C>,
) -> SparsityPattern
where
    D: SmallDim,
    C: ElementConnectivity<f64, GeometryDim = D, ReferenceDim = D>,
    DefaultAllocator: DimAllocator<f64, D>,
{
    let element_assembler = ElementEllipticAssemblerBuilder::new()
        .with_u(u)
        .with_finite_element_space(mesh)
        .with_operator(&LaplaceOperator)
        .with_quadrature_table(qtable)
        .build();
    assembler.assemble_pattern(&element_assembler)
}

fn assemble_elasticity_pattern_serial<D, C>(
    assembler: &CsrAssembler<f64>,
    u: DVectorSlice<f64>,
    qtable: &impl QuadratureTable<f64, D, Data = LameParameters<f64>>,
    mesh: &Mesh<f64, D, C>,
) -> SparsityPattern
    where
        D: SmallDim,
        C: ElementConnectivity<f64, GeometryDim = D, ReferenceDim = D>,
        DefaultAllocator: DimAllocator<f64, D>,
{
    let material = LinearElasticMaterial;
    let operator = MaterialEllipticOperator::new(&material);
    let element_assembler = ElementEllipticAssemblerBuilder::new()
        .with_u(u)
        .with_finite_element_space(mesh)
        .with_operator(&operator)
        .with_quadrature_table(qtable)
        .build();
    assembler.assemble_pattern(&element_assembler)
}

pub fn poisson_assembly_serial(c: &mut Criterion) {
    let resolutions = vec![5, 10, 20];
    let assembler = CsrAssembler::default();
    for res in resolutions {
        let tet4_mesh = create_unit_box_uniform_tet_mesh_3d(res);
        let pattern = assembler.assemble_pattern(&tet4_mesh);
        let nnz = pattern.nnz();
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, vec![0.0; nnz]).unwrap();
        let u = DVector::repeat(matrix.nrows(), 0.0);
        let qtable = tet4_mesh.canonical_stiffness_quadrature();
        c.bench_function(
            &format!("serial assembly poisson stiffness matrix tet4 (res={res})"),
            |b| {
                b.iter(|| {
                    assemble_poisson_into_serial(&mut matrix, &assembler, DVectorSlice::from(&u), &qtable, &tet4_mesh)
                })
            },
        );
    }
}

pub fn poisson_pattern_assembly_serial(c: &mut Criterion) {
    let resolutions = vec![5, 10, 20];
    let assembler = CsrAssembler::default();
    for res in resolutions {
        let tet4_mesh = create_unit_box_uniform_tet_mesh_3d(res);
        let u = DVector::repeat(tet4_mesh.vertices().len(), 0.0);
        let qtable = tet4_mesh.canonical_stiffness_quadrature();
        c.bench_function(
            &format!("serial pattern assembly poisson stiffness matrix tet4 (res={res})"),
            |b| {
                b.iter(|| {
                    black_box(assemble_poisson_pattern_serial(
                        &assembler,
                        DVectorSlice::from(&u),
                        &qtable,
                        &tet4_mesh,
                    ))
                })
            },
        );
    }
}

pub fn elasticity_3d_pattern_assembly_serial(c: &mut Criterion) {
    let resolutions = vec![5, 10, 20];
    let assembler = CsrAssembler::default();
    for res in resolutions {
        let tet4_mesh = create_unit_box_uniform_tet_mesh_3d(res);
        let u = DVector::repeat(tet4_mesh.vertices().len(), 0.0);
        let qtable = tet4_mesh.canonical_stiffness_quadrature()
            .with_uniform_data(LameParameters::default());
        c.bench_function(
            &format!("serial pattern assembly elasticity stiffness matrix tet4 (res={res})"),
            |b| {
                b.iter(|| {
                    black_box(assemble_elasticity_pattern_serial(
                        &assembler,
                        DVectorSlice::from(&u),
                        &qtable,
                        &tet4_mesh,
                    ))
                })
            },
        );
    }
}

criterion_group!(
    serial_assembly,
    poisson_assembly_serial,
    poisson_pattern_assembly_serial,
    elasticity_3d_pattern_assembly_serial,
);

criterion_main!(serial_assembly);

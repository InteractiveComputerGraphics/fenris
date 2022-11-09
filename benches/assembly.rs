use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{DefaultAllocator, DimName, DVector, DVectorSlice};
use nalgebra_sparse::CsrMatrix;
use fenris::assembly::global::CsrAssembler;
use fenris::assembly::local::{ElementEllipticAssemblerBuilder, QuadratureTable, UniformQuadratureTable};
use fenris::assembly::operators::LaplaceOperator;
use fenris::element::ElementConnectivity;
use fenris::mesh::Mesh;
use fenris::mesh::procedural::create_unit_box_uniform_tet_mesh_3d;
use fenris::quadrature::CanonicalStiffnessQuadrature;
use fenris::SmallDim;
use fenris_traits::allocators::{BiDimAllocator, DimAllocator};

fn assemble_poisson_into_serial<D, C>(matrix: &mut CsrMatrix<f64>,
                                      assembler: &CsrAssembler<f64>,
                                      u: DVectorSlice<f64>,
                                      qtable: &impl QuadratureTable<f64, D, Data=()>,
                                      mesh: &Mesh<f64, D, C>)
-> eyre::Result<()>
where
    D: SmallDim,
    C: ElementConnectivity<f64, GeometryDim=D, ReferenceDim=D>,
    DefaultAllocator: DimAllocator<f64, D>
{
    let element_assembler = ElementEllipticAssemblerBuilder::new()
        .with_u(u)
        .with_finite_element_space(mesh)
        .with_operator(&LaplaceOperator)
        .with_quadrature_table(qtable)
        .build();
    assembler.assemble_into_csr(matrix, &element_assembler)
}

pub fn assembly(c: &mut Criterion) {
    let resolutions = vec![10, 20];
    let assembler = CsrAssembler::default();
    for res in resolutions {
        let tet4_mesh = create_unit_box_uniform_tet_mesh_3d(res);
        let pattern = assembler.assemble_pattern(&tet4_mesh);
        let nnz = pattern.nnz();
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, vec![0.0; nnz]).unwrap();
        let mut u = DVector::repeat(matrix.nrows(), 0.0);
        let qtable = tet4_mesh.canonical_stiffness_quadrature();
        c.bench_function(&format!("serial assembly poisson stiffness matrix tet4 (res={res})"),
                         |b| b.iter(|| assemble_poisson_into_serial(&mut matrix, &assembler, DVectorSlice::from(&u), &qtable, &tet4_mesh)));
    }


}

criterion_group!(benches, assembly);
criterion_main!(benches);
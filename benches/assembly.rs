use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fenris::geometry::polymesh::PolyMesh3d;
use fenris::mesh::{HexMesh, Tet10Mesh, Tet4Mesh};
use fenris::model::NodalModel;
use fenris::procedural::create_rectangular_uniform_hex_mesh;
use fenris::quadrature::{tet_quadrature_strength_1, tet_quadrature_strength_5};
use fenris_solid::materials::{StableNeoHookeanMaterial, YoungPoisson};
use fenris_solid::{ElasticMaterialModel, ElasticityModel, ElasticityModelParallel};
use nalgebra::{DMatrix, DMatrixSliceMut, DVector, Dynamic, Matrix3, MatrixMN, MatrixSliceMN, U3};
use std::convert::TryFrom;
use std::ops::Add;

fn test_mesh() -> HexMesh<f64> {
    create_rectangular_uniform_hex_mesh(1.0, 1, 1, 1, 4)
}

pub fn stable_neo_hookean_contraction_3d(c: &mut Criterion) {
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });
    let deformation_gradient = Matrix3::new(
        52.85734952,
        -19.73633697,
        -30.87845429,
        26.67331831,
        -10.35380109,
        -16.15435165,
        58.20810674,
        -20.01345825,
        -31.60294891,
    );

    let contraction_vectors = black_box(MatrixMN::<f64, U3, Dynamic>::from_row_slice(&[
        0.79823853, 0.53879483, 0.6145651, 0.56647738, 0.80380162, 0.81328391, 0.92302888,
        0.81700116, 0.40729593, 0.36585753, 0.42701343, 0.69061995, 0.90149585, 0.17902489,
        0.29973298, 0.8654594, 0.39307017, 0.33597961, 0.89614737, 0.03698405, 0.9097741,
        0.90695223, 0.05189938, 0.49869605, 0.32052228, 0.44186043, 0.32517814, 0.16204256,
        0.14232612, 0.707076,
    ]));

    let output_dim = 3 * contraction_vectors.ncols();
    let mut output = DMatrix::zeros(output_dim, output_dim);
    c.bench_function("stable_neo_hookean_contraction_3d", |b| {
        b.iter(|| {
            material.contract_multiple_stress_tensors_into(
                &mut DMatrixSliceMut::from(&mut output),
                &black_box(deformation_gradient.clone()),
                &MatrixSliceMN::from(&contraction_vectors),
            )
        })
    });
}

pub fn stable_neo_hookean_assemble_tet10_mesh(c: &mut Criterion) {
    let mesh = test_mesh();
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
    let mesh = Tet10Mesh::from(&mesh);
    let quadrature = tet_quadrature_strength_5();

    let model = NodalModel::from_mesh_and_quadrature(mesh, quadrature);

    let u = DVector::zeros(model.ndof());
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });

    c.bench_function("stable_neo_hookean_assemble_tet10_mesh", |b| {
        b.iter(|| {
            model.assemble_stiffness(&u, &material);
        })
    });

    c.bench_function("stable_neo_hookean_assemble_tet10_mesh_to_csr", |b| {
        b.iter(|| {
            model.assemble_stiffness(&u, &material).to_csr(Add::add);
        })
    });
}

pub fn stable_neo_hookean_assemble_tet10_mesh_into_csr(c: &mut Criterion) {
    let mesh = test_mesh();
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
    let mesh = Tet10Mesh::from(&mesh);
    let quadrature = tet_quadrature_strength_5();

    let model = NodalModel::from_mesh_and_quadrature(mesh, quadrature);

    let u = DVector::zeros(model.ndof());
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });

    let mut csr = model.assemble_stiffness(&u, &material).to_csr(Add::add);

    c.bench_function("stable_neo_hookean_assemble_tet10_mesh_into_csr", |b| {
        b.iter(|| {
            model.assemble_stiffness_into(&mut csr, &u, &material);
        })
    });
}
pub fn stable_neo_hookean_assemble_tet4_mesh(c: &mut Criterion) {
    let mesh = test_mesh();
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
    let quadrature = tet_quadrature_strength_1();

    let model = NodalModel::from_mesh_and_quadrature(mesh, quadrature);

    let u = DVector::zeros(model.ndof());
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });

    c.bench_function("stable_neo_hookean_assemble_tet4_mesh", |b| {
        b.iter(|| {
            model.assemble_stiffness(&u, &material);
        })
    });

    c.bench_function("stable_neo_hookean_assemble_tet4_mesh_to_csr", |b| {
        b.iter(|| {
            model.assemble_stiffness(&u, &material).to_csr(Add::add);
        })
    });
}

pub fn stable_neo_hookean_assemble_tet4_mesh_into_csr(c: &mut Criterion) {
    let mesh = test_mesh();
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
    let quadrature = tet_quadrature_strength_1();

    let model = NodalModel::from_mesh_and_quadrature(mesh, quadrature);

    let u = DVector::zeros(model.ndof());
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });

    let mut csr = model.assemble_stiffness(&u, &material).to_csr(Add::add);

    c.bench_function("stable_neo_hookean_assemble_tet4_mesh_into_csr", |b| {
        b.iter(|| {
            model.assemble_stiffness_into(&mut csr, &u, &material);
        })
    });
}

pub fn stable_neo_hookean_assemble_tet10_mesh_parallel(c: &mut Criterion) {
    let mesh = test_mesh();
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
    let mesh = Tet10Mesh::from(&mesh);
    let quadrature = tet_quadrature_strength_5();

    let model = NodalModel::from_mesh_and_quadrature(mesh, quadrature);

    let u = DVector::zeros(model.ndof());
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });

    c.bench_function("stable_neo_hookean_assemble_tet10_mesh_parallel", |b| {
        b.iter(|| {
            model.assemble_stiffness_par(&u, &material);
        })
    });

    c.bench_function(
        "stable_neo_hookean_assemble_tet10_mesh_to_csr_parallel",
        |b| {
            b.iter(|| {
                model.assemble_stiffness_par(&u, &material).to_csr(Add::add);
            })
        },
    );
}

pub fn stable_neo_hookean_assemble_tet10_mesh_into_csr_parallel(c: &mut Criterion) {
    let mesh = test_mesh();
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
    let mesh = Tet10Mesh::from(&mesh);
    let quadrature = tet_quadrature_strength_5();

    let model = NodalModel::from_mesh_and_quadrature(mesh, quadrature);

    let u = DVector::zeros(model.ndof());
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });

    let mut csr = model.assemble_stiffness(&u, &material).to_csr(Add::add);

    c.bench_function("stable_neo_hookean_assemble_tet10_mesh_into_csr_par", |b| {
        b.iter(|| {
            model.assemble_stiffness_into_par(&mut csr, &u, &material);
        })
    });
}

pub fn stable_neo_hookean_assemble_tet4_mesh_into_csr_parallel(c: &mut Criterion) {
    let mesh = test_mesh();
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
    let quadrature = tet_quadrature_strength_1();

    let model = NodalModel::from_mesh_and_quadrature(mesh, quadrature);

    let u = DVector::zeros(model.ndof());
    let material = StableNeoHookeanMaterial::from(YoungPoisson {
        young: 1e6,
        poisson: 0.45,
    });

    let mut csr = model.assemble_stiffness(&u, &material).to_csr(Add::add);

    c.bench_function("stable_neo_hookean_assemble_tet4_mesh_into_csr_par", |b| {
        b.iter(|| {
            model.assemble_stiffness_into_par(&mut csr, &u, &material);
        })
    });
}

criterion_group!(
    benches,
    stable_neo_hookean_contraction_3d,
    stable_neo_hookean_assemble_tet4_mesh,
    stable_neo_hookean_assemble_tet4_mesh_into_csr,
    stable_neo_hookean_assemble_tet4_mesh_into_csr_parallel,
    stable_neo_hookean_assemble_tet10_mesh,
    stable_neo_hookean_assemble_tet10_mesh_into_csr,
    stable_neo_hookean_assemble_tet10_mesh_into_csr_parallel,
    stable_neo_hookean_assemble_tet10_mesh_parallel
);
criterion_main!(benches);

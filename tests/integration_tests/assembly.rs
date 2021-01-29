use fenris::geometry::polymesh::PolyMesh3d;
use fenris::model::{Quad4Model, Tet4Model, Tri3d2Model};
use fenris::proptest::rectangular_uniform_mesh_strategy;
use fenris::quadrature::{
    quad_quadrature_strength_5_f64, tet_quadrature_strength_5, tri_quadrature_strength_5_f64,
};
use fenris_optimize::calculus::{approximate_jacobian, VectorFunctionBuilder};
use fenris_solid::materials::{LinearElasticMaterial, YoungPoisson};
use fenris_solid::{ElasticityModel, ElasticityModelParallel};
use nalgebra::{DVector, DVectorSlice, DVectorSliceMut};

use fenris::mesh::Tet4Mesh;
use fenris::procedural::{
    create_rectangular_uniform_hex_mesh, create_unit_square_uniform_quad_mesh_2d,
};
use proptest::prelude::*;
use std::convert::TryFrom;
use std::ops::Add;

use util::assert_approx_matrix_eq;

#[test]
fn tet4_stiffness_matrix_is_negative_derivative_of_forces() {
    // TODO: Make a property-based test out of this rather than choosing a fixed mesh
    let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();

    let h = 1e-6;

    let lame = YoungPoisson {
        young: 1e6,
        poisson: 0.2,
    };
    let material = LinearElasticMaterial::from(lame);
    let quadrature = tet_quadrature_strength_5();
    let model = Tet4Model::from_mesh_and_quadrature(mesh.clone(), quadrature);

    let u = DVector::zeros(model.ndof());
    let a = model.assemble_stiffness(&u, &material).build_dense();

    let func = VectorFunctionBuilder::with_dimension(model.ndof()).with_function(move |f, u| {
        f.copy_from(&model.assemble_elastic_pseudo_forces(*u, &material));
    });

    let a_approx = -approximate_jacobian(func, &u, &h);
    let diff = &a - &a_approx;

    let approx_equals = diff.norm() / (a.norm() + a_approx.norm()) < 1e-5;
    assert!(approx_equals);
}

#[test]
fn tri3d2_stiffness_matrix_csr_and_coo_assembly_agree() {
    // let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = create_unit_square_uniform_quad_mesh_2d(3).split_into_triangles();

    let lame = YoungPoisson {
        young: 1e6,
        poisson: 0.2,
    };
    let material = LinearElasticMaterial::from(lame);
    let quadrature = tri_quadrature_strength_5_f64();
    let model = Tri3d2Model::from_mesh_and_quadrature(mesh.clone(), quadrature);

    let u = DVector::zeros(model.ndof());
    let a_coo_csr = model.assemble_stiffness(&u, &material).to_csr(Add::add);

    let mut a_csr = a_coo_csr.clone();
    a_csr.transform_values(|_, _, val| *val = 0.0);
    model.assemble_stiffness_into(&mut a_csr, &u, &material);

    assert_eq!(a_coo_csr.nnz(), a_csr.nnz());

    let a_coo_csr_dense = a_coo_csr.build_dense();
    let abstol = 1e-14 * a_coo_csr_dense.abs().max();

    assert_approx_matrix_eq!(&a_coo_csr_dense, &a_csr.build_dense(), abstol = abstol);
}

#[test]
fn tet4_stiffness_matrix_csr_and_coo_assembly_agree() {
    // let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();

    let lame = YoungPoisson {
        young: 1e6,
        poisson: 0.2,
    };
    let material = LinearElasticMaterial::from(lame);
    let quadrature = tet_quadrature_strength_5();
    let model = Tet4Model::from_mesh_and_quadrature(mesh.clone(), quadrature);

    let u = DVector::zeros(model.ndof());
    let a_coo_csr = model.assemble_stiffness(&u, &material).to_csr(Add::add);

    let mut a_csr = a_coo_csr.clone();
    a_csr.transform_values(|_, _, val| *val = 0.0);
    model.assemble_stiffness_into(&mut a_csr, &u, &material);

    assert_eq!(a_coo_csr.nnz(), a_csr.nnz());

    let a_coo_csr_dense = a_coo_csr.build_dense();
    let abstol = 1e-14 * a_coo_csr_dense.abs().max();

    assert_approx_matrix_eq!(&a_coo_csr_dense, &a_csr.build_dense(), abstol = abstol);
}

#[test]
fn tet4_elastic_forces_sequential_and_parallel_agree() {
    // let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();

    let lame = YoungPoisson {
        young: 1e6,
        poisson: 0.2,
    };
    let material = LinearElasticMaterial::from(lame);
    let quadrature = tet_quadrature_strength_5();
    let model = Tet4Model::from_mesh_and_quadrature(mesh.clone(), quadrature);
    let u = DVector::from_iterator(
        model.ndof(),
        (0..10).map(|i| i as f64).cycle().take(model.ndof()),
    );

    let f_seq = model.assemble_elastic_pseudo_forces(DVectorSlice::from(&u), &material);

    let mut f_par = DVector::zeros(model.ndof());
    model.assemble_elastic_pseudo_forces_into_par(
        DVectorSliceMut::from(&mut f_par),
        DVectorSlice::from(&u),
        &material,
    );

    assert_eq!(f_seq.len(), f_par.len());

    let abstol = 1e-14 * f_seq.abs().max();
    assert_approx_matrix_eq!(&f_seq, &f_par, abstol = abstol);
}

#[test]
fn tet4_stiffness_matrix_csr_and_csr_par_agree() {
    // let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = create_rectangular_uniform_hex_mesh(1.0, 2, 2, 2, 1);
    let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();

    let lame = YoungPoisson {
        young: 1e6,
        poisson: 0.2,
    };
    let material = LinearElasticMaterial::from(lame);
    let quadrature = tet_quadrature_strength_5();
    let model = Tet4Model::from_mesh_and_quadrature(mesh.clone(), quadrature);

    let u = DVector::zeros(model.ndof());
    // Construct CSR matrix with correct pattern
    // TODO: Implement direct construction of pattern
    let a_coo_csr = model.assemble_stiffness(&u, &material).to_csr(Add::add);

    let a_csr = {
        let mut a_csr = a_coo_csr.clone();
        a_csr.transform_values(|_, _, val| *val = 0.0);
        model.assemble_stiffness_into(&mut a_csr, &u, &material);
        a_csr
    };

    let a_csr_par = {
        let mut a_csr_par = a_coo_csr.clone();
        a_csr_par.transform_values(|_, _, val| *val = 0.0);
        model.assemble_stiffness_into_par(&mut a_csr_par, &u, &material);
        a_csr_par
    };

    assert_eq!(a_csr.nnz(), a_csr_par.nnz());

    let a_csr_dense = a_csr.build_dense();
    let abstol = 1e-14 * dbg!(a_csr_dense.abs().max());

    assert_approx_matrix_eq!(&a_csr_dense, &a_csr_par.build_dense(), abstol = abstol);
}

proptest! {
    #[test]
    fn stiffness_matrix_is_negative_derivative_of_forces_for_rectangular_grids_with_linear_material_bilinear_quad(
        mesh in rectangular_uniform_mesh_strategy(1.0, 4)) {
        prop_assume!(mesh.connectivity().len() > 0);

        let h = 1e-6;

        let lame = YoungPoisson {
            young: 1e6,
            poisson: 0.2,
        };
        let material = LinearElasticMaterial::from(lame);
        let quadrature = quad_quadrature_strength_5_f64();
        let model = Quad4Model::from_mesh_and_quadrature(mesh.clone(), quadrature);

        let u = DVector::zeros(model.ndof());
        let a = model.assemble_stiffness(&u, &material).build_dense();

        let func = VectorFunctionBuilder::with_dimension(model.ndof())
                .with_function(move |f, u| {
                    f.copy_from(&model.assemble_elastic_pseudo_forces(*u, &material));
                });

        let a_approx = - approximate_jacobian(func, &u, &h);
        let diff = &a - &a_approx;

        let approx_equals = diff.norm() / (a.norm() + a_approx.norm()) < 1e-5;
        prop_assert!(approx_equals);
    }

    #[test]
    fn stiffness_matrix_is_negative_derivative_of_forces_for_rectangular_domain_with_linear_material_linear_tri(
        mesh in rectangular_uniform_mesh_strategy(1.0, 4)) {
        let mesh = mesh.split_into_triangles();
        prop_assume!(mesh.connectivity().len() > 0);

        let h = 1e-6;

        let lame = YoungPoisson {
            young: 1e6,
            poisson: 0.2,
        };
        let material = LinearElasticMaterial::from(lame);
        let quadrature = tri_quadrature_strength_5_f64();
        let model = Tri3d2Model::from_mesh_and_quadrature(mesh.clone(), quadrature);

        let u = DVector::zeros(model.ndof());
        let a = model.assemble_stiffness(&u, &material).build_dense();

        let func = VectorFunctionBuilder::with_dimension(model.ndof())
                .with_function(move |f, u| {
                    f.copy_from(&model.assemble_elastic_pseudo_forces(*u, &material));
                });

        let a_approx = - approximate_jacobian(func, &u, &h);
        let diff = &a - &a_approx;

        let approx_equals = diff.norm() / (a.norm() + a_approx.norm()) < 1e-5;
        prop_assert!(approx_equals);
    }

    // TODO: Need way more tests for linear tri elements!
}

use crate::unit_tests::{deformation_gradient_2d, deformation_gradient_3d, lame_parameters, tet10_element};
use fenris::assembly::local::{
    assemble_element_elliptic_matrix, assemble_element_elliptic_vector, compute_element_elliptic_energy,
};
use fenris::assembly::operators::{EllipticContraction, EllipticEnergy, EllipticOperator};
use fenris::nalgebra;
use fenris::nalgebra::{
    vector, DMatrix, DVector, DVectorSlice, DVectorSliceMut, DimName, Dynamic, Matrix2, Matrix3, MatrixSliceMut,
    OMatrix, U3,
};
use fenris::quadrature;
use fenris_optimize::calculus::{approximate_gradient_fd, approximate_jacobian_fd};
use fenris_solid::materials::StVKMaterial;
use fenris_solid::{HyperelasticMaterial, MaterialEllipticOperator};
use matrixcompare::assert_matrix_eq;

#[test]
#[allow(non_snake_case)]
fn material_elliptic_operator_stvk_2d() {
    // Just check that energy and derivatives as an elliptic operator is consistent
    // We use StVK because it's the simplest non-linear energy that we have
    // (so that its contraction actually depends on F)
    let F = deformation_gradient_2d();
    let lame = lame_parameters();
    let material = StVKMaterial;

    let operator = MaterialEllipticOperator::new(&material);
    // MaterialEllipticOperator uses displacement (u) as solution field
    let u_grad = F.transpose() - Matrix2::identity();

    // We have that psi_elliptic(grad u) = psi_material(F)
    // with F = transpose(grad u)
    let energy_as_elliptic = operator.compute_energy(&u_grad, &lame);
    let energy_as_material = material.compute_energy_density(&F, &lame);
    assert_eq!(energy_as_elliptic, energy_as_material);

    // We have that g(grad u) = P(F)^T
    let derivative_as_elliptic = operator.compute_elliptic_operator(&u_grad, &lame);
    let derivative_as_stress = material.compute_stress_tensor(&F, &lame);
    assert_matrix_eq!(derivative_as_elliptic, derivative_as_stress.transpose());

    // For the contraction we expect that C_g(grad u, a, b) = C_P(F, a, b)
    let a = vector![3.0, 4.0];
    let b = vector![-3.0, 1.0];
    let contraction_as_elliptic = operator.contract(&u_grad, &a, &b, &lame);
    let contraction_as_material = material.compute_stress_contraction(&F, &a, &b, &lame);
    assert_matrix_eq!(contraction_as_elliptic, contraction_as_material);
}

#[test]
#[allow(non_snake_case)]
fn material_elliptic_operator_stvk_3d() {
    // Same as 2d test, see comments there
    let F = deformation_gradient_3d();
    let lame = lame_parameters();
    let material = StVKMaterial;

    let operator = MaterialEllipticOperator::new(&material);
    // MaterialEllipticOperator uses displacement (u) as solution field
    let u_grad = F.transpose() - Matrix3::identity();

    let energy_as_elliptic = operator.compute_energy(&u_grad, &lame);
    let energy_as_material = material.compute_energy_density(&F, &lame);
    assert_eq!(energy_as_elliptic, energy_as_material);

    let derivative_as_elliptic = operator.compute_elliptic_operator(&u_grad, &lame);
    let derivative_as_stress = material.compute_stress_tensor(&F, &lame);
    assert_matrix_eq!(derivative_as_elliptic, derivative_as_stress.transpose());

    let a = vector![3.0, 4.0, -2.0];
    let b = vector![-3.0, 1.0, 3.0];
    let contraction_as_elliptic = operator.contract(&u_grad, &a, &b, &lame);
    let contraction_as_material = material.compute_stress_contraction(&F, &a, &b, &lame);
    assert_matrix_eq!(contraction_as_elliptic, contraction_as_material);
}

#[test]
#[allow(non_snake_case)]
fn material_elliptic_operator_vector_matrix_assembly_tet10() {
    // We assume (from other tests) that the elliptic energy is correctly computed for the material elliptic operator.
    // Then we check that the element vector assembled for an arbitrary element is in fact the derivative of the
    // element energy, and that the element matrix is the derivative of the element vector.

    let lame = lame_parameters();
    let material = StVKMaterial;
    let operator = MaterialEllipticOperator::new(&material);
    let element = tet10_element();

    // Check (using finite differences) that the discrete element vector is consistent with the energy,
    // and that the discrete element matrix is consistent with the element vector
    let u_element = DVector::from_iterator(30, (0..30).map(|i| i as f64));

    let (weights, points) = quadrature::total_order::tetrahedron(2).unwrap();
    let parameters: Vec<_> = std::iter::repeat(lame).take(points.len()).collect();

    let mut gradient_buffer = OMatrix::zeros_generic(U3::name(), Dynamic::new(10));
    let mut element_vector = DVector::zeros(30);
    assemble_element_elliptic_vector(
        DVectorSliceMut::from(&mut element_vector),
        &element,
        &operator,
        DVectorSlice::from(&u_element),
        &weights,
        &points,
        &parameters,
        MatrixSliceMut::from(&mut gradient_buffer),
    )
    .unwrap();

    let mut element_matrix = DMatrix::zeros(30, 30);
    assemble_element_elliptic_matrix(
        MatrixSliceMut::from(&mut element_matrix),
        &element,
        &operator,
        DVectorSlice::from(&u_element),
        &weights,
        &points,
        &parameters,
        MatrixSliceMut::from(&mut gradient_buffer),
    )
    .unwrap();

    let mut u_h = u_element.clone();
    let approx_element_vector = approximate_gradient_fd(
        |u| {
            let mut gradient_buffer = OMatrix::zeros_generic(U3::name(), Dynamic::new(10));
            compute_element_elliptic_energy(
                &element,
                &operator,
                u,
                &weights,
                &points,
                &parameters,
                MatrixSliceMut::from(&mut gradient_buffer),
            )
            .unwrap()
        },
        &mut u_h,
        1e-5,
    );

    let approx_element_matrix = approximate_jacobian_fd(
        30,
        |u, f_grad| {
            let mut gradient_buffer = OMatrix::zeros_generic(U3::name(), Dynamic::new(10));
            assemble_element_elliptic_vector(
                f_grad,
                &element,
                &operator,
                u,
                &weights,
                &points,
                &parameters,
                MatrixSliceMut::from(&mut gradient_buffer),
            )
            .unwrap();
        },
        &mut u_h,
        1e-5,
    );

    assert_matrix_eq!(
        element_vector,
        approx_element_vector,
        comp = abs,
        tol = 1e-9 * element_vector.amax()
    );
    assert_matrix_eq!(
        element_matrix,
        approx_element_matrix,
        comp = abs,
        tol = 1e-9 * element_matrix.amax()
    );
}

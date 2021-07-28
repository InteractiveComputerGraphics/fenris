use fenris::nalgebra;
use fenris::nalgebra::{matrix, vector, Matrix2, Matrix3, SMatrix, SVector};
use fenris_solid::materials::{LameParameters, LinearElasticMaterial, YoungPoisson};
use fenris_solid::HyperelasticMaterial;
use matrixcompare::{assert_matrix_eq, assert_scalar_eq};

/// Approximates stress tensor using central Finite Differences with step size `h`.
#[allow(non_snake_case)]
fn approximate_stress_tensor_fd<const D: usize>(
    strain_energy_density: impl Fn(&SMatrix<f64, D, D>) -> f64,
    deformation_gradient: SMatrix<f64, D, D>,
    h: f64,
) -> SMatrix<f64, D, D> {
    let mut stress_tensor = SMatrix::zeros();

    let mut F = deformation_gradient;

    for i in 0..D {
        for j in 0..D {
            let f_ij = F[(i, j)];
            F[(i, j)] = f_ij + h;
            let psi_plus = strain_energy_density(&F);
            F[(i, j)] = f_ij - h;
            let psi_minus = strain_energy_density(&F);
            F[(i, j)] = f_ij;

            stress_tensor[(i, j)] = (psi_plus - psi_minus) / (2.0 * h);
        }
    }

    stress_tensor
}

/// Approximates the stress contraction using central Finite Differences with step size `h`.
#[allow(non_snake_case)]
fn approximate_stress_contraction_fd<const D: usize>(
    stress_tensor: impl Fn(&SMatrix<f64, D, D>) -> SMatrix<f64, D, D>,
    deformation_gradient: SMatrix<f64, D, D>,
    a: SVector<f64, D>,
    b: SVector<f64, D>,
    h: f64,
) -> SMatrix<f64, D, D> {
    let mut contraction = SMatrix::zeros();

    let mut F = deformation_gradient;

    // We have dP_ik / dF_jl as the "inner" derivative information. So we use jl as outer indices
    // so that we can compute the "full" stress tensor and use it in the FD expression
    for j in 0..D {
        for l in 0..D {
            let F_jl = F[(j, l)];
            F[(j, l)] = F_jl + h;
            let P_plus = stress_tensor(&F);
            F[(j, l)] = F_jl - h;
            let P_minus = stress_tensor(&F);
            F[(j, l)] = F_jl;

            let dP_dFjl = (P_plus - P_minus) / (2.0 * h);
            for i in 0..D {
                for k in 0..D {
                    let dPik_dFjl = dP_dFjl[(i, k)];
                    contraction[(i, j)] += a[k] * dPik_dFjl * b[l];
                }
            }
        }
    }

    contraction
}

#[test]
fn lame_from_young_poisson() {
    let young_poisson = YoungPoisson {
        young: 1e3,
        poisson: 0.3,
    };
    let lame = LameParameters::from(young_poisson);

    assert_scalar_eq!(lame.mu, 384.6153846153846, comp = float);
    assert_scalar_eq!(lame.lambda, 576.9230769230769, comp = float);
}

fn test_lame_parameters() -> LameParameters<f64> {
    LameParameters {
        mu: 384.0,
        lambda: 577.0,
    }
}

fn test_deformation_gradient_2d() -> Matrix2<f64> {
    matrix![1.0, 2.0;
            3.0, 4.0]
}

fn test_deformation_gradient_3d() -> Matrix3<f64> {
    matrix![1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0]
}

#[test]
fn linear_elastic_strain_energy_2d() {
    let lame = test_lame_parameters();
    let deformation_gradient = test_deformation_gradient_2d();
    let psi = LinearElasticMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 10852.5, comp = float);
}

#[test]
#[allow(non_snake_case)]
fn linear_elastic_stress_tensor_is_derivative_of_energy_2d() {
    let lame = test_lame_parameters();

    let deformation_gradient = test_deformation_gradient_2d();
    let stress_tensor = LinearElasticMaterial.compute_stress_tensor(&deformation_gradient, &lame);

    let h = 1e-5;
    let approx_stress_tensor = approximate_stress_tensor_fd(
        |F| LinearElasticMaterial.compute_energy_density(F, &lame),
        deformation_gradient,
        h,
    );

    // TODO: Find a way to get more accurate approximation so we can use stricter tolerances
    assert_matrix_eq!(stress_tensor, approx_stress_tensor, comp = abs, tol = 1e-5);
}

#[test]
#[allow(non_snake_case)]
fn linear_elastic_stress_contraction_is_consistent_with_tensor_3d() {
    let lame = test_lame_parameters();
    let deformation_gradient = test_deformation_gradient_2d();
    let a = vector![-3.0, 4.0];
    let b = vector![5.0, -2.0];
    let contraction = LinearElasticMaterial.compute_stress_contraction(&deformation_gradient, &a, &b, &lame);

    let h = 1e-5;
    let approx_contraction = approximate_stress_contraction_fd(
        |F| LinearElasticMaterial.compute_stress_tensor(F, &lame),
        deformation_gradient,
        a,
        b,
        h,
    );

    // TODO: Find a way to get more accurate approximation so we can use stricter tolerances
    assert_matrix_eq!(contraction, approx_contraction, comp = abs, tol = 1e-5);
}

#[test]
fn linear_elastic_strain_energy_3d() {
    let lame = test_lame_parameters();
    let deformation_gradient = test_deformation_gradient_3d();
    let psi = LinearElasticMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 136008.0, comp = float);
}

#[test]
#[allow(non_snake_case)]
fn linear_elastic_stress_tensor_is_derivative_of_energy_3d() {
    let lame = test_lame_parameters();

    let deformation_gradient = test_deformation_gradient_3d();
    let stress_tensor = LinearElasticMaterial.compute_stress_tensor(&deformation_gradient, &lame);

    let h = 1e-5;
    let approx_stress_tensor = approximate_stress_tensor_fd(
        |F| LinearElasticMaterial.compute_energy_density(F, &lame),
        deformation_gradient,
        h,
    );

    // TODO: Find a way to get more accurate approximation so we can use stricter tolerances
    assert_matrix_eq!(stress_tensor, approx_stress_tensor, comp = abs, tol = 1e-5);
}

#[test]
#[allow(non_snake_case)]
fn linear_elastic_stress_contraction_is_consistent_with_tensor_2d() {
    let lame = test_lame_parameters();
    let deformation_gradient = test_deformation_gradient_3d();
    let a = vector![-3.0, 4.0, -5.0];
    let b = vector![-5.0, 2.0, 1.0];
    let contraction = LinearElasticMaterial.compute_stress_contraction(&deformation_gradient, &a, &b, &lame);

    let h = 1e-5;
    let approx_contraction = approximate_stress_contraction_fd(
        |F| LinearElasticMaterial.compute_stress_tensor(F, &lame),
        deformation_gradient,
        a,
        b,
        h,
    );

    // TODO: Find a way to get more accurate approximation so we can use stricter tolerances
    assert_matrix_eq!(contraction, approx_contraction, comp = abs, tol = 1e-5);
}

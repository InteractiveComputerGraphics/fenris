use matrixcompare::{assert_matrix_eq, assert_scalar_eq};

use fenris::nalgebra;
use fenris::nalgebra::{dvector, vector, DMatrix, DMatrixSliceMut, DVectorSlice, SMatrix, SVector};
use fenris_solid::materials::{LameParameters, LinearElasticMaterial, StVKMaterial, YoungPoisson, NeoHookeanMaterial};
use fenris_solid::HyperelasticMaterial;

use crate::unit_tests::{deformation_gradient_2d, deformation_gradient_3d, lame_parameters};

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

/// Uses finite differences to check that the stress tensor is the derivative of the energy
macro_rules! test_stress_is_derivative_of_energy {
    (dim = 2, $material:expr, $test_name: ident) => {
        test_stress_is_derivative_of_energy!($material, $test_name, deformation_gradient_2d());
    };
    (dim = 3, $material:expr, $test_name: ident) => {
        test_stress_is_derivative_of_energy!($material, $test_name, deformation_gradient_3d());
    };
    ($material:expr, $test_name: ident, $deformation_gradient:expr) => {
        #[test]
        #[allow(non_snake_case)]
        fn $test_name() {
            let lame = lame_parameters();
            let deformation_gradient = $deformation_gradient;
            let material = $material;
            let stress_tensor = material.compute_stress_tensor(&deformation_gradient, &lame);

            let h = 1e-5;
            let approx_stress_tensor = approximate_stress_tensor_fd(
                |F| material.compute_energy_density(F, &lame),
                deformation_gradient,
                h,
            );

            assert_matrix_eq!(
                stress_tensor,
                approx_stress_tensor,
                comp = abs,
                tol = 1e-9 * stress_tensor.amax()
            );
        }
    };
}

/// Uses finite differences to check that the contraction operator is consistent with the stress tensor
/// for the given material model.
macro_rules! test_contraction_is_consistent_with_tensor {
    (dim = 2, $material:expr, $test_name: ident) => {
        test_contraction_is_consistent_with_tensor!(
            $material,
            $test_name,
            deformation_gradient_2d(),
            vector![-3.0, 4.0],
            vector![-5.0, 2.0]
        );
    };
    (dim = 3, $material:expr, $test_name: ident) => {
        test_contraction_is_consistent_with_tensor!(
            $material,
            $test_name,
            deformation_gradient_3d(),
            vector![-3.0, 4.0, -5.0],
            vector![-5.0, 2.0, 1.0]
        );
    };
    ($material:expr, $test_name: ident, $deformation_gradient:expr, $a:expr, $b:expr) => {
        #[test]
        #[allow(non_snake_case)]
        fn $test_name() {
            let lame = lame_parameters();
            let deformation_gradient = $deformation_gradient;
            let material = $material;
            let a = $a;
            let b = $b;
            let contraction = material.compute_stress_contraction(&deformation_gradient, &a, &b, &lame);

            let h = 1e-5;
            let approx_contraction = approximate_stress_contraction_fd(
                |F| material.compute_stress_tensor(F, &lame),
                deformation_gradient,
                a,
                b,
                h,
            );

            assert_matrix_eq!(
                contraction,
                approx_contraction,
                comp = abs,
                tol = 1e-9 * contraction.amax()
            );
        }
    };
}

/// Test that the multi-contraction for the given material is consistent with contraction for a single pair
/// of vectors.
macro_rules! test_multi_contraction_consistency {
    (dim = 2, $material:expr, $test_name: ident) => {
        test_multi_contraction_consistency!(
            dim = 2,
            $material,
            $test_name,
            deformation_gradient_2d(),
            dvector![2.0, -3.0, 4.0, 1.0, 3.0, -2.0],
            dvector![-1.0, 2.0, 5.0, -3.0, 2.0, 3.0]
        );
    };
    (dim = 3, $material:expr, $test_name: ident) => {
        test_multi_contraction_consistency!(
            dim = 3,
            $material,
            $test_name,
            deformation_gradient_3d(),
            dvector![2.0, -3.0, 4.0, 1.0, 3.0, -2.0, 0.0, 2.0, -2.0],
            dvector![-1.0, 2.0, 5.0, -3.0, 2.0, 3.0, 1.0, 5.0, -4.0]
        );
    };
    // Implementation detail, not supposed to be called outside of this macro
    (dim = $dim:expr, $material:expr, $test_name: ident, $deformation_gradient:expr, $a:expr, $b:expr) => {
        #[test]
        #[allow(non_snake_case)]
        fn $test_name() {
            let material = $material;
            let (a, b) = ($a, $b);
            let lame = lame_parameters();
            let deformation_gradient = $deformation_gradient;
            let N = 3;
            assert_eq!(a.len(), $dim * N);
            assert_eq!(b.len(), $dim * N);
            let alpha = 2.0;
            // Use a non-zero output matrix to ensure that the accumulation actually *adds* entries and not just
            // overwrites them
            let mut output = DMatrix::repeat(3 * $dim, 3 * $dim, 3.0);
            material.accumulate_stress_contractions_into(
                DMatrixSliceMut::from(&mut output),
                alpha,
                &deformation_gradient,
                DVectorSlice::from(&a),
                DVectorSlice::from(&b),
                &lame,
            );

            // Compare each block in the output matrix to individual calls to compute_stress_contraction
            for I in 0..N {
                for J in I..N {
                    let a_I = a.fixed_rows::<$dim>($dim * I).clone_owned();
                    let b_J = b.fixed_rows::<$dim>($dim * J).clone_owned();
                    let C_IJ = output.fixed_slice::<$dim, $dim>($dim * I, $dim * J);
                    let contraction = material.compute_stress_contraction(&deformation_gradient, &a_I, &b_J, &lame);

                    // The offset value was the value in each block matrix entry before accumulation
                    let offset = SMatrix::<_, $dim, $dim>::repeat(3.0);
                    // TODO: Exact equality might not work for every material! Revise this as needed
                    if I != J {
                        assert_matrix_eq!(C_IJ, alpha * contraction + offset);
                    } else {
                        // For the entries on the diagonal, we only need the upper triangle to match
                        assert_matrix_eq!(
                            C_IJ.upper_triangle(),
                            (alpha * contraction + offset).upper_triangle()
                        );
                    }
                }
            }
        }
    };
}

// Test derivatives of linear elastic material
#[test]
fn linear_elastic_strain_energy_2d() {
    let lame = lame_parameters();
    let deformation_gradient = deformation_gradient_2d();
    let psi = LinearElasticMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 11528.0, comp = float);
}

#[test]
fn linear_elastic_strain_energy_3d() {
    let lame = lame_parameters();
    let deformation_gradient = deformation_gradient_3d();
    let psi = LinearElasticMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 133154.0, comp = float);
}

test_stress_is_derivative_of_energy!(
    dim = 2,
    LinearElasticMaterial,
    linear_elastic_stress_is_derivative_of_energy_2d
);
test_stress_is_derivative_of_energy!(
    dim = 3,
    LinearElasticMaterial,
    linear_elastic_stress_is_derivative_of_energy_3d
);

test_contraction_is_consistent_with_tensor!(
    dim = 2,
    LinearElasticMaterial,
    linear_elastic_stress_contraction_is_consistent_with_tensor_2d
);
test_contraction_is_consistent_with_tensor!(
    dim = 3,
    LinearElasticMaterial,
    linear_elastic_stress_contraction_is_consistent_with_tensor_3d
);

test_multi_contraction_consistency!(
    dim = 2,
    LinearElasticMaterial,
    linear_elastic_multi_contraction_consistency_2d
);
test_multi_contraction_consistency!(
    dim = 3,
    LinearElasticMaterial,
    linear_elastic_multi_contraction_consistency_3d
);

// Tests for StVKMaterial

#[test]
fn stvk_strain_energy_2d() {
    let lame = lame_parameters();
    let deformation_gradient = deformation_gradient_2d();
    let psi = StVKMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 132578.0, comp = float);
}

#[test]
fn stvk_strain_energy_3d() {
    let lame = lame_parameters();
    let deformation_gradient = deformation_gradient_3d();
    let psi = StVKMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 9136789.125, comp = float);
}

test_stress_is_derivative_of_energy!(dim = 2, StVKMaterial, stvk_stress_is_derivative_of_energy_2d);
test_stress_is_derivative_of_energy!(dim = 3, StVKMaterial, stvk_stress_is_derivative_of_energy_3d);

test_contraction_is_consistent_with_tensor!(
    dim = 2,
    StVKMaterial,
    stvk_stress_contraction_is_consistent_with_tensor_2d
);

test_contraction_is_consistent_with_tensor!(
    dim = 3,
    StVKMaterial,
    stvk_stress_contraction_is_consistent_with_tensor_3d
);

test_multi_contraction_consistency!(dim = 2, StVKMaterial, stvk_multi_contraction_consistency_2d);
test_multi_contraction_consistency!(dim = 3, StVKMaterial, stvk_multi_contraction_consistency_3d);

#[test]
fn neo_hookean_strain_energy_2d() {
    let lame = lame_parameters();
    let deformation_gradient = deformation_gradient_2d();
    let psi = NeoHookeanMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 5313.274620288603, comp = float);
}

#[test]
fn neo_hookean_strain_energy_3d() {
    let lame = lame_parameters();
    let deformation_gradient = deformation_gradient_3d();
    let psi = NeoHookeanMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 48833.26962613859, comp = float);
}

test_stress_is_derivative_of_energy!(dim = 2, NeoHookeanMaterial, neo_hookean_stress_is_derivative_of_energy_2d);
test_stress_is_derivative_of_energy!(dim = 3, NeoHookeanMaterial, neo_hookean_stress_is_derivative_of_energy_3d);

test_contraction_is_consistent_with_tensor!(
    dim = 2,
    NeoHookeanMaterial,
    neo_hookean_stress_contraction_is_consistent_with_tensor_2d
);

test_contraction_is_consistent_with_tensor!(
    dim = 3,
    NeoHookeanMaterial,
    neo_hookean_stress_contraction_is_consistent_with_tensor_3d
);

test_multi_contraction_consistency!(dim = 2, NeoHookeanMaterial, neo_hookean_multi_contraction_consistency_2d);
test_multi_contraction_consistency!(dim = 3, NeoHookeanMaterial, neo_hookean_multi_contraction_consistency_3d);
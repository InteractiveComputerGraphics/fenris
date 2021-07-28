use fenris::nalgebra;
use fenris::nalgebra::matrix;
use fenris_solid::materials::{LameParameters, LinearElasticMaterial, YoungPoisson};
use fenris_solid::HyperelasticMaterial;
use matrixcompare::assert_scalar_eq;

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

#[test]
fn linear_elastic_strain_energy_2d() {
    let lame = LameParameters {
        mu: 384.0,
        lambda: 577.0,
    };

    let deformation_gradient = matrix![1.0, 2.0;
                                       3.0, 4.0];
    let psi = LinearElasticMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 10852.5, comp = float);
}

#[test]
fn linear_elastic_strain_energy_3d() {
    let lame = LameParameters {
        mu: 384.0,
        lambda: 577.0,
    };

    let deformation_gradient = matrix![1.0, 2.0, 3.0;
                                       4.0, 5.0, 6.0;
                                       7.0, 8.0, 9.0];
    let psi = LinearElasticMaterial.compute_energy_density(&deformation_gradient, &lame);

    assert_scalar_eq!(psi, 136008.0, comp = float);
}

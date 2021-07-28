use crate::HyperelasticMaterial;
use fenris::allocators::SmallDimAllocator;
use fenris::nalgebra::{DefaultAllocator, DimName, OMatrix, OVector, RealField};
use numeric_literals::replace_float_literals;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LameParameters<T> {
    pub mu: T,
    pub lambda: T,
}

impl<T> Default for LameParameters<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
    fn default() -> Self {
        // TODO: Any sensible default?
        Self { mu: 0.0, lambda: 0.0 }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct YoungPoisson<T> {
    pub young: T,
    pub poisson: T,
}

impl<T> From<YoungPoisson<T>> for LameParameters<T>
where
    T: RealField,
{
    #[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
    fn from(params: YoungPoisson<T>) -> Self {
        // TODO: Test this!
        let YoungPoisson { young, poisson } = params;
        let mu = 0.5 * young / (1.0 + poisson);
        let lambda = 2.0 * mu * poisson / (1.0 - 2.0 * poisson);
        Self { mu, lambda }
    }
}

/// A linear elastic material.
///
/// Given Lamé parameters $\mu$ and $\lambda$, the strain energy density is
/// $$
/// \psi(\vec F) =
///     \mu \vec \epsilon : \vec \epsilon
///   + \frac{\lambda}{2} \operatorname{tr}^2(\vec \epsilon),
/// $$
/// where
/// $$
/// \vec \epsilon(\vec F) = \frac{(\vec F + \vec F^T)}{2} - \vec I
/// $$
/// is the infinitesimal strain tensor. The associated stress tensor is
/// $$
/// \vec P(\vec F) = 2 \mu \vec \epsilon + \lambda \operatorname{tr}(\vec \epsilon) \vec I.
/// $$
/// Finally, the contraction operator associated with the stress tensor is
/// $$
/// \mathcal{C}_{\vec P}(\vec F, \vec a, \vec b) =
///     \mu \left[ (\vec a \cdot \vec b) \vec I + \vec b \vec a^T \right]
///     + \lambda \vec a \vec b^T.
/// $$
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearElasticMaterial;

#[allow(non_snake_case)]
fn infinitesimal_strain_tensor<T, D>(deformation_gradient: &OMatrix<T, D, D>) -> OMatrix<T, D, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    let F = deformation_gradient;
    F.symmetric_part() - OMatrix::<T, D, D>::identity()
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> HyperelasticMaterial<T, D> for LinearElasticMaterial
where
    T: RealField,
    D: DimName,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    type Parameters = LameParameters<T>;

    fn compute_energy_density(&self, deformation_gradient: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> T {
        let &LameParameters { mu, lambda } = parameters;
        let eps = infinitesimal_strain_tensor(deformation_gradient);
        mu * eps.dot(&eps) + 0.5 * lambda * eps.trace().powi(2)
    }

    fn compute_stress_tensor(
        &self,
        deformation_gradient: &OMatrix<T, D, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let &LameParameters { mu, lambda } = parameters;
        let eps = infinitesimal_strain_tensor(deformation_gradient);
        let eps_tr = eps.trace();
        eps * 2.0 * mu + OMatrix::from_diagonal(&OVector::<T, D>::repeat(lambda * eps_tr))
    }

    fn compute_stress_contraction(
        &self,
        _deformation_gradient: &OMatrix<T, D, D>,
        a: &OVector<T, D>,
        b: &OVector<T, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let &LameParameters { mu, lambda } = parameters;
        let I = OMatrix::<T, D, D>::identity();
        (I * a.dot(b) + b * a.transpose()) * mu + a * b.transpose() * lambda

        // TODO: Implement multi-contraction for efficiency? There doesn't seem to be any
        // computations that can be re-used across different vectors though, so it's not clear
        // if there are any efficiency gains to be found
    }
}

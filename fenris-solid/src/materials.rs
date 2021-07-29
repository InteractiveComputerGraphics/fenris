use crate::HyperelasticMaterial;
use fenris::allocators::SmallDimAllocator;
use fenris::nalgebra::{min, DMatrixSliceMut, DVectorSlice, DefaultAllocator, DimName, OMatrix, OVector, RealField};
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

/// The linear elastic material model.
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

/// The Neo-Hookean material model.
///
/// The strain energy density is given by
/// $$
/// \psi(\vec F) = \frac{\mu}{2}(I_C - 3) - \mu \log J + \frac{\lambda}{2}(\log J)^2,
/// $$
/// where $J = \det \vec F$ and $I_C = \tr{\vec C} = \tr{\vec F^T \vec F}$ is the first right Cauchy-Green invariant.
///
///
///
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeoHookeanMaterial;

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> HyperelasticMaterial<T, D> for NeoHookeanMaterial
where
    T: RealField,
    D: DimName,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    type Parameters = LameParameters<T>;

    fn compute_energy_density(&self, deformation_gradient: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> T {
        let _ = (deformation_gradient, parameters);
        // let F = deformation_gradient;
        // let C = F.transpose() * F;
        todo!()
    }

    fn compute_stress_tensor(
        &self,
        deformation_gradient: &OMatrix<T, D, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let _ = (deformation_gradient, parameters);
        todo!()
    }

    fn compute_stress_contraction(
        &self,
        deformation_gradient: &OMatrix<T, D, D>,
        a: &OVector<T, D>,
        b: &OVector<T, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let _ = (deformation_gradient, a, b, parameters);
        todo!()
    }
}

/// The Saint Venant-Kirchhoff material model.
///
/// This material model is characterized by the strain energy density
/// $$
/// \psi(\vec F) = \mu \vec E : \vec E + \frac{\lambda}{2} \operatorname{tr}^2(\vec E)
/// $$
/// where $\mu$ and $\lambda$ are Lamé parameters and $\vec E = \frac{1}{2} \left( \vec F^T \vec F - \vec I \right)$
/// is the Green strain tensor. The stress tensor is
/// $$
/// \vec P(\vec F) = \vec F (2 \mu \vec E + \lambda \tr{E} \vec I)
/// $$
/// and the contraction operator is
/// $$
/// \mathcal{C}_{\vec P}(\vec F, \vec a, \vec b) =
///     \left[ 2 \mu \vec a^T \vec E \vec b + \lambda \tr{\vec E} (\vec a \cdot \vec b) \right] \vec I
///     + \mu (\vec F \vec b) (\vec F \vec a)^T
///     + \lambda (\vec F \vec a) (\vec F \vec b)^T
///     + \mu (\vec a \cdot \vec b) \vec F \vec F^T.
/// $$
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StVKMaterial;

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
fn green_strain_tensor<T, D>(deformation_gradient: &OMatrix<T, D, D>) -> OMatrix<T, D, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    let I = &OMatrix::<T, D, D>::identity();
    let F = deformation_gradient;
    (F.transpose() * F - I) * 0.5
}

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> HyperelasticMaterial<T, D> for StVKMaterial
where
    T: RealField,
    D: DimName,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    type Parameters = LameParameters<T>;

    fn compute_energy_density(&self, deformation_gradient: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> T {
        let &LameParameters { mu, lambda } = parameters;
        let E = green_strain_tensor(deformation_gradient);
        mu * E.dot(&E) + 0.5 * lambda * E.trace().powi(2)
    }

    fn compute_stress_tensor(
        &self,
        deformation_gradient: &OMatrix<T, D, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let &LameParameters { mu, lambda } = parameters;
        let F = deformation_gradient;
        let E = green_strain_tensor(deformation_gradient);
        F * &E * 2.0 * mu + F * lambda * E.trace()
    }

    fn compute_stress_contraction(
        &self,
        deformation_gradient: &OMatrix<T, D, D>,
        a: &OVector<T, D>,
        b: &OVector<T, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let &LameParameters { mu, lambda } = parameters;
        let I = &OMatrix::<T, D, D>::identity();
        let F = deformation_gradient;
        let E = green_strain_tensor(&F);
        let a_dot_b = a.dot(b);

        let ref Fa = F * a;
        let ref Fb = F * b;
        let Eb = &E * b;

        I * (2.0 * mu * a.dot(&Eb) + lambda * E.trace() * a_dot_b)
            + Fb * Fa.transpose() * mu
            + Fa * Fb.transpose() * lambda
            + F * F.transpose() * mu * a_dot_b
    }

    fn accumulate_stress_contractions_into(
        &self,
        mut output: DMatrixSliceMut<T>,
        alpha: T,
        deformation_gradient: &OMatrix<T, D, D>,
        a: DVectorSlice<T>,
        b: DVectorSlice<T>,
        parameters: &Self::Parameters,
    ) {
        // Note: this is an adaption of the default implementation that exploits structure of the StVK material model
        // by computing some shared quantities once for all contraction vectors
        let d = D::dim();
        assert!(a.len() % d == 0, "Dimension of a must be divisible by d (GeometryDim)");
        assert!(b.len() % d == 0, "Dimension of b must be divisible by d (GeometryDim)");
        let M = a.len() / d;
        let N = b.len() / d;
        assert_eq!(
            output.nrows(),
            d * M,
            "Number of rows in output matrix is not consistent with a"
        );
        assert_eq!(
            output.ncols(),
            d * N,
            "Number of columns in output matrix is not consistent with b"
        );
        let d_times_d = (D::name(), D::name());

        let &LameParameters { mu, lambda } = parameters;
        let eye = &OMatrix::<T, D, D>::identity();
        let F = deformation_gradient;
        let E = green_strain_tensor(&F);
        let E_trace = E.trace();
        let ref FFt = F * F.transpose();

        // Note: We fill the matrix (block) column-by-column since the matrix is stored in
        // column-major format
        for J in 0..N {
            // Compute quantities that only depend on J to prevent computing these over and over again
            let b_J = b.rows_generic(d * J, D::name()).clone_owned();
            let ref Fb = F * &b_J;
            let Eb = &E * &b_J;

            for I in 0..min(J + 1, M) {
                let a_I = a.rows_generic(d * I, D::name()).clone_owned();

                let a_dot_b = a_I.dot(&b_J);
                let ref Fa = F * &a_I;

                let contraction = eye * ((2.0 * mu * a_I.dot(&Eb) + lambda * E_trace * a_dot_b) * alpha)
                    + Fb * Fa.transpose() * (mu * alpha)
                    + Fa * Fb.transpose() * (lambda * alpha)
                    + FFt * (mu * a_dot_b * alpha);
                let mut c_IJ = output.generic_slice_mut((d * I, d * J), d_times_d);
                c_IJ += contraction;
            }
        }
    }
}

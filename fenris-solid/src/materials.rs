use crate::{compute_batch_contraction, HyperelasticMaterial};
use fenris::allocators::SmallDimAllocator;
use fenris::nalgebra::{DMatrixSliceMut, DVectorSlice, DefaultAllocator, DimName, OMatrix, OVector, RealField};
use fenris::SmallDim;
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
/// Note that the energy is only well-defined when $J > 0$. We explicitly return infinity in this case, so that
/// it may be used e.g. as a barrier in optimization.
///
/// The Piola-Kirchhoff stress tensor is given by
/// $$
///  \vec P = (-\mu + \lambda \log J) \vec F^{-T} + \mu \vec F.
/// $$
/// With $\alpha = -\mu + \lambda \log J$, the stress contraction for arbitrary vectors
/// $\vec a, \vec b \in \mathbb{R}^d$ is given by
/// <div>$$
/// \begin{align*}
///   \mathcal{C}_{\vec P}(\vec F, \vec a, \vec b)
///   &= \lambda (\vec F^{-T} \vec a) \otimes (\vec F^{-T} \vec b)
///    - \alpha (\vec F^{-T} \vec b) \otimes (\vec F^{-T} \vec a)
///    + \mu (\vec a \cdot \vec b) \vec I.
/// \end{align*}
/// $$</div>
///
/// # Derivation
///
/// For posterity, we sketch out the derivation. We assume throughout that $J > 0$, and thus $\vec F$ is invertible.
///
/// ## Stress tensor (first derivative)
///
/// We have that
///
/// $$
///  \pd{J}{\vec F} = J \vec F^{-T}
///  \qquad \qquad
///  \pd{I_C}{\vec F} = 2 \vec F.
/// $$
/// and
/// $$
///  \pd{\psi}{J} = J^{-1} (-\mu + \lambda \log J)
///  \qquad \qquad
///  \pd{\psi}{I_C} = \frac{\mu}{2}.
/// $$
///
/// The Piola-Kirchhoff stress tensor becomes
/// $$
///  \vec P = \pd{\psi}{\vec F} = \pd{\psi}{J} \pd{J}{\vec F} + \pd{\psi}{I_C} \pd{I_C}{\vec F}
///         = (-\mu + \lambda \log J) \vec F^{-T} + \mu \vec F
/// $$
///
/// ## Stress contraction (second derivative)
///
/// We define $\alpha := -\mu + \lambda \log J$, and write out the derivative of $\vec P$ as
/// <div>$$
///  \pd{P_{ij}}{F_{kl}} =
///     \pd{\alpha}{F_{kl}} (\vec F^{-T})_{ij}
///     + \alpha \pd{(\vec F^{-T})_{ij}}{F_{kl}}
///     + \mu \pd{F_{ij}}{F_{kl}}
///  \\
///  = A + B + C.
/// $$</div>
///
/// Next, we'll find expressions for $A$, $B$ and $C$ in turn. We have
/// <div>$$
///  \pd{\alpha}{F_{kl}} = \lambda (\vec F^{-T})_{kl},
/// $$</div>
/// and so
/// <div>$$
///  A = \lambda \, (\vec F^{-T})_{ij} (\vec F^{-T})_{kl}.
/// $$</div>
///
/// Using the relation (see, e.g., The Matrix Cookbook)
/// <div>$$
///   \pd{(\vec X^{-1})_{ij}}{X_{kl}} = - (\vec X^{-1})_{ik} (\vec X)^{-1}_{lj},
/// $$</div>
/// we have that
/// <div>$$
///   B = - \alpha \; (\vec F^{-T})_{il} (\vec F^{-T})_{kj}.
/// $$</div>
///
/// Finally,
/// <div>$$
///   C = \mu \delta_{ik} \delta_{jl}.
/// $$</div>
///
/// Our final expression for the second derivative thus becomes
/// <div>$$
/// \pd{P_{ij}}{F_{kl}} =
///     \lambda \, (\vec F^{-T})_{ij} (\vec F^{-T})_{kl}
///     - \alpha \; (\vec F^{-T})_{il} (\vec F^{-T})_{kj}
///     + \mu \delta_{ik} \delta_{jl}.
/// $$</div>
///
/// The stress contraction becomes
/// <div>$$
/// \begin{align*}
///   \mathcal{C}_{\vec P}(\vec F, \vec a, \vec b)
///   &= a_k \pd{P_{ik}}{F_{jm}} (\vec F) \, b_m \; \vec e_i \otimes \vec e_j \\
///   &= \lambda (\vec F^{-T} \vec a) \otimes (\vec F^{-T} \vec b)
///    - \alpha (\vec F^{-T} \vec b) \otimes (\vec F^{-T} \vec a)
///    + \mu (\vec a \cdot \vec b) \vec I.
/// \end{align*}
/// $$</div>
///
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeoHookeanMaterial;

#[allow(non_snake_case)]
#[replace_float_literals(T::from_f64(literal).expect("literal must fit in T"))]
impl<T, D> HyperelasticMaterial<T, D> for NeoHookeanMaterial
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    type Parameters = LameParameters<T>;

    fn compute_energy_density(&self, deformation_gradient: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> T {
        let LameParameters { mu, lambda } = parameters.clone();
        let F = deformation_gradient;
        let J = F.determinant();

        if J <= T::zero() {
            T::from_f64(f64::INFINITY).expect("T must be able to represent infinity")
        } else {
            let C = F.transpose() * F;
            let I_C = C.trace();
            let logJ = J.ln();
            // Note: 2D/3D need different constants to ensure rest state has zero energy
            let d = T::from_usize(D::dim()).unwrap();
            mu / 2.0 * (I_C - d) - mu * logJ + (lambda / 2.0) * (logJ.powi(2))
        }
    }

    fn compute_stress_tensor(
        &self,
        deformation_gradient: &OMatrix<T, D, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let LameParameters { mu, lambda } = parameters.clone();
        let F = deformation_gradient;
        let J = F.determinant();

        if J <= T::zero() {
            todo!("How to address non-positive J? (J = {})", J);
        } else {
            let logJ = J.ln();
            let F_inv = F
                .clone()
                .try_inverse()
                .expect("F is guaranteed to be invertible here");
            let F_inv_T = F_inv.transpose();
            F_inv_T * (-mu + lambda * logJ) + F * mu
        }
    }

    fn compute_stress_contraction(
        &self,
        deformation_gradient: &OMatrix<T, D, D>,
        a: &OVector<T, D>,
        b: &OVector<T, D>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, D, D> {
        let LameParameters { mu, lambda } = parameters.clone();
        let F = deformation_gradient;
        let J = F.determinant();

        if J <= T::zero() {
            todo!("How to address non-positive J? (J = {})", J);
        } else {
            let logJ = J.ln();
            let F_inv = F
                .clone()
                .try_inverse()
                .expect("F is guaranteed to be invertible here");
            let F_inv_T = F_inv.transpose();
            let ref F_inv_T_a = &F_inv_T * a;
            let ref F_inv_T_b = &F_inv_T * b;
            let ref I = OMatrix::<_, D, D>::identity();
            let alpha = -mu + lambda * logJ;
            (F_inv_T_a) * (F_inv_T_b.transpose() * lambda) - F_inv_T_b * (F_inv_T_a.transpose() * alpha)
                + I * (mu * a.dot(&b))
        }
    }

    fn accumulate_stress_contractions_into(
        &self,
        output: DMatrixSliceMut<T>,
        alpha: T,
        deformation_gradient: &OMatrix<T, D, D>,
        a: DVectorSlice<T>,
        b: DVectorSlice<T>,
        parameters: &Self::Parameters,
    ) {
        let LameParameters { mu, lambda } = parameters.clone();
        let F = deformation_gradient;
        let J = F.determinant();

        if J <= T::zero() {
            todo!("How to address non-positive J? (J = {})", J);
        } else {
            // Precompute all the quantities that are independent of a and b
            let logJ = J.ln();
            let F_inv = F
                .clone()
                .try_inverse()
                .expect("F is guaranteed to be invertible here");
            let F_inv_T = F_inv.transpose();
            let ref I = OMatrix::<_, D, D>::identity();
            // Note: This alpha is from the formula, not from the alpha contraction parameter!
            // TODO: Use different formula in derivation?
            let alpha_nh = -mu + lambda * logJ;

            compute_batch_contraction(output, alpha, a, b, |a, b| {
                let ref F_inv_T_a = &F_inv_T * a;
                let ref F_inv_T_b = &F_inv_T * b;
                (F_inv_T_a) * (F_inv_T_b.transpose() * lambda) - F_inv_T_b * (F_inv_T_a.transpose() * alpha_nh)
                    + I * (mu * a.dot(&b))
            })
        }
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
        output: DMatrixSliceMut<T>,
        alpha: T,
        deformation_gradient: &OMatrix<T, D, D>,
        a: DVectorSlice<T>,
        b: DVectorSlice<T>,
        parameters: &Self::Parameters,
    ) {
        let &LameParameters { mu, lambda } = parameters;
        let eye = &OMatrix::<T, D, D>::identity();
        let F = deformation_gradient;
        let E = green_strain_tensor(&F);
        let E_trace = E.trace();
        let ref FFt = F * F.transpose();

        compute_batch_contraction(output, alpha, a, b, |a_I, b_J| {
            let a_dot_b = a_I.dot(&b_J);
            let ref Fa = F * a_I;
            let ref Fb = F * b_J;
            let Eb = &E * b_J;

            let contraction = eye * (2.0 * mu * a_I.dot(&Eb) + lambda * E_trace * a_dot_b)
                + Fb * (Fa.transpose() * mu)
                + Fa * (Fb.transpose() * lambda)
                + FFt * (mu * a_dot_b);
            contraction
        })
    }
}

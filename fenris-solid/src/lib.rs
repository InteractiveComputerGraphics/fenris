//! Solid mechanics functionality for `fenris`.
use fenris::allocators::DimAllocator;
use fenris::assembly::operators::{EllipticContraction, EllipticEnergy, EllipticOperator, Operator};
use fenris::nalgebra::{
    DMatrixViewMut, DVectorView, DefaultAllocator, DimName, OMatrix, OVector, RealField, U1, U2, U3,
};
use fenris::{Real, SmallDim, Symmetry};
use std::cmp::min;

pub mod materials;

mod logdet;
pub use logdet::log_det_F;

mod gravity_source;
pub use gravity_source::GravitySource;

/// Compute the deformation gradient $\vec F$ given the displacement gradient $\nabla \vec u$.
#[allow(non_snake_case)]
pub fn deformation_gradient<T, D>(u_grad: &OMatrix<T, D, D>) -> OMatrix<T, D, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: DimAllocator<T, D>,
{
    let I = OMatrix::<T, D, D>::identity();
    let F = I + u_grad.transpose();
    F
}

/// Compute the displacement gradient $\nabla \vec u$ given the deformation gradient $\vec F$.
#[allow(non_snake_case)]
pub fn u_grad_from_F<T, D>(F: &OMatrix<T, D, D>) -> OMatrix<T, D, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: DimAllocator<T, D>,
{
    let I = OMatrix::<T, D, D>::identity();
    (F - I).transpose()
}

/// A hyperelastic material defined by its energy density $\psi(\vec F)$.
///
/// Although hyperelastic materials in literature are defined in terms of the deformation gradient
/// $\vec F$, computing $\vec F = \vec I + \nabla \vec u^T$ incurs a significant loss in accuracy
/// if $\nabla \vec u$ is very small (such as for stiff materials). Therefore, in addition
/// to the required methods that work with $\vec F$, such as
/// [`compute_energy_density`](Self::compute_energy_density), there are additional methods with the
/// `_du` suffix which refer to a variant of the method where the input is the displacement
/// gradient $\nabla \vec u$ instead of $\vec F$. These methods have default impls which rely
/// on computing $\vec F$, but can be overridden by individual materials for more accurate results.
///
/// See the [libCEED documentation](https://libceed.org/en/latest/examples/solids) for a discussion
/// of the numerical problems associated with forming $\vec F$ and derived quantities such as
/// $\log \det \vec F$.
///
/// In the future, the trait may be changed to always work with $\nabla \vec u$ instead of $\vec F$.
pub trait HyperelasticMaterial<T, GeometryDim>
where
    T: Real,
    GeometryDim: DimName,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    type Parameters: Clone + Default + 'static;

    /// Compute the energy density $\psi = \psi(\vec F)$ associated with the material.
    fn compute_energy_density(
        &self,
        deformation_gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> T;

    /// Compute the energy density $\psi = \psi(\vec F)$ associated with the material.
    ///
    /// Takes as input the displacement gradient $\nabla \vec u$ instead of the deformation
    /// gradient $\vec F$. See [the trait docs](Self) for more information.
    fn compute_energy_density_du(
        &self,
        u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> T {
        self.compute_energy_density(&deformation_gradient(u_grad), parameters)
    }

    /// Compute the First Piola-Kirchhoff stress tensor $\vec P = \vec P(\vec F)$.
    fn compute_stress_tensor(
        &self,
        deformation_gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, GeometryDim, GeometryDim>;

    /// Compute the First Piola-Kirchhoff stress tensor $\vec P = \vec P(\vec F)$.
    ///
    /// Takes as input the displacement gradient $\nabla \vec u$ instead of the deformation
    /// gradient $\vec F$. See [the trait docs](Self) for more information.
    #[allow(non_snake_case)]
    fn compute_stress_tensor_du(
        &self,
        u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, GeometryDim, GeometryDim> {
        self.compute_stress_tensor(&deformation_gradient(u_grad), parameters)
    }

    /// Compute the stress contraction operator $\\mathcal{C}\_{\vec P}(\vec F, \vec a, \vec b)$ with the given
    /// material parameters.
    ///
    /// The contraction operator is defined by
    /// $$
    /// \\mathcal{C}\_{\vec P} (\vec F, \vec a, \vec b)
    ///     := a_k \pd{P_{ik}}{F_{jm}} (\vec F) \\, b_m \enspace \vec e_i \otimes \vec e_j.
    /// $$
    fn compute_stress_contraction(
        &self,
        deformation_gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        a: &OVector<T, GeometryDim>,
        b: &OVector<T, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, GeometryDim, GeometryDim>;

    /// Compute the stress contraction operator $\\mathcal{C}\_{\vec P}(\vec F, \vec a, \vec b)$ with the given
    /// material parameters.
    ///
    /// Takes as input the displacement gradient $\nabla \vec u$ instead of the deformation
    /// gradient $\vec F$. See [the trait docs](Self) for more information.
    fn compute_stress_contraction_du(
        &self,
        u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
        a: &OVector<T, GeometryDim>,
        b: &OVector<T, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, GeometryDim, GeometryDim> {
        self.compute_stress_contraction(&deformation_gradient(u_grad), a, b, parameters)
    }

    /// Compute the contraction for a number of vectors at the same time, with the given
    /// parameters.
    ///
    /// This method is analogous to
    /// [`EllipticContraction::accumulate_contractions_into`](`::fenris::assembly::operators::EllipticContraction::accumulate_contractions_into`),
    /// but specialized for hyperelastic materials.
    ///
    /// The vectors $\vec a \in \mathbb{R}^{dM}$ and $\vec b \in \mathbb{R}^{dN}$ are stacked vectors
    /// $$
    /// \begin{align*}
    /// \vec a := \begin{pmatrix}
    /// \vec a_1 \newline
    /// \vdots \newline
    /// \vec a_M
    /// \end{pmatrix},
    /// \qquad
    /// \vec b := \begin{pmatrix}
    /// \vec b_1 \newline
    /// \vdots \newline
    /// \vec b_N
    /// \end{pmatrix}
    /// \end{align*}
    /// $$
    ///
    /// and $\vec a_I, \vec b_J \in \mathbb{R}^d$ for $I = 1, \dots, M, J = 1, \dots, N$.
    /// Let $C \in \mathbb{R}^{sM \times sN}$ denote the output matrix, which is a block
    /// matrix of the form
    /// $$
    /// \begin{align*}
    /// C := \begin{pmatrix}
    /// C_{11} & \dots  & C_{1N} \newline
    /// \vdots & \ddots & \vdots \newline
    /// C_{M1} & \dots  & C_{MN}
    /// \end{pmatrix}
    /// \end{align*},
    /// $$
    /// where each block $C_{IJ}$ is an $d \times d$ matrix. This method **accumulates**
    /// the block-wise **scaled** contractions in the following manner:
    /// $$
    /// C_{IJ} \gets C_{IJ} + \alpha \mathcal{C}_{\vec P}(\vec F, \vec a_I, \vec b_J)
    /// $$
    ///
    /// The default implementation repeatedly calls
    /// [`compute_stress_contraction`](`Self::compute_stress_contraction`). However, since
    /// $\vec F$ is constant for all vectors $\vec a_I, \vec b_J$, it is often possible to
    /// share computations and therefore significantly improve performance.
    ///
    /// The contraction operator associated with a hyperelastic material is *always* symmetric,
    /// in the sense that
    ///
    /// $$
    /// \\mathcal{C}_{\vec P} (\vec F, \vec a, \vec b)
    /// = \mathcal{C}\_{\vec P}(\vec F, \vec b, \vec a)^T,
    /// $$
    ///
    /// for all $\vec a, \vec b$. Therefore an implementation only needs to fill the upper
    /// triangle of the output matrix.
    /// *Consumers of this method **must** take this into account!*
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != b.len()` or `a.len()` is not divisible by $d$ (`GeometryDim`).
    ///
    /// Panics if `output.nrows() != d * M` or `output.ncols() != d * N`.
    ///
    /// TODO: Test the default impl
    #[allow(non_snake_case)]
    fn accumulate_stress_contractions_into(
        &self,
        output: DMatrixViewMut<T>,
        alpha: T,
        deformation_gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        a: DVectorView<T>,
        b: DVectorView<T>,
        parameters: &Self::Parameters,
    ) {
        compute_batch_contraction(output, alpha, a, b, |a_I, b_J| {
            self.compute_stress_contraction(deformation_gradient, a_I, b_J, parameters)
        })
    }

    /// Compute the contraction for a number of vectors at the same time, with the given
    /// parameters.
    ///
    /// Takes as input the displacement gradient $\nabla \vec u$ instead of the deformation
    /// gradient $\vec F$. See [the trait docs](Self) for more information.
    #[allow(non_snake_case)]
    fn accumulate_stress_contractions_du_into(
        &self,
        output: DMatrixViewMut<T>,
        alpha: T,
        u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
        a: DVectorView<T>,
        b: DVectorView<T>,
        parameters: &Self::Parameters,
    ) {
        compute_batch_contraction(output, alpha, a, b, |a_I, b_J| {
            self.compute_stress_contraction_du(&u_grad, a_I, b_J, parameters)
        })
    }
}

// TODO: Remove this or develop it further. The idea is to be able to
// easily implement materials only in terms of du/dX and have some mechanism for automatically
// producing the whole HyperelasticMaterial trait. An alternative, simpler direction,
// would be to simply have the HyperelasticMaterial trait only work directly with u_grad.

// pub trait HyperelasticMaterialDu<T, GeometryDim>
// where
//     T: Real,
//     GeometryDim: DimName,
//     DefaultAllocator: DimAllocator<T, GeometryDim>,
// {
//     type Parameters: Clone + Default + 'static;
//
//     /// TODO: Docs
//     fn compute_energy_density_du(
//         &self,
//         u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
//         parameters: &Self::Parameters,
//     ) -> T;
//
//     /// TODO: Docs
//     fn compute_stress_tensor_du(
//         &self,
//         u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
//         parameters: &Self::Parameters,
//     ) -> OMatrix<T, GeometryDim, GeometryDim>;
//
//     /// TODO: Docs
//     fn compute_stress_contraction_du(
//         &self,
//         u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
//         a: &OVector<T, GeometryDim>,
//         b: &OVector<T, GeometryDim>,
//         parameters: &Self::Parameters,
//     ) -> OMatrix<T, GeometryDim, GeometryDim>;
//
//     /// TODO: Docs
//     #[allow(non_snake_case)]
//     fn accumulate_stress_contractions_du_into(
//         &self,
//         output: DMatrixViewMut<T>,
//         alpha: T,
//         u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
//         a: DVectorView<T>,
//         b: DVectorView<T>,
//         parameters: &Self::Parameters,
//     ) {
//         compute_batch_contraction(output, alpha, a, b, |a_I, b_J| {
//             self.compute_stress_contraction_du(&u_grad, a_I, b_J, parameters)
//         })
//     }
// }

// #[macro_export]
// macro_rules! impl_hyperelastic_from_du_impl {
//     ($material:ty) => {
//         impl<T, D> HyperelasticMaterial<T, D> for $material
//         where
//             T: Real,
//             D: DimName,
//             $material: HyperelasticMaterialDu<T, D>,
//             DefaultAllocator: DimAllocator<T, D>,
//         {
//             type Parameters = <$material as HyperelasticMaterialDu<T, D>>::Parameters;
//
//             fn compute_energy_density(&self, deformation_gradient: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> T {
//                 let u_grad = crate::u_grad_from_F(deformation_gradient);
//                 <Self as HyperelasticMaterialDu<T, D>>::compute_energy_density_du(self, &u_grad, parameters)
//             }
//
//             fn compute_energy_density_du(&self, u_grad: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> T {
//                 <Self as HyperelasticMaterialDu<T, D>>::compute_energy_density_du(self, u_grad, parameters)
//             }
//
//             fn compute_stress_tensor(&self, deformation_gradient: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> OMatrix<T, D, D> {
//                 let u_grad = crate::u_grad_from_F(deformation_gradient);
//                 <Self as HyperelasticMaterialDu<T, D>>::compute_stress_tensor_du(self, &u_grad, parameters)
//             }
//
//             fn compute_stress_tensor_du(&self, u_grad: &OMatrix<T, D, D>, parameters: &Self::Parameters) -> OMatrix<T, D, D> {
//                 <Self as HyperelasticMaterialDu<T, D>>::compute_stress_tensor_du(self, u_grad, parameters)
//             }
//
//             fn compute_stress_contraction(&self, deformation_gradient: &OMatrix<T, D, D>, a: &OVector<T, D>, b: &OVector<T, D>, parameters: &Self::Parameters) -> OMatrix<T, D, D> {
//                 let u_grad = crate::u_grad_from_F(deformation_gradient);
//                 <Self as HyperelasticMaterialDu<T, D>>::compute_stress_contraction_du(self, &u_grad, a, b, parameters)
//             }
//
//             fn compute_stress_contraction_du(&self, u_grad: &OMatrix<T, D, D>, a: &OVector<T, D>, b: &OVector<T, D>, parameters: &Self::Parameters) -> OMatrix<T, D, D> {
//                 <Self as HyperelasticMaterialDu<T, D>>::compute_stress_contraction_du(self, u_grad, a, b, parameters)
//             }
//
//             fn accumulate_stress_contractions_into(&self, output: DMatrixViewMut<T>, alpha: T, deformation_gradient: &OMatrix<T, D, D>, a: DVectorView<T>, b: DVectorView<T>, parameters: &Self::Parameters) {
//                 let u_grad = crate::u_grad_from_F(deformation_gradient);
//                 <Self as HyperelasticMaterialDu<T, D>>::accumulate_stress_contractions_du_into(self, output, alpha, &u_grad, a, b, parameters)
//             }
//
//             fn accumulate_stress_contractions_du_into(&self, output: DMatrixViewMut<T>, alpha: T, u_grad: &OMatrix<T, D, D>, a: DVectorView<T>, b: DVectorView<T>, parameters: &Self::Parameters) {
//                 <Self as HyperelasticMaterialDu<T, D>>::accumulate_stress_contractions_du_into(self, output, alpha, u_grad, a, b, parameters)
//             }
//         }
//     }
// }

/// Helper function to ease implementation of [`HyperelasticMaterial::accumulate_stress_contractions_into`].
///
/// Often implementations of this method will tend to look very similar. This method is provided as a means
/// to eliminate most of the boilerplate. See implementations of various materials for how to use it.
#[allow(non_snake_case)]
#[inline]
pub fn compute_batch_contraction<T, GeometryDim>(
    mut output: DMatrixViewMut<T>,
    alpha: T,
    a: DVectorView<T>,
    b: DVectorView<T>,
    mut contraction: impl FnMut(&OVector<T, GeometryDim>, &OVector<T, GeometryDim>) -> OMatrix<T, GeometryDim, GeometryDim>,
) where
    T: Real,
    GeometryDim: DimName,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    // Note: This implementation is just an adaptation of the default impl
    // of EllipticContraction::accumulate_contractions_into
    let d = GeometryDim::dim();
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
    let d_times_d = (GeometryDim::name(), GeometryDim::name());

    // Note: We fill the matrix (block) column-by-column since the matrix is stored in
    // column-major format
    for J in 0..N {
        // Contraction is always symmetric
        for I in 0..min(J + 1, M) {
            let a_I = a.rows_generic(d * I, GeometryDim::name()).clone_owned();
            let b_J = b.rows_generic(d * J, GeometryDim::name()).clone_owned();
            let mut c_IJ = output.generic_view_mut((d * I, d * J), d_times_d);
            let contraction = contraction(&a_I, &b_J);
            // c_IJ += contraction * alpha
            c_IJ.zip_apply(&contraction, |c, y| *c += alpha * y);
        }
    }
}

/// A wrapper that turns any hyper elastic material into an elliptic operator for use
/// with `fenris` assembly operations.
///
/// The wrapper assumes a **displacement-based** formulation, i.e. that the solution field is the displacement
/// $\vec u (\vec X) = \vec x(\vec X) - \vec X$. In other words, the nodal weights should correspond to displacements,
/// not deformed positions. Alternatively, you may transform deformed positions to displacements as a
/// preprocessing step before handing off the resulting displacements to assembly functionality relying on this
/// operator wrapper.
///
/// This implies the following relations:
///
/// $$
/// \begin{aligned}
/// \vec F &= \vec I + (\nabla \vec u)^T, \\\\
/// \vec P (\vec F) &= g^T (\nabla \vec u), \\\\
/// \mathcal{C}_{\vec P}(\vec F, \vec a, \vec b) &= \mathcal{C}_g(\nabla \vec u, \vec a, \vec b). \\\\
/// \end{aligned}
/// $$
pub struct MaterialEllipticOperator<'a, Material>(&'a Material);

impl<'a, Material> MaterialEllipticOperator<'a, Material> {
    pub fn new(material: &'a Material) -> Self {
        Self(material)
    }
}

impl<'a, T, GeometryDim, Material> Operator<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: Real,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    type SolutionDim = GeometryDim;
    type Parameters = Material::Parameters;
}

impl<'a, T, GeometryDim, Material> EllipticEnergy<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: Real,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    fn compute_energy(&self, u_grad: &OMatrix<T, GeometryDim, GeometryDim>, parameters: &Self::Parameters) -> T {
        self.0.compute_energy_density_du(u_grad, parameters)
    }
}

impl<'a, T, GeometryDim, Material> EllipticOperator<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: Real,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    fn compute_elliptic_operator(
        &self,
        u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, GeometryDim, Self::SolutionDim> {
        // The material operator g is related to the stress tensor P by g = P^T,
        // so it's more convenient to implement the transpose opereation g^T = P,
        // which is anyway what is called by the assembler
        self.compute_elliptic_operator_transpose(u_grad, parameters)
            .transpose()
    }

    fn compute_elliptic_operator_transpose(
        &self,
        u_grad: &OMatrix<T, GeometryDim, Self::SolutionDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, Self::SolutionDim, GeometryDim> {
        // We avoid forming the deformation gradient here so that we can avoid
        // the loss of accuracy implied by forming `F = I + grad u` for small `grad u`
        self.0.compute_stress_tensor_du(u_grad, parameters)
    }
}

impl<'a, T, GeometryDim, Material> EllipticContraction<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: Real,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: DimAllocator<T, GeometryDim>,
{
    fn contract(
        &self,
        u_grad: &OMatrix<T, GeometryDim, GeometryDim>,
        a: &OVector<T, GeometryDim>,
        b: &OVector<T, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, Self::SolutionDim, Self::SolutionDim> {
        self.0
            .compute_stress_contraction_du(u_grad, a, b, parameters)
    }

    fn symmetry(&self) -> Symmetry {
        Symmetry::Symmetric
    }

    #[allow(non_snake_case)]
    fn accumulate_contractions_into(
        &self,
        output: DMatrixViewMut<T>,
        alpha: T,
        u_grad: &OMatrix<T, GeometryDim, Self::SolutionDim>,
        a: DVectorView<T>,
        b: DVectorView<T>,
        parameters: &Self::Parameters,
    ) {
        self.0
            .accumulate_stress_contractions_du_into(output, alpha, u_grad, a, b, parameters)
    }
}

mod internal {
    use fenris::nalgebra::{U1, U2, U3};

    pub trait Sealed {}

    impl Sealed for U1 {}
    impl Sealed for U2 {}
    impl Sealed for U3 {}
}

/// A fixed-size dimension corresponding to physical space.
///
/// Physical dimensions are comprised of the dimensions $1$, $2$ and $3$. The primary utility
/// of this trait is to support writing generic code that needs to evaluate functions
/// whose implementation differs from dimension to dimension, and that might not have an easily
/// accessible n-dimensional variant.
pub trait PhysicalDim: internal::Sealed + SmallDim {}

impl PhysicalDim for U1 {}
impl PhysicalDim for U2 {}
impl PhysicalDim for U3 {}

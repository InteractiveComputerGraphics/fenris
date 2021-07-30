//! Solid mechanics functionality for `fenris`.
use fenris::allocators::SmallDimAllocator;
use fenris::assembly::operators::{EllipticContraction, EllipticEnergy, EllipticOperator, Operator};
use fenris::nalgebra::{DMatrixSliceMut, DVectorSlice, DefaultAllocator, DimName, OMatrix, OVector, RealField};
use fenris::{SmallDim, Symmetry};
use std::cmp::min;

pub mod materials;

pub trait HyperelasticMaterial<T, GeometryDim>
where
    T: RealField,
    GeometryDim: DimName,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    type Parameters: Clone + Default + 'static;

    /// Compute the energy density $\psi = \psi(\vec F)$ associated with the material.
    fn compute_energy_density(
        &self,
        deformation_gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> T;

    /// Compute the First Piola-Kirchhoff stress tensor $\vec P = \vec P(\vec F)$.
    fn compute_stress_tensor(
        &self,
        deformation_gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, GeometryDim, GeometryDim>;

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
        mut output: DMatrixSliceMut<T>,
        alpha: T,
        deformation_gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        a: DVectorSlice<T>,
        b: DVectorSlice<T>,
        parameters: &Self::Parameters,
    ) {
        // Note: This implementation is just an adaption of the default impl
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
                let mut c_IJ = output.generic_slice_mut((d * I, d * J), d_times_d);
                let contraction = self.compute_stress_contraction(deformation_gradient, &a_I, &b_J, parameters);
                c_IJ += contraction * alpha;
            }
        }
    }
}

/// A wrapper that turns any hyper elastic material into an elliptic operator for use
/// with `fenris` assembly operations.
pub struct MaterialEllipticOperator<'a, Material>(&'a Material);

impl<'a, Material> MaterialEllipticOperator<'a, Material> {
    pub fn new(material: &'a Material) -> Self {
        Self(material)
    }
}

impl<'a, T, GeometryDim, Material> Operator<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: RealField,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    type SolutionDim = GeometryDim;
    type Parameters = Material::Parameters;
}

impl<'a, T, GeometryDim, Material> EllipticEnergy<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: RealField,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    fn compute_energy(&self, gradient: &OMatrix<T, GeometryDim, GeometryDim>, parameters: &Self::Parameters) -> T {
        let f = gradient.transpose();
        self.0.compute_energy_density(&f, parameters)
    }
}

impl<'a, T, GeometryDim, Material> EllipticOperator<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: RealField,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    fn compute_elliptic_operator(
        &self,
        gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, GeometryDim, Self::SolutionDim> {
        let f = gradient.transpose();
        let p = self.0.compute_stress_tensor(&f, parameters);
        p.transpose()
    }
}

impl<'a, T, GeometryDim, Material> EllipticContraction<T, GeometryDim> for MaterialEllipticOperator<'a, Material>
where
    T: RealField,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    fn contract(
        &self,
        gradient: &OMatrix<T, GeometryDim, GeometryDim>,
        a: &OVector<T, GeometryDim>,
        b: &OVector<T, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> OMatrix<T, Self::SolutionDim, Self::SolutionDim> {
        let f = gradient.transpose();
        self.0.compute_stress_contraction(&f, a, b, parameters)
    }

    fn symmetry(&self) -> Symmetry {
        Symmetry::Symmetric
    }

    #[allow(non_snake_case)]
    fn accumulate_contractions_into(
        &self,
        output: DMatrixSliceMut<T>,
        alpha: T,
        gradient: &OMatrix<T, GeometryDim, Self::SolutionDim>,
        a: DVectorSlice<T>,
        b: DVectorSlice<T>,
        parameters: &Self::Parameters,
    ) {
        // Note: This implementation is basically the same as the default implementation,
        // however we must
        let f = gradient.transpose();
        self.0
            .accumulate_stress_contractions_into(output, alpha, &f, a, b, parameters)
    }
}

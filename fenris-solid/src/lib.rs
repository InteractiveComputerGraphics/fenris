//! Solid mechanics functionality for `fenris`.
use fenris::allocators::SmallDimAllocator;
use fenris::assembly::operators::{
    EllipticContraction, EllipticEnergy, EllipticOperator, Operator,
};
use fenris::nalgebra::{DefaultAllocator, DimName, MatrixMN, RealField, Scalar, VectorN};
use fenris::{SmallDim, Symmetry};

pub trait HyperelasticMaterial<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    type Parameters: Clone + Default + 'static;

    /// Compute the energy density $\psi = \psi(\vec F)$ associated with the material.
    fn compute_energy_density(
        &self,
        deformation_gradient: &MatrixMN<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> T;

    /// Compute the First Piola-Kirchhoff stress tensor $\vec P = \vec P(\vec F)$.
    fn compute_stress_tensor(
        &self,
        deformation_gradient: &MatrixMN<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> MatrixMN<T, GeometryDim, GeometryDim>;

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
        deformation_gradient: &MatrixMN<T, GeometryDim, GeometryDim>,
        a: &VectorN<T, GeometryDim>,
        b: &VectorN<T, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> MatrixMN<T, GeometryDim, GeometryDim>;

    // /// Compute the contraction for a number of vectors at the same time, with the given
    // /// parameters.
    // ///
    // /// TODO: Clarify that we can assume symmetry here
    // fn accumulate_stress_contractions_into(&self,
    //                                        output: DMatrixSliceMut<T>,
    //                                        alpha: T,
    //                                        deformation_gradient: &MatrixMN<T, GeometryDim, GeometryDim>,
    //                                        a: DVectorSlice<T>,
    //                                        b: DVectorSlice<T>,
    //                                        parameters: &Self::Parameters)
    // {
    //     let _ = (output, alpha, deformation_gradient, a, b, parameters);
    //     todo!("Default impl")
    // }
}

/// A wrapper that turns any hyper elastic material into an elliptic operator for use
/// with `fenris` assembly operations.
pub struct MaterialEllipticOperator<'a, Material>(&'a Material);

impl<'a, T, GeometryDim, Material> Operator<T, GeometryDim>
    for MaterialEllipticOperator<'a, Material>
where
    T: Scalar,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    type SolutionDim = GeometryDim;
    type Parameters = Material::Parameters;
}

impl<'a, T, GeometryDim, Material> EllipticEnergy<T, GeometryDim>
    for MaterialEllipticOperator<'a, Material>
where
    T: RealField,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    fn compute_energy(
        &self,
        gradient: &MatrixMN<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> T {
        let f = gradient.transpose();
        self.0.compute_energy_density(&f, parameters)
    }
}

impl<'a, T, GeometryDim, Material> EllipticOperator<T, GeometryDim>
    for MaterialEllipticOperator<'a, Material>
where
    T: RealField,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    fn compute_elliptic_operator(
        &self,
        gradient: &MatrixMN<T, GeometryDim, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> MatrixMN<T, GeometryDim, Self::SolutionDim> {
        let f = gradient.transpose();
        let p = self.0.compute_stress_tensor(&f, parameters);
        p.transpose()
    }
}

impl<'a, T, GeometryDim, Material> EllipticContraction<T, GeometryDim>
    for MaterialEllipticOperator<'a, Material>
where
    T: RealField,
    GeometryDim: SmallDim,
    Material: HyperelasticMaterial<T, GeometryDim>,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    fn contract(
        &self,
        gradient: &MatrixMN<T, GeometryDim, GeometryDim>,
        a: &VectorN<T, GeometryDim>,
        b: &VectorN<T, GeometryDim>,
        parameters: &Self::Parameters,
    ) -> MatrixMN<T, Self::SolutionDim, Self::SolutionDim> {
        let f = gradient.transpose();
        self.0.compute_stress_contraction(&f, a, b, parameters)
    }

    fn symmetry(&self) -> Symmetry {
        Symmetry::Symmetric
    }

    // fn accumulate_contractions_into(&self,
    //                                 output: DMatrixSliceMut<T>,
    //                                 alpha: T,
    //                                 gradient: &MatrixMN<T, GeometryDim, Self::SolutionDim>,
    //                                 a: DVectorSlice<T>,
    //                                 b: DVectorSlice<T>,
    //                                 parameters: &Self::Parameters) {
    //     let _ = (alpha, output, gradient, a, b, parameters);
    //     todo!("Call stress batch contraction method")
    // }
}

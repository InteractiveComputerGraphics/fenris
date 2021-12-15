use crate::element::Tet4Element;
use nalgebra::RealField;
use crate::quadrature::QuadraturePair;
use crate::quadrature::{univariate, tensor, total_order};
use crate::element::*;
use crate::mesh::Mesh;
use crate::nalgebra::{Scalar, DefaultAllocator, DimName};
use crate::allocators::{BiDimAllocator};
use crate::assembly::local::UniformQuadratureTable;

/// A canonical quadrature for integrating the mass matrix terms.
///
/// The quadrature exactly integrates the expression
/// <div>$$
///   \int_K \phi_i \, \phi_j \, \mathrm{d} x
/// $$</div>
/// on a region $K$ for basis functions $\phi_k$.
pub trait CanonicalMassQuadrature {
    type Quadrature;

    fn canonical_mass_quadrature() -> Self::Quadrature;
}

/// A canonical quadrature for integrating the stiffness terms.
///
/// The quadrature exactly integrates the expression
/// <div>$$
///   \int_K \nabla \phi_i \cdot \nabla \phi_j \, \mathrm{d} x
/// $$</div>
/// on a region $K$ for basis functions $\phi_k$.
pub trait CanonicalStiffnessQuadrature {
    type Quadrature;

    fn canonical_stiffness_quadrature() -> Self::Quadrature;
}

macro_rules! impl_canonical_mass_for_element {
    ($element:ty, $quadrature:expr) => {
        impl<T> CanonicalMassQuadrature for $element
        where
            T: RealField
        {
            type Quadrature = QuadraturePair<T, <$element as ReferenceFiniteElement<T>>::ReferenceDim>;

            fn canonical_mass_quadrature() -> Self::Quadrature {
                $quadrature
            }
        }
    }
}

macro_rules! impl_canonical_stiffness_for_element {
    ($element:ty, $quadrature:expr) => {
        impl<T> CanonicalStiffnessQuadrature for $element
        where
            T: RealField
        {
            type Quadrature = QuadraturePair<T, <$element as ReferenceFiniteElement<T>>::ReferenceDim>;

            fn canonical_stiffness_quadrature() -> Self::Quadrature {
                $quadrature
            }
        }
    }
}

// Segment elements
impl_canonical_mass_for_element!(Segment2d2Element<T>, univariate::gauss(2));
impl_canonical_stiffness_for_element!(Segment2d2Element<T>, univariate::gauss(1));

// Triangular elements
impl_canonical_mass_for_element!(Tri3d2Element<T>, total_order::triangle(2).unwrap());
impl_canonical_mass_for_element!(Tri3d3Element<T>, total_order::triangle(2).unwrap());
impl_canonical_mass_for_element!(Tri6d2Element<T>, total_order::triangle(4).unwrap());
impl_canonical_stiffness_for_element!(Tri3d2Element<T>, total_order::triangle(1).unwrap());
impl_canonical_stiffness_for_element!(Tri3d3Element<T>, total_order::triangle(1).unwrap());
impl_canonical_stiffness_for_element!(Tri6d2Element<T>, total_order::triangle(2).unwrap());

// Quadrilateral elements
impl_canonical_mass_for_element!(Quad4d2Element<T>, tensor::quadrilateral_gauss(2));
impl_canonical_mass_for_element!(Quad9d2Element<T>, tensor::quadrilateral_gauss(3));
impl_canonical_stiffness_for_element!(Quad4d2Element<T>, tensor::quadrilateral_gauss(2));
impl_canonical_stiffness_for_element!(Quad9d2Element<T>, tensor::quadrilateral_gauss(3));

// Tetrahedral elements
impl_canonical_mass_for_element!(Tet4Element<T>, total_order::tetrahedron(2).unwrap());
impl_canonical_mass_for_element!(Tet10Element<T>, total_order::tetrahedron(4).unwrap());
impl_canonical_mass_for_element!(Tet20Element<T>, total_order::tetrahedron(6).unwrap());
impl_canonical_stiffness_for_element!(Tet4Element<T>, total_order::tetrahedron(1).unwrap());
impl_canonical_stiffness_for_element!(Tet10Element<T>, total_order::tetrahedron(2).unwrap());
impl_canonical_stiffness_for_element!(Tet20Element<T>, total_order::tetrahedron(4).unwrap());

// Hexahedral elements
impl_canonical_mass_for_element!(Hex8Element<T>, tensor::hexahedron_gauss(2));
impl_canonical_mass_for_element!(Hex20Element<T>, tensor::hexahedron_gauss(3));
impl_canonical_mass_for_element!(Hex27Element<T>, tensor::hexahedron_gauss(3));
impl_canonical_stiffness_for_element!(Hex8Element<T>, tensor::hexahedron_gauss(2));
impl_canonical_stiffness_for_element!(Hex20Element<T>, tensor::hexahedron_gauss(3));
impl_canonical_stiffness_for_element!(Hex27Element<T>, tensor::hexahedron_gauss(3));

impl<T, D, C> CanonicalMassQuadrature for Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: ElementConnectivity<T, GeometryDim=D>,
    C::Element: CanonicalMassQuadrature<Quadrature=QuadraturePair<T, C::ReferenceDim>>,
    DefaultAllocator: BiDimAllocator<T, C::GeometryDim, C::ReferenceDim>
{
    type Quadrature = UniformQuadratureTable<T, C::ReferenceDim>;

    fn canonical_mass_quadrature() -> Self::Quadrature {
        UniformQuadratureTable::from_quadrature(C::Element::canonical_mass_quadrature())
    }
}

impl<T, D, C> CanonicalStiffnessQuadrature for Mesh<T, D, C>
    where
        T: Scalar,
        D: DimName,
        C: ElementConnectivity<T, GeometryDim=D>,
        C::Element: CanonicalStiffnessQuadrature<Quadrature=QuadraturePair<T, C::ReferenceDim>>,
        DefaultAllocator: BiDimAllocator<T, C::GeometryDim, C::ReferenceDim>
{
    type Quadrature = UniformQuadratureTable<T, C::ReferenceDim>;

    fn canonical_stiffness_quadrature() -> Self::Quadrature {
        UniformQuadratureTable::from_quadrature(C::Element::canonical_stiffness_quadrature())
    }
}
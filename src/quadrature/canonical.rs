use crate::assembly::local::{UniformQuadratureTable};
use crate::element::Tet4Element;
use crate::element::*;
use crate::connectivity::*;
use crate::mesh::Mesh;
use crate::quadrature::QuadraturePair;
use crate::quadrature::{tensor, total_order};
use crate::Real;

/// A canonical quadrature for integrating the mass matrix terms.
///
/// The quadrature exactly integrates the expression
/// <div>$$
///   \int_K \phi_i \, \phi_j \, \mathrm{d} x
/// $$</div>
/// on a region $K$ for basis functions $\phi_k$.
pub trait CanonicalMassQuadrature {
    type Quadrature;

    fn canonical_mass_quadrature(&self) -> Self::Quadrature;
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

    fn canonical_stiffness_quadrature(&self) -> Self::Quadrature;
}

macro_rules! impl_canonical_rule_for_element {
    ($trait_name:ty, $method_name:ident, $connectivity:ty, $element:ty, $quadrature:expr) => {
        impl<T> $trait_name for $element
        where
            T: Real,
        {
            type Quadrature = QuadraturePair<T, ConnectivityReferenceDim<T, $connectivity>>;

            fn $method_name(&self) -> Self::Quadrature {
                $quadrature
            }
        }

        impl<T> $trait_name for Mesh<T, ConnectivityGeometryDim<T, $connectivity>, $connectivity>
        where
            T: Real
        {
            type Quadrature = UniformQuadratureTable<T, ConnectivityReferenceDim<T, $connectivity>>;

            fn $method_name(&self) -> Self::Quadrature {
                UniformQuadratureTable::from_quadrature($quadrature)
            }
        }
    }
}

macro_rules! impl_canonical_mass_for_element {
    ($connectivity:ty, $element:ty, $quadrature:expr) => {
        impl_canonical_rule_for_element!(
            CanonicalMassQuadrature,
            canonical_mass_quadrature,
            $connectivity, $element, $quadrature);
    };
}

macro_rules! impl_canonical_stiffness_for_element {
    ($connectivity:ty, $element:ty, $quadrature:expr) => {
        impl_canonical_rule_for_element!(
            CanonicalStiffnessQuadrature,
            canonical_stiffness_quadrature,
            $connectivity, $element, $quadrature);
    };
}

// Triangular elements
impl_canonical_mass_for_element!(Tri3d2Connectivity, Tri3d2Element<T>, total_order::triangle(2).unwrap());
impl_canonical_mass_for_element!(Tri6d2Connectivity, Tri6d2Element<T>, total_order::triangle(4).unwrap());
impl_canonical_stiffness_for_element!(Tri3d2Connectivity, Tri3d2Element<T>, total_order::triangle(1).unwrap());
impl_canonical_stiffness_for_element!(Tri6d2Connectivity, Tri6d2Element<T>, total_order::triangle(2).unwrap());

// Quadrilateral elements
impl_canonical_mass_for_element!(Quad4d2Connectivity, Quad4d2Element<T>, tensor::quadrilateral_gauss(2));
impl_canonical_mass_for_element!(Quad9d2Connectivity, Quad9d2Element<T>, tensor::quadrilateral_gauss(3));
impl_canonical_stiffness_for_element!(Quad4d2Connectivity, Quad4d2Element<T>, tensor::quadrilateral_gauss(2));
impl_canonical_stiffness_for_element!(Quad9d2Connectivity, Quad9d2Element<T>, tensor::quadrilateral_gauss(3));

// Tetrahedral elements
impl_canonical_mass_for_element!(Tet4Connectivity, Tet4Element<T>, total_order::tetrahedron(2).unwrap());
impl_canonical_mass_for_element!(Tet10Connectivity, Tet10Element<T>, total_order::tetrahedron(4).unwrap());
impl_canonical_mass_for_element!(Tet20Connectivity, Tet20Element<T>, total_order::tetrahedron(6).unwrap());
impl_canonical_stiffness_for_element!(Tet4Connectivity, Tet4Element<T>, total_order::tetrahedron(1).unwrap());
impl_canonical_stiffness_for_element!(Tet10Connectivity, Tet10Element<T>, total_order::tetrahedron(2).unwrap());
impl_canonical_stiffness_for_element!(Tet20Connectivity, Tet20Element<T>, total_order::tetrahedron(4).unwrap());

// Hexahedral elements
impl_canonical_mass_for_element!(Hex8Connectivity, Hex8Element<T>, tensor::hexahedron_gauss(2));
impl_canonical_mass_for_element!(Hex20Connectivity, Hex20Element<T>, tensor::hexahedron_gauss(3));
impl_canonical_mass_for_element!(Hex27Connectivity, Hex27Element<T>, tensor::hexahedron_gauss(3));
impl_canonical_stiffness_for_element!(Hex8Connectivity, Hex8Element<T>, tensor::hexahedron_gauss(2));
impl_canonical_stiffness_for_element!(Hex20Connectivity, Hex20Element<T>, tensor::hexahedron_gauss(3));
impl_canonical_stiffness_for_element!(Hex27Connectivity, Hex27Element<T>, tensor::hexahedron_gauss(3));
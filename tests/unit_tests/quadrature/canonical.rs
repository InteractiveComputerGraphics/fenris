use fenris::allocators::BiDimAllocator;
use fenris::assembly::local::{assemble_element_elliptic_matrix, assemble_element_mass_matrix};
use fenris::assembly::operators::LaplaceOperator;
use fenris::element::*;
use fenris::nalgebra::{DefaultAllocator, Dyn};
use fenris::quadrature;
use fenris::quadrature::{
    CanonicalMassQuadrature, CanonicalStiffnessQuadrature, Quadrature, QuadraturePair2d, QuadraturePair3d,
};
use fenris::Real;
use matrixcompare::comparators::FloatElementwiseComparator;
use matrixcompare::{assert_matrix_eq, compare_matrices};
use nalgebra::{DMatrix, DMatrixViewMut, DVector, DVectorView, MatrixViewMut, OMatrix, U2, U3};
use paste::paste;

fn assemble_mass_for_element<T, Element>(
    element: &Element,
    quadrature: impl Quadrature<T, Element::ReferenceDim>,
) -> DMatrix<T>
where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    DefaultAllocator: BiDimAllocator<T, Element::GeometryDim, Element::ReferenceDim>,
{
    let n = element.num_nodes();
    let mut output = DMatrix::zeros(n, n);
    let density = vec![T::one(); quadrature.weights().len()];
    assemble_element_mass_matrix(
        &mut output,
        element,
        quadrature.weights(),
        quadrature.points(),
        &density,
        1,
        &mut vec![T::one(); n],
    )
    .unwrap();
    output
}

fn assemble_stiffness_for_element<T, Element>(
    element: &Element,
    quadrature: impl Quadrature<T, Element::ReferenceDim>,
) -> DMatrix<T>
where
    T: Real,
    Element: VolumetricFiniteElement<T>,
    DefaultAllocator: BiDimAllocator<T, Element::GeometryDim, Element::ReferenceDim>,
{
    let n = element.num_nodes();
    let mut output = DMatrix::zeros(n, n);
    let dummy_data = vec![(); quadrature.weights().len()];
    let u_element = DVector::zeros(n);
    let operator = LaplaceOperator;
    assemble_element_elliptic_matrix::<T, Element, LaplaceOperator>(
        DMatrixViewMut::from(&mut output),
        element,
        &operator,
        DVectorView::from(&u_element),
        quadrature.weights(),
        quadrature.points(),
        &dummy_data,
        MatrixViewMut::from(&mut OMatrix::<T, Element::ReferenceDim, Dyn>::zeros(n)),
    )
    .unwrap();
    output
}

macro_rules! test_canonical_assembly_is_exact_and_minimal {
    ($test_name:ident, $element:ident, $reference_quadrature:expr, $quadrature_iter:expr,
        $canonical_quadrature_method:ident,
        $assembly_fn:ident) => {
        #[test]
        fn $test_name() {
            let element = $element::<f64>::reference();

            let reference_quadrature = $reference_quadrature;
            let canonical_quadrature = element.$canonical_quadrature_method();
            let canonical_matrix = $assembly_fn(&element, &canonical_quadrature);
            let reference_matrix = $assembly_fn(&element, &reference_quadrature);

            let ulp_tol = 64;
            assert_matrix_eq!(canonical_matrix, reference_matrix, comp = float, ulp = ulp_tol);

            let comparator = FloatElementwiseComparator::<f64>::default().ulp(ulp_tol);
            let minimal_exact_quadrature = $quadrature_iter
                .find(|candidate_quadrature| {
                    let candidate_matrix = $assembly_fn(&element, &candidate_quadrature);
                    compare_matrices(&candidate_matrix, &reference_matrix, &comparator).is_ok()
                })
                .expect("Internal error: Could not find a minimal quadrature");

            assert_eq!(
                canonical_quadrature.weights().len(),
                minimal_exact_quadrature.weights().len()
            );
        }
    };
}

macro_rules! test_canonical_mass_assembly_is_exact_and_minimal {
    ($element:ident, $reference_quadrature:expr, $quadrature_iter:expr) => {
        paste! {
            test_canonical_assembly_is_exact_and_minimal!(
                [<$element:snake _canonical_mass_assembly>],
                $element,
                $reference_quadrature,
                $quadrature_iter,
                canonical_mass_quadrature,
                assemble_mass_for_element);

            test_canonical_assembly_is_exact_and_minimal!(
                [<$element:snake _canonical_stiffness_assembly>],
                $element,
                $reference_quadrature,
                $quadrature_iter,
                canonical_stiffness_quadrature,
                assemble_stiffness_for_element);
        }
    };
}

fn tet_reference_quadrature() -> impl Quadrature<f64, U3> {
    quadrature::total_order::tetrahedron(10).unwrap()
}

fn tet_quadrature_iter() -> impl Iterator<Item = QuadraturePair3d<f64>> {
    (0..=10).map(|i| quadrature::total_order::tetrahedron(i).unwrap())
}

fn hex_reference_quadrature() -> impl Quadrature<f64, U3> {
    quadrature::tensor::hexahedron_gauss(8)
}

fn hex_quadrature_iter() -> impl Iterator<Item = QuadraturePair3d<f64>> {
    (1..=8).map(|i| quadrature::tensor::hexahedron_gauss(i))
}

fn quad_reference_quadrature() -> impl Quadrature<f64, U2> {
    quadrature::tensor::quadrilateral_gauss(8)
}

fn quad_quadrature_iter() -> impl Iterator<Item = QuadraturePair2d<f64>> {
    (1..=8).map(|i| quadrature::tensor::quadrilateral_gauss(i))
}

fn tri_reference_quadrature() -> impl Quadrature<f64, U2> {
    quadrature::total_order::triangle(10).unwrap()
}

fn tri_quadrature_iter() -> impl Iterator<Item = QuadraturePair2d<f64>> {
    (0..=10).map(|i| quadrature::total_order::triangle(i).unwrap())
}

// Triangle elements
test_canonical_mass_assembly_is_exact_and_minimal!(Tri3d2Element, tri_reference_quadrature(), tri_quadrature_iter());
test_canonical_mass_assembly_is_exact_and_minimal!(Tri6d2Element, tri_reference_quadrature(), tri_quadrature_iter());

// Quadrilateral elements
test_canonical_mass_assembly_is_exact_and_minimal!(Quad4d2Element, quad_reference_quadrature(), quad_quadrature_iter());
test_canonical_mass_assembly_is_exact_and_minimal!(Quad9d2Element, quad_reference_quadrature(), quad_quadrature_iter());

// Tetrahedral elements
test_canonical_mass_assembly_is_exact_and_minimal!(Tet4Element, tet_reference_quadrature(), tet_quadrature_iter());
test_canonical_mass_assembly_is_exact_and_minimal!(Tet10Element, tet_reference_quadrature(), tet_quadrature_iter());
test_canonical_mass_assembly_is_exact_and_minimal!(Tet20Element, tet_reference_quadrature(), tet_quadrature_iter());

// Hexahedral elements
test_canonical_mass_assembly_is_exact_and_minimal!(Hex8Element, hex_reference_quadrature(), hex_quadrature_iter());
test_canonical_mass_assembly_is_exact_and_minimal!(Hex20Element, hex_reference_quadrature(), hex_quadrature_iter());
test_canonical_mass_assembly_is_exact_and_minimal!(Hex27Element, hex_reference_quadrature(), hex_quadrature_iter());

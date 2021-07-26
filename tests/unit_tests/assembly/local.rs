use fenris::allocators::{BiDimAllocator, SmallDimAllocator};
use fenris::assembly::local::assemble_generalized_element_mass;
use fenris::element::{MatrixSliceMut, Quad4d2Element, VolumetricFiniteElement};
use fenris::geometry::Quad2d;
use fenris::nalgebra::{
    DMatrix, DVector, DefaultAllocator, DimName, Matrix4, MatrixN, Point, Point2, RealField, VectorN, U2, U8,
};
use fenris::quadrature;
use fenris::quadrature::QuadraturePair;
use itertools::izip;
use num::Zero;

mod elliptic;
mod source;

fn reference_quad<T>() -> Quad2d<T>
where
    T: RealField,
{
    Quad2d([
        Point2::new(-T::one(), -T::one()),
        Point2::new(T::one(), -T::one()),
        Point2::new(T::one(), T::one()),
        Point2::new(-T::one(), T::one()),
    ])
}

#[test]
fn analytic_comparison_of_element_mass_matrix_for_reference_element() {
    let density = 3.0;

    let quadrature = quadrature::total_order::quadrilateral(5).unwrap();
    let quad = Quad4d2Element::from(reference_quad());

    let ndof = 8;
    let mut m = DMatrix::zeros(ndof, ndof);
    assemble_generalized_element_mass::<_, U2, _, _>(MatrixSliceMut::from(&mut m), &quad, density, &quadrature);

    #[rustfmt::skip]
    let expected4x4 = (density / 9.0) * Matrix4::new(
        4.0, 2.0, 1.0, 2.0,
        2.0, 4.0, 2.0, 1.0,
        1.0, 2.0, 4.0, 2.0,
        2.0, 1.0, 2.0, 4.0);

    let mut expected8x8: MatrixN<f64, U8> = MatrixN::zero();
    expected8x8
        .slice_with_steps_mut((0, 0), (4, 4), (1, 1))
        .copy_from(&expected4x4);
    expected8x8
        .slice_with_steps_mut((1, 1), (4, 4), (1, 1))
        .copy_from(&expected4x4);

    let diff = m - expected8x8;
    assert!(diff.norm() <= 1e-6);
}

/// An artificial density function that we use to validate that quadrature parameters are correctly
/// employed in the assembly.
fn density<D>(x: &Point<f64, D>) -> f64
where
    D: DimName,
    DefaultAllocator: SmallDimAllocator<f64, D>,
{
    x.coords.norm_squared()
}

/// Constructs a specific quadrature rule for the given element by transforming an input
/// quadrature rule for the reference element.
fn construct_quadrature_rule_for_element<Element>(
    element: &Element,
    reference_rule: &QuadraturePair<f64, Element::GeometryDim>,
) -> QuadraturePair<f64, Element::GeometryDim>
where
    Element: VolumetricFiniteElement<f64>,
    DefaultAllocator: SmallDimAllocator<f64, Element::GeometryDim>,
{
    // Construct a quadrature rule for this particular element
    let (weights, points) = reference_rule;
    izip!(weights, points)
        .map(|(w, p)| {
            let x = element.map_reference_coords(&p);
            let j = element.reference_jacobian(&p);
            let new_w = w * j.determinant().abs();
            (new_w, x)
        })
        .unzip()
}

fn u_element_from_vertices_and_u_exact<D, S>(
    vertices: &[Point<f64, D>],
    u_exact: impl Fn(&Point<f64, D>) -> VectorN<f64, S>,
) -> DVector<f64>
where
    D: DimName,
    S: DimName,
    DefaultAllocator: BiDimAllocator<f64, D, S>,
{
    let mut entries = Vec::with_capacity(D::dim());
    for v in vertices {
        let u = u_exact(v);
        entries.extend(u.iter());
    }
    DVector::from_vec(entries)
}

fn evaluate_density_at_quadrature_points<Element>(
    element: &Element,
    points: &[Point<f64, Element::GeometryDim>],
    density: impl Fn(&Point<f64, Element::GeometryDim>) -> f64,
) -> Vec<f64>
where
    Element: VolumetricFiniteElement<f64>,
    DefaultAllocator: SmallDimAllocator<f64, Element::GeometryDim>,
{
    points
        .iter()
        .map(|xi| element.map_reference_coords(xi))
        .map(|x| density(&x))
        .collect()
}

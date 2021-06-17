use fenris::assembly::local::assemble_generalized_element_mass;
use fenris::element::{MatrixSliceMut, Quad4d2Element};
use fenris::geometry::Quad2d;
use fenris::nalgebra::{DMatrix, Matrix4, MatrixN, Point2, RealField, U2, U8};
use fenris::quadrature;
use num::Zero;

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
    assemble_generalized_element_mass::<_, U2, _, _>(
        MatrixSliceMut::from(&mut m),
        &quad,
        density,
        &quadrature,
    );

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

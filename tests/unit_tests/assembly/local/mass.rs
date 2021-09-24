use fenris::element::{Tet4Element, Tet20Element, MatrixSliceMut, FiniteElement};
use nalgebra::{Point3, Vector1, Matrix2, Matrix3};
use fenris::nalgebra::{DVector, DVectorSlice, DMatrix};
use fenris::error::estimate_element_L2_error_squared;
use fenris::quadrature;
use fenris::quadrature::Quadrature;
use fenris::assembly::local::assemble_element_mass_matrix;
use matrixcompare::{assert_scalar_eq, assert_matrix_eq};

#[test]
#[allow(non_snake_case)]
fn squared_norm_agrees_with_element_mass_matrix_quadratic_form_tet20() {
    // We test that the element mass matrix is correct by carefully constructing functions f, g and rho such that
    //  ||g||^2 = f_h^T M f_h
    // where ||g|| is the L2 norm of g and f_h are the FE interpolation weights of f.
    // We use a cubic element so that we can test more involved functions.

    let a = Point3::new(2.0, 0.0, 1.0);
    let b = Point3::new(3.0, 4.0, 1.0);
    let c = Point3::new(1.0, 1.0, 2.0);
    let d = Point3::new(3.0, 1.0, 4.0);
    let tet4_element = Tet4Element::from_vertices([a, b, c, d]);

    // f is an arbitrary quadratic function
    let f = |x, y, z| 3.0 * x * x - 4.0 * y * y + 2.0 * z * z - x * y + 3.0 * x * z + y + z + 5.0;

    // omega is an arbitrary linear function
    let omega = |x, y, z| 3.0 * x + 2.0 * y - 4.0 * z + 2.0;

    // Consequently, g becomes a cubic function
    let g = |x, y, z| omega(x, y, z) * f(x, y, z);

    // And rho becomes a quadratic function
    let rho = |x, y, z| f64::powi(omega(x, y, z), 2);

    // A Tet20 element can reproduce any cubic solution field exactly
    let element = Tet20Element::from(&tet4_element);

    // Interpolate f and g onto the nodes of the element
    let f_h: DVector<_> = DVector::from_iterator(20, element.vertices()
        .iter()
        .map(|v| f(v.x, v.y, v.z)));
    let g_h: DVector<_> = DVector::from_iterator(20, element.vertices()
        .iter()
        .map(|v| g(v.x, v.y, v.z)));

    // We need a quadrature rule of polynomial strength 8 to guarantee exact computation of the integrals
    // (since mass matrix needs rho * phi_I * phi_J and phi_I, phi_J are cubic)
    let quadrature = quadrature::total_order::tetrahedron(8).unwrap();
    let quadrature_density: Vec<_> = quadrature.points()
        .iter()
        .map(|xi| {
            let p = element.map_reference_coords(xi);
            rho(p.x, p.y, p.z)
        })
        .collect();

    let mut basis_buffer = vec![0.0; 20];

    // Compute the squared norm of g_h as the L^2 squared error ||0 - g_h||^2
    let g_h_squared_norm = estimate_element_L2_error_squared(&element,
                                                             |_| Vector1::zeros(),
                                                             DVectorSlice::from(&g_h),
                                                             quadrature.weights(),
                                                             quadrature.points(),
                                                             &mut basis_buffer);

    let mut M = DMatrix::zeros(20, 20);
    assemble_element_mass_matrix(MatrixSliceMut::from(&mut M), &element, quadrature.weights(), quadrature.points(), &quadrature_density, 1, &mut basis_buffer).unwrap();

    let fT_M_f = f_h.dot(&(&M * &f_h));

    assert_scalar_eq!(g_h_squared_norm, fT_M_f, comp = float);

    // So far we've only checked for solution dim == 1.
    // In general, it holds that
    //  M_s = M kron I_s,
    // where M_s is the mass matrix with solution dim s, I_s is the s x s identity matrix
    // and kron denotes the kronecker product. So let's check that this holds for some values of s!
    {
        let mut M2 = DMatrix::zeros(40, 40);
        assemble_element_mass_matrix(MatrixSliceMut::from(&mut M2), &element, quadrature.weights(), quadrature.points(), &quadrature_density, 2, &mut basis_buffer).unwrap();
        assert_matrix_eq!(M2, M.kronecker(&Matrix2::identity()));
    }

    {
        let mut M3 = DMatrix::zeros(60, 60);
        assemble_element_mass_matrix(MatrixSliceMut::from(&mut M3), &element, quadrature.weights(), quadrature.points(), &quadrature_density, 3, &mut basis_buffer).unwrap();
        assert_matrix_eq!(M3, M.kronecker(&Matrix3::identity()));
    }

}
use fenris::assembly::global::CsrAssembler;
use fenris::assembly::local::{assemble_element_mass_matrix, Density, ElementMassAssembler, GeneralQuadratureTable};
use fenris::element::{ElementConnectivity, FiniteElement, Tet20Element, Tet4Element};
use fenris::error::{estimate_L2_error_squared, estimate_element_L2_error_squared};
use fenris::integrate::IntegrationWorkspace;
use fenris::mesh::procedural::create_unit_box_uniform_tet_mesh_3d;
use fenris::mesh::Tet10Mesh;
use fenris::nalgebra::{DMatrix, DVector, DVectorSlice};
use fenris::quadrature;
use fenris::quadrature::Quadrature;
use matrixcompare::{assert_matrix_eq, assert_scalar_eq};
use nalgebra::{Matrix2, Matrix3, MatrixSliceMut, Point3, Vector1};
use std::iter::repeat;

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
    let f_h: DVector<_> = DVector::from_iterator(20, element.vertices().iter().map(|v| f(v.x, v.y, v.z)));
    let g_h: DVector<_> = DVector::from_iterator(20, element.vertices().iter().map(|v| g(v.x, v.y, v.z)));

    // We need a quadrature rule of polynomial strength 8 to guarantee exact computation of the integrals
    // (since mass matrix needs rho * phi_I * phi_J and phi_I, phi_J are cubic)
    let quadrature = quadrature::total_order::tetrahedron(8).unwrap();
    let quadrature_density: Vec<_> = quadrature
        .points()
        .iter()
        .map(|xi| {
            let p = element.map_reference_coords(xi);
            rho(p.x, p.y, p.z)
        })
        .collect();

    let mut basis_buffer = vec![0.0; 20];
    let mut workspace = IntegrationWorkspace::default();

    // Compute the squared norm of g_h as the L^2 squared error ||0 - g_h||^2
    let g_h_squared_norm = estimate_element_L2_error_squared(
        &element,
        &|_: &Point3<_>| Vector1::zeros(),
        DVectorSlice::from(&g_h),
        quadrature.weights(),
        quadrature.points(),
        &mut workspace,
    );

    let mut M = DMatrix::zeros(20, 20);
    assemble_element_mass_matrix(
        MatrixSliceMut::from(&mut M),
        &element,
        quadrature.weights(),
        quadrature.points(),
        &quadrature_density,
        1,
        &mut basis_buffer,
    )
    .unwrap();

    let fT_M_f = f_h.dot(&(&M * &f_h));

    assert_scalar_eq!(g_h_squared_norm, fT_M_f, comp = float);

    // So far we've only checked for solution dim == 1.
    // In general, it holds that
    //  M_s = M kron I_s,
    // where M_s is the mass matrix with solution dim s, I_s is the s x s identity matrix
    // and kron denotes the kronecker product. So let's check that this holds for some values of s!
    {
        let mut M2 = DMatrix::zeros(40, 40);
        assemble_element_mass_matrix(
            MatrixSliceMut::from(&mut M2),
            &element,
            quadrature.weights(),
            quadrature.points(),
            &quadrature_density,
            2,
            &mut basis_buffer,
        )
        .unwrap();
        assert_matrix_eq!(M2, M.kronecker(&Matrix2::identity()));
    }

    {
        let mut M3 = DMatrix::zeros(60, 60);
        assemble_element_mass_matrix(
            MatrixSliceMut::from(&mut M3),
            &element,
            quadrature.weights(),
            quadrature.points(),
            &quadrature_density,
            3,
            &mut basis_buffer,
        )
        .unwrap();
        assert_matrix_eq!(M3, M.kronecker(&Matrix3::identity()));
    }
}

#[test]
#[allow(non_snake_case)]
fn squared_norm_agrees_with_mass_matrix_quadratic_form_full_mesh_tet10() {
    // This is basically the same test as the test for a single element - except we do it for a full mesh this time.
    // Unfortunately we cannot use Tet20Mesh at the moment because we're lacking the conversion functionality,
    // so we use Tet10 instead and lower the order of the functions involved
    let tet4_mesh = create_unit_box_uniform_tet_mesh_3d(3);
    let mesh = Tet10Mesh::from(&tet4_mesh);

    // f is an arbitrary linear function
    let f = |x, y, z| x * y + 3.0 * x * z + y + z + 5.0;
    // omega is an arbitrary linear function
    let omega = |x, y, z| 3.0 * x + 2.0 * y - 4.0 * z + 2.0;
    // Consequently, g becomes a quadratic function
    let g = |x, y, z| omega(x, y, z) * f(x, y, z);
    // And rho becomes a quadratic function
    let rho = |x, y, z| Density(f64::powi(omega(x, y, z), 2));

    // We need to construct a quadrature table where every element has a different quadrature rule
    // in order to accommodate varying density values (TODO: Should be possible to combine a uniform quadrature rule
    // with per-element data/parameters)
    let quadrature_table = {
        let base_quadrature = quadrature::total_order::tetrahedron(6).unwrap();
        let num_elements = mesh.connectivity().len();
        let points: Vec<_> = repeat(base_quadrature.1.clone())
            .take(num_elements)
            .collect();
        let weights: Vec<_> = repeat(base_quadrature.0.clone())
            .take(num_elements)
            .collect();
        let density: Vec<_> = mesh
            .connectivity()
            .iter()
            .map(|conn| {
                let element = conn.element(mesh.vertices()).unwrap();
                // Evaluate densities at quadrature points
                let densities = base_quadrature
                    .points()
                    .iter()
                    .map(|xi| {
                        let p = element.map_reference_coords(xi);
                        rho(p.x, p.y, p.z)
                    })
                    .collect::<Vec<_>>();
                densities
            })
            .collect();

        GeneralQuadratureTable::from_points_weights_and_data(points.into(), weights.into(), density.into())
    };

    let element_mass_assembler = ElementMassAssembler::with_solution_dim(1)
        .with_space(&mesh)
        .with_quadrature_table(&quadrature_table);

    // We compute the squared norm by computing the error ||g - 0||^2_L^2, where 0 corresponds to a finite element
    // interpolation of the zero function
    let g_h_squared_norm = estimate_L2_error_squared(
        &mesh,
        &|p: &Point3<_>| Vector1::new(g(p.x, p.y, p.z)),
        &DVector::zeros(mesh.vertices().len()),
        &quadrature_table,
    )
    .unwrap();

    let M = CsrAssembler::default()
        .assemble(&element_mass_assembler)
        .unwrap();
    let f_h = DVector::from_iterator(mesh.vertices().len(), mesh.vertices().iter().map(|v| f(v.x, v.y, v.z)));
    let fT_M_f = f_h.dot(&(&M * &f_h));

    assert_scalar_eq!(fT_M_f, g_h_squared_norm, comp = float, ulp = 50);

    // Check that M_s = M kron I_s like in the single-element test

    // solution dim 2
    {
        let assembler2 = ElementMassAssembler::with_solution_dim(2)
            .with_space(&mesh)
            .with_quadrature_table(&quadrature_table);
        let M2 = CsrAssembler::default().assemble(&assembler2).unwrap();
        assert_matrix_eq!(M2, DMatrix::from(&M).kronecker(&Matrix2::identity()));
    }

    // solution dim 3
    {
        let assembler3 = ElementMassAssembler::with_solution_dim(3)
            .with_space(&mesh)
            .with_quadrature_table(&quadrature_table);
        let M3 = CsrAssembler::default().assemble(&assembler3).unwrap();
        assert_matrix_eq!(M3, DMatrix::from(&M).kronecker(&Matrix3::identity()));
    }
}

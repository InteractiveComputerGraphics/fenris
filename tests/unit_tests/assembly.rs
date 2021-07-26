// use fenris_solid::assembly::MaterialEllipticOperator;
// use fenris_solid::materials::{LameParameters, LinearElasticMaterial, StVKMaterial, YoungPoisson};
// use fenris_solid::ElasticMaterialModel;
// use fenris_solid::ElasticityModel;

mod global;
mod local;

// TODO: Re-enable/rewrite tests here as appropriate when possible (most tests rely on some
// solid mechanics stuff)

// #[derive(Debug, Copy, Clone)]
// struct MockIdentityMaterial;
//
// impl<T> ElasticMaterialModel<T, U2> for MockIdentityMaterial
// where
//     T: RealField,
// {
//     fn compute_strain_energy_density(&self, _deformation_gradient: &Matrix2<T>) -> T {
//         unimplemented!()
//     }
//
//     fn compute_stress_tensor(&self, _deformation_gradient: &Matrix2<T>) -> Matrix2<T> {
//         Matrix2::identity()
//     }
//
//     fn contract_stress_tensor_with(
//         &self,
//         _deformation_gradient: &Matrix2<T>,
//         _a: &OVector<T, U2>,
//         _b: &OVector<T, U2>,
//     ) -> Matrix2<T> {
//         Matrix2::zero()
//     }
// }
//
// #[derive(Debug, Copy, Clone)]
// struct MockSimpleMaterial;
//
// #[allow(non_snake_case)]
// impl<T> ElasticMaterialModel<T, U2> for MockSimpleMaterial
// where
//     T: RealField,
// {
//     fn compute_strain_energy_density(&self, _deformation_gradient: &Matrix2<T>) -> T {
//         unimplemented!()
//     }
//
//     fn compute_stress_tensor(&self, F: &Matrix2<T>) -> Matrix2<T> {
//         F - Matrix2::identity()
//     }
//
//     fn contract_stress_tensor_with(
//         &self,
//         _F: &Matrix2<T>,
//         a: &OVector<T, U2>,
//         b: &OVector<T, U2>,
//     ) -> Matrix2<T> {
//         Matrix2::identity() * a.dot(&b)
//     }
// }

// #[test]
// fn quad4d2_constant_displacement_gives_zero_elastic_forces_for_reference_element() {
//     let lame = fenris_solid::materials::LameParameters {
//         mu: 2.0,
//         lambda: 3.0,
//     };
//     let material = fenris_solid::materials::LinearElasticMaterial::from(lame);
//
//     let u = 3.0 * OMatrix::<f64, U2, Dynamic>::repeat(4, 1.0);
//
//     let quadrature = quad_quadrature_strength_5_f64();
//     let quad = Quad4d2Element::from(reference_quad());
//
//     let elliptic_operator = MaterialEllipticOperator(&material);
//     let mut f_e = DMatrix::zeros(2, 4);
//     assemble_generalized_element_elliptic_term(MatrixSliceMut::from(&mut f_e),
//                                                &quad,
//                                                &elliptic_operator,
//                                                &MatrixSlice::from(&u),
//                                                &quadrature);
//     assert!(f_e.norm() < 1e-14);
// }

// #[test]
// fn quad4d2_constant_displacement_gives_zero_elastic_forces_for_arbitrary_quad() {
//     let lame = fenris_solid::materials::LameParameters {
//         mu: 2.0,
//         lambda: 3.0,
//     };
//     let material = fenris_solid::materials::LinearElasticMaterial::from(lame);
//     let u = 3.0 * OMatrix::<f64, U2, Dynamic>::repeat(4, 1.0);
//
//     let quadrature = quad_quadrature_strength_5_f64();
//     let quad = Quad4d2Element::from_vertices([
//         Point2::new(-2.0, -3.0),
//         Point2::new(1.0, -1.0),
//         Point2::new(2.0, 4.0),
//         Point2::new(-1.0, 3.0),
//     ]);
//
//     let elliptic_operator = MaterialEllipticOperator(&material);
//     let mut f_e = DMatrix::zeros(2, 4);
//     assemble_generalized_element_elliptic_term(MatrixSliceMut::from(&mut f_e),
//                                                &quad,
//                                                &elliptic_operator,
//                                                &MatrixSlice::from(&u),
//                                                &quadrature);
//     assert!(f_e.norm() < 1e-14);
// }

// #[test]
// fn analytic_comparison_of_element_elastic_force_for_reference_element() {
//     let u = 3.0 * OMatrix::<f64, U2, Dynamic>::repeat(4, 1.0);
//     let quadrature = quad_quadrature_strength_5_f64();
//     let material = MockIdentityMaterial;
//     let quad = Quad4d2Element::from(reference_quad());
//
//     let elliptic_operator = MaterialEllipticOperator(&material);
//     let mut f_e = DMatrix::zeros(2, 4);
//     assemble_generalized_element_elliptic_term(MatrixSliceMut::from(&mut f_e),
//                                                &quad,
//                                                &elliptic_operator,
//                                                &MatrixSlice::from(&u),
//                                                &quadrature);
//     #[rustfmt::skip]
//     let expected = Matrix2x4::new(1.0, -1.0, -1.0,  1.0,
//                                   1.0,  1.0, -1.0, -1.0);
//     let diff = f_e - expected;
//     assert!(diff.norm() < 1e-14);
// }

// TODO: Test elastic forces for arbitrary element

// #[test]
// fn analytic_comparison_of_element_stiffness_matrix_for_reference_element() {
//     let u = 3.0 * OMatrix::<f64, U2, Dynamic>::repeat(4, 1.0);
//     let material = MockSimpleMaterial;
//     let quadrature = quad_quadrature_strength_5_f64();
//     let quad = Quad4d2Element::from(reference_quad());
//
//     let elliptic_operator = MaterialEllipticOperator(&material);
//     let mut a = DMatrix::zeros(8, 8);
//     assemble_generalized_element_stiffness(DMatrixSliceMut::from(&mut a),
//                                            &quad,
//                                            &elliptic_operator,
//                                            MatrixSlice::from(&u),
//                                            &quadrature);
//
//     // For the given mock material, the contraction yields tr(B) I,
//     // and so the integral over the element K reads
//     //  A^K_IJ = int_K tr(B_IJ) * I dX
//     // with tr(B_IJ) = grad phi_I dot grad phi_J
//     // in other words, the 2x2 matrix A^K_IJ corresponds to
//     // the value of the *scalar* Laplacian stiffness matrix for basis
//     // functions IJ multiplied by the 2x2 identity matrix.
//     #[rustfmt::skip]
//     let expected4x4 = Matrix4::new( 2.0/3.0, -1.0/6.0, -1.0/3.0, -1.0/6.0,
//                                    -1.0/6.0,  2.0/3.0, -1.0/6.0, -1.0/3.0,
//                                    -1.0/3.0, -1.0/6.0,  2.0/3.0, -1.0/6.0,
//                                    -1.0/6.0, -1.0/3.0, -1.0/6.0,  2.0/3.0);
//     let mut expected8x8: MatrixN<f64, U8> = MatrixN::zero();
//     expected8x8
//         .slice_with_steps_mut((0, 0), (4, 4), (1, 1))
//         .copy_from(&expected4x4);
//     expected8x8
//         .slice_with_steps_mut((1, 1), (4, 4), (1, 1))
//         .copy_from(&expected4x4);
//
//     let diff = a - expected8x8;
//     assert!(diff.norm() <= 1e-6);
// }

// #[test]
// #[allow(non_snake_case)]
// fn quad4d2_mass_matrix_vector_product_with_ones() {
//     // It can be shown that, assuming the transformation from the reference
//     // element to each individual element K is a linear transformation,
//     // then
//     //   (M * 1)_Ii = rho_0 * sum_{K in S_I} |det J_K|
//     // where S_I = { K | intersection of K and support of basis function I is non-empty},
//     // rho_0 is the rest density and J_K is the Jacobian of the (linear)
//     // transformation from the reference element to each element K.
//     //
//     // If furthermore all cells have the same size, we must only compute
//     // |det J| once and scale it by the number of elements the node appears in.
//
//     let resolutions = [1, 2, 3, 4, 8, 9, 11];
//     let quadrature = quad_quadrature_strength_5_f64();
//
//     for resolution in &resolutions {
//         let mesh = create_unit_square_uniform_quad_mesh_2d(*resolution);
//         let ndof = 2 * mesh.vertices().len();
//         let model = Quad4Model::from_mesh_and_quadrature(mesh, quadrature.clone());
//         let rho_0 = 3.0;
//
//         let mass_matrix = model.assemble_mass(rho_0).build_dense();
//
//         // Cells all have the same size
//         let cell_size = 1.0 / f64::from_usize(*resolution).unwrap();
//         let abs_det_J = cell_size * cell_size / 4.0;
//         let num_nodes = model.vertices().len();
//
//         let mut node_counts = vec![0u32; num_nodes];
//         for connectivity in model.connectivity() {
//             for node_index in &connectivity.0 {
//                 node_counts[*node_index] += 1;
//             }
//         }
//
//         let expected_values = node_counts
//             .iter()
//             .map(|i| rho_0 * abs_det_J * f64::from(*i))
//             .flat_map(|v| once(v).chain(once(v)));
//         let expected = DVector::from_iterator(ndof, expected_values);
//
//         let result = mass_matrix * DVector::repeat(ndof, 1.0);
//         let diff = result - expected;
//
//         assert!(diff.norm() < ndof as f64 * 1e-12);
//     }
// }

// #[test]
// #[allow(non_snake_case)]
// fn tet4_mass_matrix_vector_product_with_ones() {
//     // See the comment above for the explanation for this test
//     let resolutions = [1, 2, 3, 4, 8];
//     let quadrature = tet_quadrature_strength_5();
//
//     for resolution in &resolutions {
//         let mesh = create_rectangular_uniform_hex_mesh(1.0, 1, 1, 1, *resolution);
//         let mesh = Tet4Mesh::try_from(&PolyMesh3d::from(&mesh).triangulate().unwrap()).unwrap();
//
//         let model = Tet4Model::from_mesh_and_quadrature(mesh, quadrature.clone());
//         let ndof = model.ndof();
//         let rho_0 = 3.0;
//
//         let mass_matrix = model.assemble_mass(rho_0).build_dense();
//
//         // Cells all have the same size
//         let cell_size = 1.0 / f64::from_usize(*resolution).unwrap();
//         let abs_det_J = cell_size * cell_size * cell_size / (6.0 * 4.0);
//         let num_nodes = model.vertices().len();
//
//         let mut node_counts = vec![0u32; num_nodes];
//         for connectivity in model.connectivity() {
//             for node_index in &connectivity.0 {
//                 node_counts[*node_index] += 1;
//             }
//         }
//
//         let expected_values = node_counts
//             .iter()
//             .map(|i| rho_0 * abs_det_J * f64::from(*i))
//             .flat_map(|v| once(v).chain(once(v)).chain(once(v)));
//         let expected = DVector::from_iterator(ndof, expected_values);
//
//         let result = mass_matrix * DVector::repeat(ndof, 1.0);
//         println!("result: {}", result);
//         println!("expected: {}", expected);
//         let diff = result - expected;
//         assert!(diff.norm() < ndof as f64 * 1e-12);
//     }
// }

// /// Creates an instance of a VectorFunction that corresponds to the elastic pseudo forces F(u)
// /// given displacements u.
// fn create_single_element_elastic_force_vector_function<'a, Connectivity>(
//     element: &'a Connectivity::Element,
//     indices: &'a Connectivity,
//     material: impl ElasticMaterialModel<f64, Connectivity::GeometryDim> + 'a,
//     quadrature: impl Quadrature<f64, Connectivity::GeometryDim> + 'a,
// ) -> impl VectorFunction<f64> + 'a
// where
//     Connectivity: ElementConnectivity<
//         f64,
//         ReferenceDim = <Connectivity as ElementConnectivity<f64>>::GeometryDim,
//     >,
//     Connectivity::GeometryDim: DimName
//         + DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
//     DefaultAllocator: ElementConnectivityAllocator<f64, Connectivity>,
// {
//     let d = Connectivity::GeometryDim::dim();
//     let vector_space_dim = d * element.num_nodes();
//     let mut f_element = OMatrix::<f64, Connectivity::GeometryDim, Dynamic>::zeros(element.num_nodes());
//     VectorFunctionBuilder::with_dimension(vector_space_dim).with_function(move |f, u| {
//         let u_element: OMatrix<f64, Connectivity::GeometryDim, Dynamic> = indices.element_variables(u);
//         let elliptic_operator = MaterialEllipticOperator(&material);
//         assemble_generalized_element_elliptic_term(
//             MatrixSliceMut::from(&mut f_element),
//             element,
//             &elliptic_operator,
//             &MatrixSlice::from(&u_element),
//             &quadrature,
//         );
//         f.copy_from_slice(f_element.as_slice());
//     })
// }

// #[test]
// fn element_stiffness_matrix_is_negative_derivative_of_forces_for_linear_material_arbitrary_displacement(
// ) {
//     let u = DVector::from_vec(vec![3.0, -2.0, 1.0, -4.0, 13.0, -2.0, 13.0, 15.0]);
//     let lame = LameParameters {
//         mu: 2.0,
//         lambda: 3.0,
//     };
//     let material = LinearElasticMaterial::from(lame);
//
//     let h = 1e-6;
//     let quadrature = quad_quadrature_strength_5_f64();
//     let quad = Quad4d2Element::from_vertices([
//         Point2::new(0.5, 0.25),
//         Point2::new(1.25, 0.5),
//         Point2::new(1.5, 1.0),
//         Point2::new(0.25, 1.5),
//     ]);
//
//     let quad_indices = Quad4d2Connectivity([0, 1, 2, 3]);
//
//     let u_element: OMatrix<_, U2, Dynamic> = quad_indices.element_variables(&u);
//     let elliptic_operator = MaterialEllipticOperator(&material);
//     let mut a = DMatrix::zeros(8, 8);
//     assemble_generalized_element_stiffness(DMatrixSliceMut::from(&mut a),
//                                            &quad,
//                                            &elliptic_operator,
//                                            MatrixSlice::from(&u_element),
//                                            &quadrature);
//
//     let func = create_single_element_elastic_force_vector_function(
//         &quad,
//         &quad_indices,
//         &material,
//         &quadrature,
//     );
//
//     let a_approx = -approximate_jacobian(func, &u, &h);
//
//     let diff = a - a_approx;
//     assert!(diff.norm() < 1e-6);
// }
//
// #[test]
// fn tet4_element_stiffness_matrix_is_negative_derivative_of_forces_for_linear_material_arbitrary_displacement(
// ) {
//     let u = DVector::from_vec(vec![
//         0.1, -0.2, 0.1, -0.0, 0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.0, -0.2,
//     ]);
//     let lame = LameParameters {
//         mu: 2.0,
//         lambda: 3.0,
//     };
//     let material = LinearElasticMaterial::from(lame);
//
//     let h = 1e-6;
//     let quadrature = tet_quadrature_strength_5();
//     let tet = Tet4Element::from_vertices([
//         Point3::new(-1.0, -0.5, -1.0),
//         Point3::new(1.0, -0.5, 0.0),
//         Point3::new(0.0, 1.0, -1.0),
//         Point3::new(0.0, 0.0, 0.5),
//     ]);
//     let tet_conn = Tet4Connectivity([0, 1, 2, 3]);
//
//     let u_element: OMatrix<_, U3, Dynamic> = tet_conn.element_variables(&u);
//     let elliptic_operator = MaterialEllipticOperator(&material);
//     let mut a = DMatrix::zeros(12, 12);
//     assemble_generalized_element_stiffness(DMatrixSliceMut::from(&mut a),
//                                            &tet,
//                                            &elliptic_operator,
//                                            MatrixSlice::from(&u_element),
//                                            &quadrature);
//
//     let func = create_single_element_elastic_force_vector_function(
//         &tet,
//         &tet_conn,
//         &material,
//         &quadrature,
//     );
//
//     let a_approx = -approximate_jacobian(func, &u, &h);
//
//     let diff = &a - &a_approx;
//     assert!(diff.norm() < 1e-6);
// }
//
// #[test]
// fn element_stiffness_matrix_is_negative_derivative_of_forces_for_stvk_material_arbitrary_displacement(
// ) {
//     let u = DVector::from_vec(vec![3.0, -2.0, 1.0, -4.0, 13.0, -2.0, 13.0, 15.0]);
//     let lame = LameParameters {
//         mu: 2.0,
//         lambda: 3.0,
//     };
//     let material = StVKMaterial::from(lame);
//
//     let h = 1e-6;
//     let quadrature = quad_quadrature_strength_5_f64();
//     let quad = Quad4d2Element::from_vertices([
//         Point2::new(0.5, 0.25),
//         Point2::new(1.25, 0.5),
//         Point2::new(1.5, 1.0),
//         Point2::new(0.25, 1.5),
//     ]);
//
//     let quad_indices = Quad4d2Connectivity([0, 1, 2, 3]);
//
//     let u_element: OMatrix<_, U2, _> = quad_indices.element_variables(&u);
//     let elliptic_operator = MaterialEllipticOperator(&material);
//
//     let mut a = DMatrix::zeros(8, 8);
//     assemble_generalized_element_stiffness(DMatrixSliceMut::from(&mut a),
//                                            &quad,
//                                            &elliptic_operator,
//                                            MatrixSlice::from(&u_element),
//                                            &quadrature);
//
//     let func = create_single_element_elastic_force_vector_function(
//         &quad,
//         &quad_indices,
//         &material,
//         &quadrature,
//     );
//
//     let a_approx = -approximate_jacobian(func, &u, &h);
//
//     let diff = a - a_approx;
//     assert!(diff.norm() < 1e-5);
// }
//
// #[test]
// fn element_stiffness_matrix_is_negative_derivative_of_forces_for_stvk_material_problematic_element()
// {
//     // This is an example where the eigenvalues of the element matrices turned out to be
//     // strongly negative
//
//     let u = DVector::from_vec(vec![
//         0.0,
//         0.0,
//         -0.033613342105786675,
//         -0.21919651627727949,
//         -0.26977755029543005,
//         -0.19110852394892108,
//         0.0,
//         0.0,
//     ]);
//     let lame = YoungPoisson {
//         young: 1e8,
//         poisson: 0.2,
//     };
//     let material = StVKMaterial::from(lame);
//
//     let h = 1e-6;
//     let quadrature = quad_quadrature_strength_5_f64();
//     let quad = Quad4d2Element::from_vertices([
//         Point2::new(0.0, -1.0),
//         Point2::new(1.0, -1.0),
//         Point2::new(1.0, 0.0),
//         Point2::new(0.0, 0.0),
//     ]);
//
//     let quad_indices = Quad4d2Connectivity([0, 1, 2, 3]);
//
//     let u_element: OMatrix<_, U2, _> = quad_indices.element_variables(&u);
//     let elliptic_operator = MaterialEllipticOperator(&material);
//     let mut a = DMatrix::zeros(8, 8);
//     assemble_generalized_element_stiffness(DMatrixSliceMut::from(&mut a),
//                                            &quad,
//                                            &elliptic_operator,
//                                            MatrixSlice::from(&u_element),
//                                            &quadrature);
//
//     let func = create_single_element_elastic_force_vector_function(
//         &quad,
//         &quad_indices,
//         &material,
//         &quadrature,
//     );
//
//     let a_approx = -approximate_jacobian(func, &u, &h);
//
//     let diff = &a - &a_approx;
//
//     assert!(diff.norm() / (a.norm() + a_approx.norm()) < 1e-5);
//
//     // TODO: Report issue with this matrix to nalgebra as example of failing eigenvalue decomposition
// }

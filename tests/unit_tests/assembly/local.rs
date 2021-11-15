use fenris::allocators::{BiDimAllocator, SmallDimAllocator};
use fenris::assembly::global::{compute_global_potential, CsrAssembler, VectorAssembler};
use fenris::assembly::local::{
    assemble_element_mass_matrix, AggregateElementAssembler, ElementConnectivityAssembler,
    ElementEllipticAssemblerBuilder, UniformQuadratureTable,
};
use fenris::assembly::operators::LaplaceOperator;
use fenris::element::{Quad4d2Element, VolumetricFiniteElement};
use fenris::geometry::Quad2d;
use fenris::mesh::procedural::create_unit_square_uniform_quad_mesh_2d;
use fenris::mesh::QuadMesh2d;
use fenris::nalgebra::{DMatrix, DVector, DefaultAllocator, DimName, Matrix4, OPoint, OVector, Point2, RealField};
use fenris::quadrature;
use fenris::quadrature::QuadraturePair;
use itertools::izip;
use matrixcompare::{assert_matrix_eq, assert_scalar_eq};
use nalgebra::{DMatrixSliceMut, Matrix2};
use std::iter::repeat;

mod elliptic;
mod mass;
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

    let (weights, points) = quadrature::total_order::quadrilateral(5).unwrap();
    let densities: Vec<_> = repeat(density).take(weights.len()).collect();
    let quad = Quad4d2Element::from(reference_quad());

    let ndof = 8;
    let mut m = DMatrix::zeros(ndof, ndof);

    let mut buffer = vec![3.0; 4];
    assemble_element_mass_matrix(
        DMatrixSliceMut::from(&mut m),
        &quad,
        &weights,
        &points,
        &densities,
        2,
        &mut buffer,
    )
    .unwrap();

    #[rustfmt::skip]
    let expected4x4 = (density / 9.0) * Matrix4::<f64>::new(
        4.0, 2.0, 1.0, 2.0,
        2.0, 4.0, 2.0, 1.0,
        1.0, 2.0, 4.0, 2.0,
        2.0, 1.0, 2.0, 4.0);
    let expected8x8 = expected4x4.kronecker(&Matrix2::identity());
    assert_matrix_eq!(expected8x8, m, comp = float);
}

/// An artificial density function that we use to validate that quadrature parameters are correctly
/// employed in the assembly.
fn density<D>(x: &OPoint<f64, D>) -> f64
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
    vertices: &[OPoint<f64, D>],
    u_exact: impl Fn(&OPoint<f64, D>) -> OVector<f64, S>,
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
    points: &[OPoint<f64, Element::GeometryDim>],
    density: impl Fn(&OPoint<f64, Element::GeometryDim>) -> f64,
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

#[test]
fn element_connectivity_assembler_map_node() {
    struct MockElementConnectivityAssembler;

    impl ElementConnectivityAssembler for MockElementConnectivityAssembler {
        fn solution_dim(&self) -> usize {
            2
        }

        fn num_elements(&self) -> usize {
            3
        }

        fn num_nodes(&self) -> usize {
            6
        }

        fn element_node_count(&self, element_index: usize) -> usize {
            match element_index {
                0 => 3,
                1 => 5,
                2 => 4,
                _ => panic!(),
            }
        }

        fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
            let slice = match element_index {
                0 => &[0, 2, 4].as_ref(),
                1 => [1, 2, 3, 4, 5].as_ref(),
                2 => &[0, 1, 3, 5].as_ref(),
                _ => panic!(),
            };
            output.copy_from_slice(slice);
        }
    }

    let new_node_count = 11;
    let mapped_assembler = MockElementConnectivityAssembler.map_element_nodes(new_node_count, |node_idx| 2 * node_idx);
    assert_eq!(mapped_assembler.num_nodes(), new_node_count);

    let mut nodes = vec![0; 5];

    mapped_assembler.populate_element_nodes(&mut nodes[0..3], 0);
    assert_eq!(&nodes[0..3], &vec![0, 4, 8]);

    mapped_assembler.populate_element_nodes(&mut nodes[0..5], 1);
    assert_eq!(&nodes[0..5], &vec![2, 4, 6, 8, 10]);

    mapped_assembler.populate_element_nodes(&mut nodes[0..4], 2);
    assert_eq!(&nodes[0..4], &vec![0, 2, 6, 10]);
}

#[test]
fn aggregate_element_assembler_repeated_assembler() {
    let mesh: QuadMesh2d<f64> = create_unit_square_uniform_quad_mesh_2d(3);
    let qtable =
        UniformQuadratureTable::from_quadrature_and_uniform_data(quadrature::tensor::quadrilateral_gauss(1), ());
    let u = DVector::zeros(mesh.vertices().len());
    let assembler = ElementEllipticAssemblerBuilder::new()
        .with_operator(&LaplaceOperator)
        .with_finite_element_space(&mesh)
        .with_quadrature_table(&qtable)
        .with_u(&u)
        .build();
    let assemblers = vec![assembler.clone(), assembler.clone()];
    let aggregate = AggregateElementAssembler::from_assemblers(&assemblers);

    // Scalar
    let aggregate_scalar = compute_global_potential(&aggregate).unwrap();
    let expected_scalar = 2.0 * compute_global_potential(&assembler).unwrap();
    assert_scalar_eq!(aggregate_scalar, expected_scalar, comp = float);

    // Vector
    let aggregate_vector = VectorAssembler::default()
        .assemble_vector(&aggregate)
        .unwrap();
    let expected_vector = 2.0
        * VectorAssembler::default()
            .assemble_vector(&assembler)
            .unwrap();
    assert_matrix_eq!(aggregate_vector, expected_vector, comp = float);

    // Matrix
    let aggregate_matrix = CsrAssembler::default().assemble(&aggregate).unwrap();
    let expected_matrix = 2.0 * CsrAssembler::default().assemble(&assembler).unwrap();
    assert_matrix_eq!(aggregate_matrix, expected_matrix, comp = float);
}

#[test]
fn aggregate_element_assembler_multibody() {
    // "Simulate" a multibody setup where we want to assemble into different blocks, e.g.
    // [ K_1   0 ]
    // [  0   K2 ]
    let qtable =
        UniformQuadratureTable::from_quadrature_and_uniform_data(quadrature::tensor::quadrilateral_gauss(1), ());
    let mesh1: QuadMesh2d<f64> = create_unit_square_uniform_quad_mesh_2d(3);
    let mesh2: QuadMesh2d<f64> = create_unit_square_uniform_quad_mesh_2d(4);
    let u1 = DVector::zeros(mesh1.vertices().len());
    let u2 = DVector::zeros(mesh2.vertices().len());
    let num_global_nodes = mesh1.vertices().len() + mesh2.vertices().len();

    let build_assembler = |mesh, u| {
        ElementEllipticAssemblerBuilder::new()
            .with_operator(&LaplaceOperator)
            .with_finite_element_space(mesh)
            .with_quadrature_table(&qtable)
            .with_u(u)
            .build()
    };
    let build_assembler_with_offset = |mesh, u, offset| {
        build_assembler(mesh, u).map_element_nodes(num_global_nodes, move |node_idx: usize| node_idx + offset)
    };

    // build "local" and "globaL" (with offset into global node list) assemblers
    let assembler1 = build_assembler(&mesh1, &u1);
    let assembler1_global = build_assembler_with_offset(&mesh1, &u1, 0);
    let assembler2 = build_assembler(&mesh2, &u2);
    let assembler2_global = build_assembler_with_offset(&mesh2, &u2, mesh1.vertices().len());

    let assemblers = vec![assembler1_global, assembler2_global];
    let aggregate = AggregateElementAssembler::from_assemblers(&assemblers);

    // Scalar
    let aggregate_scalar = compute_global_potential(&aggregate).unwrap();
    let expected_scalar =
        compute_global_potential(&assembler1).unwrap() + compute_global_potential(&assembler2).unwrap();
    assert_scalar_eq!(aggregate_scalar, expected_scalar, comp = float);

    // Vector
    let vector_assembler = VectorAssembler::default();
    let aggregate_vector = vector_assembler.assemble_vector(&aggregate).unwrap();
    let expected_vector1 = vector_assembler.assemble_vector(&assembler1).unwrap();
    let expected_vector2 = vector_assembler.assemble_vector(&assembler2).unwrap();
    assert_eq!(aggregate_vector.len(), num_global_nodes);
    assert_matrix_eq!(aggregate_vector.index((0..expected_vector1.len(), 0)), expected_vector1);
    assert_matrix_eq!(
        aggregate_vector.index((expected_vector1.len()..num_global_nodes, 0)),
        expected_vector2
    );

    // Matrix
    // Here we convert to a dense matrix in order to be able to extract blocks
    // (TODO: need to implement something like this in nalgebra_sparse...)
    let matrix_assembler = CsrAssembler::default();
    let aggregate_matrix = DMatrix::from(&matrix_assembler.assemble(&aggregate).unwrap());
    let expected_matrix1 = DMatrix::from(&matrix_assembler.assemble(&assembler1).unwrap());
    let expected_matrix2 = DMatrix::from(&matrix_assembler.assemble(&assembler2).unwrap());

    let n1 = mesh1.vertices().len();
    let top_left_block = aggregate_matrix.index((0..n1, 0..n1));
    let top_right_block = aggregate_matrix.index((0..n1, n1..));
    let bottom_left_block = aggregate_matrix.index((n1.., 0..n1));
    let bottom_right_block = aggregate_matrix.index((n1.., n1..));

    assert_matrix_eq!(top_left_block, expected_matrix1);
    assert_matrix_eq!(bottom_right_block, expected_matrix2);

    assert!(top_right_block.iter().all(|&x_i| x_i == 0.0));
    assert!(bottom_left_block.iter().all(|&x_i| x_i == 0.0));
}

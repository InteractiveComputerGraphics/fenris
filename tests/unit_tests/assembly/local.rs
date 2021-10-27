use fenris::allocators::{BiDimAllocator, SmallDimAllocator};
use fenris::assembly::local::{assemble_element_mass_matrix, ElementConnectivityAssembler};
use fenris::element::{Quad4d2Element, VolumetricFiniteElement};
use fenris::geometry::Quad2d;
use fenris::nalgebra::{DMatrix, DVector, DefaultAllocator, DimName, Matrix4, OPoint, OVector, Point2, RealField};
use fenris::quadrature;
use fenris::quadrature::QuadraturePair;
use itertools::izip;
use matrixcompare::assert_matrix_eq;
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

    let mapped_assembler = MockElementConnectivityAssembler.map_element_nodes(|node_idx| 2 * node_idx);

    let mut nodes = vec![0; 5];

    mapped_assembler.populate_element_nodes(&mut nodes[0..3], 0);
    assert_eq!(&nodes[0..3], &vec![0, 4, 8]);

    mapped_assembler.populate_element_nodes(&mut nodes[0..5], 1);
    assert_eq!(&nodes[0..5], &vec![2, 4, 6, 8, 10]);

    mapped_assembler.populate_element_nodes(&mut nodes[0..4], 2);
    assert_eq!(&nodes[0..4], &vec![0, 2, 6, 10]);
}

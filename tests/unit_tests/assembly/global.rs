use std::cmp::min;

use proptest::collection::vec;
use proptest::num::i32;
use proptest::prelude::*;

use eyre::eyre;
use fenris::assembly::global::{
    apply_homogeneous_dirichlet_bc_csr, apply_homogeneous_dirichlet_bc_matrix, compute_global_potential,
    gather_global_to_local, par_compute_global_potential, CsrAssembler, CsrParAssembler,
};
use fenris::assembly::local::{ElementConnectivityAssembler, ElementScalarAssembler};
use fenris::nalgebra::{DMatrix, DVector, U2};
use fenris::nalgebra_sparse::pattern::SparsityPattern;
use fenris::nalgebra_sparse::CsrMatrix;
use matrixcompare::assert_scalar_eq;

#[test]
fn apply_homogeneous_dirichlet_bc_matrix_simple_example() {
    let mut matrix = DMatrix::repeat(8, 8, 2.0);
    apply_homogeneous_dirichlet_bc_matrix::<f64, U2>(&mut matrix, &[0, 2]);

    #[rustfmt::skip]
    let expected = DMatrix::from_column_slice(8, 8, &[
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0
    ]);

    let diff = &matrix - &expected;
    assert!(diff.norm() < 1e-12);

    // TODO: Design a more elaborate test that also checks for appropriate diagonal scaling
    // of the diagonal elements
}

#[test]
fn apply_homogeneous_dirichlet_bc_csr_simple_example() {
    let mut matrix = CsrMatrix::from(&DMatrix::repeat(8, 8, 2.0));

    apply_homogeneous_dirichlet_bc_csr(&mut matrix, &[0, 2], 2);

    // Note: We don't enforce exactly what values the matrix should take on
    // the diagonal entries, only that they are somewhere between the
    // smallest and largest diagonal entries
    #[rustfmt::skip]
        let expected = DMatrix::from_column_slice(8, 8, &[
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0,
        0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0
    ]);

    // TODO: Assert that the sparsity pattern is as expected
    assert_eq!(DMatrix::from(&matrix), expected)

    // TODO: Design a more elaborate test that also checks for appropriate diagonal scaling
    // of the diagonal elements
}

#[test]
fn csr_assemble_mock_pattern() {
    // Solution dim == 1

    // Empty pattern
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 1,
            num_nodes: 0,
            element_connectivities: vec![vec![]],
        };
        let csr_assembler = CsrAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(0, 0, vec![0], vec![]).unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // Empty pattern
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 2,
            num_nodes: 5,
            element_connectivities: vec![vec![]],
        };
        let csr_assembler = CsrAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(10, 10, vec![0; 11], vec![]).unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // Simple pattern, solution dim == 1
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 1,
            num_nodes: 6,
            element_connectivities: vec![vec![0, 1, 2], vec![2, 3], vec![], vec![3, 4, 4, 4, 4, 4, 4]],
        };
        let csr_assembler = CsrAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(
            6,
            6,
            vec![0, 3, 6, 10, 13, 15, 15],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 2, 3, 4, 3, 4],
        )
        .unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // Simple pattern, solution dim == 2
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 2,
            num_nodes: 6,
            element_connectivities: vec![vec![0, 1, 2], vec![2, 3], vec![], vec![3, 4, 4, 4, 4, 4, 4]],
        };
        let csr_assembler = CsrParAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(
            12,
            12,
            vec![0, 6, 12, 18, 24, 32, 40, 46, 52, 56, 60, 60, 60],
            vec![
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
                2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9,
            ],
        )
        .unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // TODO: Would be good to have some property tests...
}

#[test]
fn csr_par_assemble_mock_pattern() {
    // Solution dim == 1

    // Empty pattern
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 1,
            num_nodes: 0,
            element_connectivities: vec![vec![]],
        };
        let csr_assembler = CsrParAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(0, 0, vec![0], vec![]).unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // Empty pattern
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 2,
            num_nodes: 5,
            element_connectivities: vec![vec![]],
        };
        let csr_assembler = CsrParAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(10, 10, vec![0; 11], vec![]).unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // Simple pattern, solution dim == 1
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 1,
            num_nodes: 6,
            element_connectivities: vec![vec![0, 1, 2], vec![2, 3], vec![], vec![3, 4, 4, 4, 4, 4, 4]],
        };
        let csr_assembler = CsrParAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(
            6,
            6,
            vec![0, 3, 6, 10, 13, 15, 15],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 2, 3, 4, 3, 4],
        )
        .unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // Simple pattern, solution dim == 2
    {
        let element_assembler = MockElementAssembler {
            solution_dim: 2,
            num_nodes: 6,
            element_connectivities: vec![vec![0, 1, 2], vec![2, 3], vec![], vec![3, 4, 4, 4, 4, 4, 4]],
        };
        let csr_assembler = CsrParAssembler::<i32>::default();
        let pattern = csr_assembler.assemble_pattern(&element_assembler);
        let expected_pattern = SparsityPattern::try_from_offsets_and_indices(
            12,
            12,
            vec![0, 6, 12, 18, 24, 32, 40, 46, 52, 56, 60, 60, 60],
            vec![
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
                2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9,
            ],
        )
        .unwrap();
        assert_eq!(pattern, expected_pattern);
    }

    // TODO: Would be good to have some property tests...
}

fn gather_global_to_local_args() -> impl Strategy<Value = GatherGlobalToLocalArgs> {
    let sol_dim = 0..10usize;
    let num_nodes = 0..10usize;

    (sol_dim, num_nodes)
        .prop_flat_map(|(s, n)| {
            let u = vec(i32::ANY, s * n);
            // The first `min` is just a trick to prevent having an empty range
            // (in that case `v.len() == 0`) and we'll anyway get an empty vec
            // The second min is to ensure that we generate an empty vec if s == 0
            let indices = vec(0..min(1, n), min(s, n));
            (Just(s), u, indices)
        })
        .prop_map(|(sol_dim, u, indices)| GatherGlobalToLocalArgs {
            solution_dim: sol_dim,
            u: DVector::from(u),
            indices,
        })
}

struct MockElementAssembler {
    solution_dim: usize,
    num_nodes: usize,
    element_connectivities: Vec<Vec<usize>>,
}

impl ElementConnectivityAssembler for MockElementAssembler {
    fn solution_dim(&self) -> usize {
        self.solution_dim
    }

    fn num_elements(&self) -> usize {
        self.element_connectivities.len()
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.element_connectivities[element_index].len()
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        output.copy_from_slice(&self.element_connectivities[element_index])
    }
}

struct MockScalarElementAssembler;

#[rustfmt::skip]
impl ElementConnectivityAssembler for MockScalarElementAssembler {
    fn solution_dim(&self) -> usize { unreachable!() }
    fn num_elements(&self) -> usize { 4 }
    fn num_nodes(&self) -> usize { unreachable!() }
    fn element_node_count(&self, _element_index: usize) -> usize { unreachable!() }
    fn populate_element_nodes(&self, _output: &mut [usize], _element_index: usize) { unreachable!() }
}

#[rustfmt::skip]
impl ElementScalarAssembler<f64> for MockScalarElementAssembler {
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<f64> {
        match element_index {
            0 => Ok(3.0),
            1 => Ok(4.0),
            2 => Ok(5.0),
            3 => Ok(-3.0),
            _ => Err(eyre!("Element out of bounds"))
        }
    }
}

#[test]
fn test_compute_global_potential() {
    let global_potential = compute_global_potential(&MockScalarElementAssembler).unwrap();
    assert_scalar_eq!(global_potential, 9.0, comp = float);
}

#[test]
fn test_par_compute_global_potential() {
    let par_global_potential = par_compute_global_potential(&MockScalarElementAssembler).unwrap();
    assert_scalar_eq!(par_global_potential, 9.0, comp = float);
}

#[derive(Debug)]
struct GatherGlobalToLocalArgs {
    solution_dim: usize,
    u: DVector<i32>,
    indices: Vec<usize>,
}

// TODO: Test scatter_local_to_global

proptest! {
    #[test]
    fn gather_global_to_local_test(args in gather_global_to_local_args()) {
        let mut local = DVector::zeros(args.indices.len() * args.solution_dim);
        gather_global_to_local(&args.u, &mut local, &args.indices, args.solution_dim);

        let s = args.solution_dim;
        let n = args.indices.len();

        let mut all_correct = true;
        for i in 0 .. n {
            for j in 0 .. s {
                if local[s * i + j] != args.u[args.indices[i] + j] {
                    all_correct = false;
                }
            }
        }

        prop_assert!(all_correct);
    }
}

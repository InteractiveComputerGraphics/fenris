use fenris::mesh::reorder::{cuthill_mckee, reverse_cuthill_mckee};
use fenris::nalgebra_sparse::CsrMatrix;
use nalgebra::DMatrix;

#[test]
fn cuthill_mckee_basic_examples() {
    // Basic example
    {
        let matrix = DMatrix::from_row_slice(4, 4, &[1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]);
        let csr = CsrMatrix::from(&matrix);
        let perm = cuthill_mckee(csr.pattern());

        assert_eq!(perm.perm(), &[1, 3, 0, 2]);

        let mut rcm_expected_perm = perm.clone();
        rcm_expected_perm.reverse();
        assert_eq!(&reverse_cuthill_mckee(csr.pattern()), &rcm_expected_perm);
    }

    // Diagonal pattern
    // Note that the "standard" CM algorithm
    {
        let matrix = DMatrix::from_row_slice(4, 4, &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        let csr = CsrMatrix::from(&matrix);
        let perm = cuthill_mckee(csr.pattern());
        assert_eq!(perm.perm(), &[0, 1, 2, 3]);
    }

    // TODO: Property-based tests
}

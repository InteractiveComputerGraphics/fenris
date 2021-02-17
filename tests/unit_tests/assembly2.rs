pub use fenris::assembly2::gather_global_to_local;
use fenris::nalgebra::{min, DVector};
use proptest::collection::vec;
use proptest::num::i32;
pub use proptest::prelude::*;

#[derive(Debug)]
struct GatherGlobalToLocalArgs {
    solution_dim: usize,
    u: DVector<i32>,
    indices: Vec<usize>,
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

#!/bin/sh
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export TSAN_OPTIONS="suppressions=$SCRIPTPATH/tsan_suppression.txt"
export RUST_TEST_THREADS=1
export CARGO_INCREMENTAL=0
export RUSTFLAGS="-Z sanitizer=thread"
export RAYON_NUM_THREADS=4
cargo +nightly test -p paradis --target x86_64-unknown-linux-gnu --tests

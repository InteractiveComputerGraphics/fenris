use std::path::{PathBuf};

// TODO: Rewrite the assembly tests once we have fenris-solid up and running again
// mod assembly;
mod interpolation;
mod geometry;

fn data_output_path() -> PathBuf {
    PathBuf::from("data/integration_tests/")
}

use std::path::PathBuf;

// TODO: Rewrite the assembly tests once we have fenris-solid up and running again
// mod assembly;
mod geometry;
mod interpolation;

fn data_output_path() -> PathBuf {
    PathBuf::from("data/integration_tests/")
}

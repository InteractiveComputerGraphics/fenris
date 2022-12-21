//! Tests for estimation of L2/H1 seminorm errors approximated by using a higher-resolution
//! reference solution.
use std::f64::consts::PI;
use std::fs::{create_dir_all, File};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use itertools::izip;
use nalgebra::coordinates::XY;
use nalgebra::{Point2, U1, vector, Vector1, Vector2};
use serde::{Serialize, Deserialize};
use fenris::assembly::local::UniformQuadratureTable;
use fenris::error::{estimate_H1_seminorm_error, estimate_L2_error, SpaceInterpolationFn};
use fenris::mesh::procedural::create_unit_square_uniform_tri_mesh_2d;
use fenris::quadrature;
use fenris::space::SpatiallyIndexed;
use fenris::util::global_vector_from_point_fn;
use fenris_quadrature::polyquad::triangle;

fn sin(x: f64) -> f64 {
    x.sin()
}
fn cos(x: f64) -> f64 {
    x.cos()
}

fn u_2d(x: &Point2<f64>) -> Vector1<f64> {
    let &XY { x, y } = x.coords.deref();
    vector![sin(PI * x) * sin(PI * y)]
}

fn u_grad_2d(x: &Point2<f64>) -> Vector2<f64> {
    let &XY { x, y } = x.coords.deref();
    let u_x = PI * cos(PI * x) * sin(PI * y);
    let u_y = PI * sin(PI * x) * cos(PI * y);
    Vector2::new(u_x, u_y)
}

#[derive(Debug, Default, Clone, PartialEq)]
#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct ErrorSample {
    pub coarse_res: usize,
    /// The resolution of the fine mesh used for error estimation, or `0` if analytic
    pub fine_res: usize,
    pub L2_error: f64,
    pub H1_semi_error: f64,
}

/// For serializing to JSON for subsequent analysis/plots
#[derive(Debug, Default, PartialEq)]
#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct ErrorSummary {
    pub element_name: String,
    pub samples: Vec<ErrorSample>,
}

fn export_summary(summary: &ErrorSummary) {
    let base_path = PathBuf::from("data/convergence_tests/error_estimation/");
    let path = base_path.join(format!("{}_summary.json", summary.element_name.to_ascii_lowercase()));
    create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = File::create(path).unwrap();
    serde_json::to_writer_pretty(&mut file, summary).unwrap();
}

fn load_reference_summary(element_name: &str) -> ErrorSummary {
    let path = format!("tests/convergence_tests/reference_values/error_estimation_{}_summary.json",
                       element_name.to_ascii_lowercase());
    let json_str = std::fs::read_to_string(&path)
        .expect(&format!("I/O error: Failed to load reference summary for \
                          element {} at location \"{}\"", element_name, &path));
    match serde_json::from_str(&json_str) {
        Ok(summary) => summary,
        Err(error) => panic!("Failed to deserialize reference summary for element {}. Error: {}",
                             element_name, error)
    }
}

fn rel_error(value: f64, reference: f64) -> f64 {
    (value - reference).abs() / reference.abs()
}

#[allow(non_snake_case)]
fn assert_summary_close_to_reference(summary: &ErrorSummary) {
    let reference = load_reference_summary(&summary.element_name);

    assert_eq!(summary.element_name, reference.element_name, "Element name mismatch");
    assert_eq!(summary.samples.len(), reference.samples.len(), "Sample count mismatch");

    for (summary_sample, reference_sample) in izip!(&summary.samples, &reference.samples) {
        assert_eq!(summary_sample.coarse_res, reference_sample.coarse_res,
            "Coarse resolution mismatch");
        assert_eq!(summary_sample.fine_res, reference_sample.fine_res,
            "Fine resolution mismatch");

        let L2_rel_error = rel_error(summary_sample.L2_error, reference_sample.L2_error);
        let H1_semi_rel_error = rel_error(summary_sample.H1_semi_error, reference_sample.H1_semi_error);
        assert!(L2_rel_error <= 0.01,
            "L2 error deviates by more than 1% (summary: {}, reference: {}, rel error: {}",
            summary_sample.L2_error, reference_sample.L2_error, L2_rel_error);
        assert!(H1_semi_rel_error <= 0.01,
            "H1 seminorm error deviates by more than 1% (summary: {}, reference: {}, rel error: {}",
            summary_sample.H1_semi_error, reference_sample.H1_semi_error, H1_semi_rel_error);
    }
}

#[allow(non_snake_case)]
#[test]
fn tri3_error_estimation() {
    let fine_resolutions = vec![64, 71, 91, 128, 131, 512, 1024];
    let coarse_resolutions = vec![1, 2, 3, 4, 8, 16, 32];

    let base_error_quadrature = quadrature::total_order::triangle(20).unwrap();
    let qtable = UniformQuadratureTable::from_quadrature(base_error_quadrature);

    let mut summary = ErrorSummary::default();
    summary.element_name = "Tri3".into();

    for &fine_res in &fine_resolutions {
        let mesh_fine = create_unit_square_uniform_tri_mesh_2d(fine_res);
        let u_weights_fine = global_vector_from_point_fn(mesh_fine.vertices(), u_2d);
        let fine_interpolator = SpatiallyIndexed::from_space(mesh_fine);
        let u_fine = SpaceInterpolationFn(&fine_interpolator, &u_weights_fine);
        for &coarse_res in &coarse_resolutions {
            let mesh_coarse = create_unit_square_uniform_tri_mesh_2d(coarse_res);
            let u_weights_coarse = global_vector_from_point_fn(mesh_coarse.vertices(), u_2d);
            // TODO: Can we make SolutionDim unambiguous, so that we don't have to specify generic types?
            // I tried to make SolutionDim an associated type of SolutionFunction/Gradient,
            // but unfortunately then we run into some compiler bugs where rustc gets
            // the types wrong :-(
            let L2_error_approx = estimate_L2_error::<_, U1, _, _>(&mesh_coarse, &u_fine, &u_weights_coarse, &qtable).unwrap();
            let H1_semi_error_approx = estimate_H1_seminorm_error::<_, U1, _, _>(&mesh_coarse, &u_fine, &u_weights_coarse, &qtable).unwrap();
            let sample = ErrorSample {
                coarse_res,
                fine_res,
                L2_error: L2_error_approx,
                H1_semi_error: H1_semi_error_approx,
            };

            summary.samples.push(sample);
        }
    }

    for &coarse_res in &coarse_resolutions {
        let mesh_coarse = create_unit_square_uniform_tri_mesh_2d(coarse_res);
        let u_weights_coarse = global_vector_from_point_fn(mesh_coarse.vertices(), u_2d);
        let L2_error = estimate_L2_error::<_, U1, _, _>(&mesh_coarse, &u_2d, &u_weights_coarse, &qtable).unwrap();
        let H1_semi_error = estimate_H1_seminorm_error::<_, U1, _, _>(&mesh_coarse, &u_grad_2d, &u_weights_coarse, &qtable).unwrap();
        let sample = ErrorSample {
            coarse_res,
            fine_res: 0,
            L2_error,
            H1_semi_error,
        };
        summary.samples.push(sample);
    }

    export_summary(&summary);
    assert_summary_close_to_reference(&summary);
}
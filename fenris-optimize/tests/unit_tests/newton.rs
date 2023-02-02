use fenris_optimize::calculus::{DifferentiableVectorFunction, VectorFunction};
use fenris_optimize::newton::*;
use nalgebra::{DVector, DVectorView, DVectorViewMut, Matrix3, Vector3};
use numeric_literals::replace_numeric_literals;
use std::error::Error;

struct MockLinearVectorFunction;

impl VectorFunction<f64> for MockLinearVectorFunction {
    fn dimension(&self) -> usize {
        3
    }

    #[replace_numeric_literals(f64::from(literal))]
    fn eval_into(&mut self, f: &mut DVectorViewMut<f64>, x: &DVectorView<f64>) {
        let a = Matrix3::new(5, 1, 2, 1, 4, 2, 2, 2, 4);
        let b = Vector3::new(1, 2, 3);
        let r = a * x - b;
        f.copy_from(&r);
    }
}

impl DifferentiableVectorFunction<f64> for MockLinearVectorFunction {
    #[replace_numeric_literals(f64::from(literal))]
    fn solve_jacobian_system(
        &mut self,
        sol: &mut DVectorViewMut<f64>,
        _x: &DVectorView<f64>,
        rhs: &DVectorView<f64>,
    ) -> Result<(), Box<dyn Error>> {
        let a = Matrix3::new(5, 1, 2, 1, 4, 2, 2, 2, 4);
        let a_inv = a.try_inverse().unwrap();
        sol.copy_from(&(a_inv * rhs));
        Ok(())
    }
}

#[test]
fn newton_converges_in_single_iteration_for_linear_system() {
    // TODO: Use VectorFunctionBuilder or a mock from mockiato

    let expected_solution = Vector3::new(-0.125, 0.16666667, 0.72916667);

    let settings = NewtonSettings {
        max_iterations: Some(2),
        tolerance: Vector3::new(1.0, 2.0, 3.0).norm() * 1e-6,
    };

    let mut f = DVector::zeros(3);
    let mut x = DVector::zeros(3);
    let mut dx = DVector::zeros(3);

    let iterations =
        newton(MockLinearVectorFunction, &mut x, &mut f, &mut dx, settings).expect("Newton iterations must succeed");
    let diff = x - expected_solution;
    assert!(diff.norm() < 1e-6);
    assert_eq!(iterations, 1);
}

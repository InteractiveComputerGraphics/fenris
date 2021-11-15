//! Use method of manufactured solutions on a 2D Poisson problem to verify convergence.
//!
//! The problem is:
//!   - Delta u = f,
//! where Delta = nabla^2 is the Laplace operator.
use fenris::assembly::local::SourceFunction;
use fenris::assembly::operators::Operator;
use fenris::element::ElementConnectivity;
use fenris::io::vtk::VtkCellConnectivity;
use fenris::mesh::procedural::{create_unit_square_uniform_quad_mesh_2d, create_unit_square_uniform_tri_mesh_2d};
use fenris::mesh::{Mesh2d, Quad9Mesh2d, Tri6Mesh2d};
use fenris::nalgebra::coordinates::XY;
use fenris::nalgebra::{OPoint, OVector, Point2, Vector1, Vector2, U1, U2};
use fenris::quadrature;
use fenris::quadrature::QuadraturePair2d;
use std::f64::consts::PI;
use std::ops::Deref;

fn sin(x: f64) -> f64 {
    x.sin()
}
fn cos(x: f64) -> f64 {
    x.cos()
}

// Exact solution
fn u_exact(x: &Point2<f64>) -> f64 {
    let &XY { x, y } = x.coords.deref();
    sin(PI * x) * sin(PI * y)
}

fn u_exact_grad(x: &Point2<f64>) -> Vector2<f64> {
    let &XY { x, y } = x.coords.deref();
    let u_x = PI * cos(PI * x) * sin(PI * y);
    let u_y = PI * sin(PI * x) * cos(PI * y);
    Vector2::new(u_x, u_y)
}

fn f(x: &Point2<f64>) -> f64 {
    // Derived from f = - Del u = - u_xx - u_yy
    2.0 * PI * PI * u_exact(x)
}

#[derive(Debug)]
pub struct PoissonProblemSourceFunction;

impl<T> Operator<T, U2> for PoissonProblemSourceFunction {
    type SolutionDim = U1;
    type Parameters = ();
}

impl SourceFunction<f64, U2> for PoissonProblemSourceFunction {
    fn evaluate(&self, coords: &OPoint<f64, U2>, _data: &Self::Parameters) -> OVector<f64, Self::SolutionDim> {
        Vector1::new(f(coords))
    }
}

pub fn solve_and_produce_output<C>(
    element_name: &str,
    resolutions: &[usize],
    // Produce a mesh for the given resolution
    mesh_producer: impl Fn(usize) -> Mesh2d<f64, C>,
    quadrature: QuadraturePair2d<f64>,
    error_quadrature: QuadraturePair2d<f64>,
) where
    C: VtkCellConnectivity + ElementConnectivity<f64, GeometryDim = U2, ReferenceDim = U2> + Sync,
{
    crate::convergence_tests::poisson_mms_common::solve_and_produce_output(
        element_name,
        resolutions,
        mesh_producer,
        quadrature,
        error_quadrature,
        &PoissonProblemSourceFunction,
        u_exact,
        &u_exact_grad,
    );
}

#[test]
fn poisson_2d_quad4() {
    let resolutions = [1, 2, 4, 8, 16, 32];
    let mesh_producer = |res| create_unit_square_uniform_quad_mesh_2d(res);
    let quadrature = quadrature::tensor::quadrilateral_gauss(2);
    let error_quadrature = quadrature::tensor::quadrilateral_gauss(6);
    solve_and_produce_output("Quad4", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_2d_quad8() {
    let resolutions = [1, 2, 4, 8, 16, 32];
    let mesh_producer = |res| Quad9Mesh2d::from(create_unit_square_uniform_quad_mesh_2d(res));
    let quadrature = quadrature::tensor::quadrilateral_gauss(2);
    let error_quadrature = quadrature::tensor::quadrilateral_gauss(6);
    solve_and_produce_output("Quad9", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_2d_tri3() {
    let resolutions = [1, 2, 4, 8, 16, 32];
    let mesh_producer = |res| create_unit_square_uniform_tri_mesh_2d(res);
    let quadrature = quadrature::total_order::triangle(0).unwrap();
    let error_quadrature = quadrature::total_order::triangle(6).unwrap();
    solve_and_produce_output("Tri3", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_2d_tri6() {
    let resolutions = [1, 2, 4, 8, 16, 32];
    let mesh_producer = |res| Tri6Mesh2d::from(create_unit_square_uniform_tri_mesh_2d(res));
    let quadrature = quadrature::total_order::triangle(2).unwrap();
    let error_quadrature = quadrature::total_order::triangle(6).unwrap();
    solve_and_produce_output("Tri6", &resolutions, mesh_producer, quadrature, error_quadrature);
}

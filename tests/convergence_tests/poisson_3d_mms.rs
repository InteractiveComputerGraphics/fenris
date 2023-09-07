//! Use method of manufactured solutions on a 3D Poisson problem to verify convergence.
//!
//! The problem is:
//!   - Delta u = f,
//! where Delta = nabla^2 is the Laplace operator.
use fenris::assembly::local::SourceFunction;
use fenris::assembly::operators::Operator;
use fenris::element::ElementConnectivity;
use fenris::io::vtk::VtkCellConnectivity;
use fenris::mesh::procedural::{create_unit_box_uniform_hex_mesh_3d, create_unit_box_uniform_tet_mesh_3d};
use fenris::mesh::{Hex20Mesh, Hex27Mesh, Mesh3d, Tet10Mesh, Tet20Mesh};
use fenris::nalgebra::coordinates::XYZ;
use fenris::nalgebra::{OPoint, OVector, Point3, Vector1, Vector3, U1, U3};
use fenris::quadrature;
use fenris::quadrature::QuadraturePair3d;
use std::f64::consts::PI;
use std::ops::Deref;

fn sin(x: f64) -> f64 {
    x.sin()
}
fn cos(x: f64) -> f64 {
    x.cos()
}

// Exact solution
fn u_exact(x: &Point3<f64>) -> f64 {
    let &XYZ { x, y, z } = x.coords.deref();
    sin(PI * x) * sin(PI * y) * sin(PI * z)
}

fn u_exact_grad(x: &Point3<f64>) -> Vector3<f64> {
    let &XYZ { x, y, z } = x.coords.deref();
    let u_x = PI * cos(PI * x) * sin(PI * y) * sin(PI * z);
    let u_y = PI * sin(PI * x) * cos(PI * y) * sin(PI * z);
    let u_z = PI * sin(PI * x) * sin(PI * y) * cos(PI * z);
    Vector3::new(u_x, u_y, u_z)
}

fn f(x: &Point3<f64>) -> f64 {
    // Derived from f = - Del u = - u_xx - u_yy
    3.0 * PI * PI * u_exact(x)
}

#[derive(Debug)]
pub struct PoissonProblemSourceFunction;

impl Operator<f64, U3> for PoissonProblemSourceFunction {
    type SolutionDim = U1;
    type Parameters = ();
}

impl SourceFunction<f64, U3> for PoissonProblemSourceFunction {
    fn evaluate(&self, coords: &OPoint<f64, U3>, _data: &Self::Parameters) -> OVector<f64, Self::SolutionDim> {
        Vector1::new(f(coords))
    }
}

pub fn solve_and_produce_output<C>(
    element_name: &str,
    resolutions: &[usize],
    // Produce a mesh for the given resolution
    mesh_producer: impl Fn(usize) -> Mesh3d<f64, C>,
    quadrature: QuadraturePair3d<f64>,
    error_quadrature: QuadraturePair3d<f64>,
) where
    C: VtkCellConnectivity + ElementConnectivity<f64, GeometryDim = U3, ReferenceDim = U3> + Sync,
{
    crate::convergence_tests::poisson_mms_common::solve_and_produce_output(
        element_name,
        resolutions,
        mesh_producer,
        quadrature,
        error_quadrature,
        &PoissonProblemSourceFunction,
        u_exact,
        u_exact_grad,
    );
}

#[test]
fn poisson_3d_hex8() {
    let resolutions = [1, 2, 4, 8, 16, 32];
    let mesh_producer = |res| create_unit_box_uniform_hex_mesh_3d(res);
    let quadrature = quadrature::tensor::hexahedron_gauss(2);
    let error_quadrature = quadrature::tensor::hexahedron_gauss(6);
    solve_and_produce_output("Hex8", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_3d_hex20() {
    let resolutions = [1, 2, 4, 8, 16];
    let mesh_producer = |res| Hex20Mesh::from(&create_unit_box_uniform_hex_mesh_3d(res));
    // TODO: Use "correct" quadrature
    let quadrature = quadrature::tensor::hexahedron_gauss(4);
    let error_quadrature = quadrature::tensor::hexahedron_gauss(6);
    solve_and_produce_output("Hex20", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_3d_hex27() {
    let resolutions = [1, 2, 4, 8, 16];
    let mesh_producer = |res| Hex27Mesh::from(&create_unit_box_uniform_hex_mesh_3d(res));
    // TODO: Use "correct" quadrature
    let quadrature = quadrature::tensor::hexahedron_gauss(4);
    let error_quadrature = quadrature::tensor::hexahedron_gauss(6);
    solve_and_produce_output("Hex27", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_3d_tet4() {
    let resolutions = [1, 2, 4, 8, 16];
    let mesh_producer = |res| create_unit_box_uniform_tet_mesh_3d(res);
    // TODO: Use "correct" quadrature
    let quadrature = quadrature::total_order::tetrahedron(0).unwrap();
    let error_quadrature = quadrature::total_order::tetrahedron(6).unwrap();
    solve_and_produce_output("Tet4", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_3d_tet10() {
    let resolutions = [1, 2, 4, 8, 12];
    let mesh_producer = |res| Tet10Mesh::from(&create_unit_box_uniform_tet_mesh_3d(res));
    // TODO: Use "correct" quadrature
    let quadrature = quadrature::total_order::tetrahedron(2).unwrap();
    let error_quadrature = quadrature::total_order::tetrahedron(6).unwrap();
    solve_and_produce_output("Tet10", &resolutions, mesh_producer, quadrature, error_quadrature);
}

#[test]
fn poisson_3d_tet20() {
    let resolutions = [1, 2, 4, 6, 8, 12];
    let mesh_producer = |res| Tet20Mesh::from(&create_unit_box_uniform_tet_mesh_3d(res));
    // TODO: Use "correct" quadrature
    let quadrature = quadrature::total_order::tetrahedron(4).unwrap();
    let error_quadrature = quadrature::total_order::tetrahedron(6).unwrap();
    solve_and_produce_output("Tet20", &resolutions, mesh_producer, quadrature, error_quadrature);
}

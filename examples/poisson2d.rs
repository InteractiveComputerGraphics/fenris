use fenris::mesh::{QuadMesh2d};
use fenris::procedural::create_unit_square_uniform_quad_mesh_2d;
use fenris::assembly2::{EllipticOperator, Operator, EllipticContraction, ElementEllipticAssembler, UniformQuadratureTable};
use fenris::nalgebra::{U2, MatrixMN, VectorN, U1, DVectorSlice, DVector};
use nalgebra::{Vector1, Point};
use fenris::assembly::global::{CsrAssembler, CsrParAssembler};
use fenris::quadrature::quad_quadrature_strength_5_f64;
use fenris_sparse::CsrMatrix;
use std::sync::Arc;
use std::error::Error;

pub struct PoissonOperator2d;

impl Operator<f64> for PoissonOperator2d {
    type SolutionDim = U1;
    type Data = ();
}

impl EllipticOperator<f64, U2> for PoissonOperator2d {
    fn compute_elliptic_term(&self,
                             gradient: &MatrixMN<f64, U2, Self::SolutionDim>,
                             _data: &Self::Data) -> MatrixMN<f64, U2, Self::SolutionDim> {
        *gradient
    }
}

impl EllipticContraction<f64, U2> for PoissonOperator2d {
    fn contract(&self,
                _gradient: &MatrixMN<f64, U2, Self::SolutionDim>,
                _data: &Self::Data,
                a: &VectorN<f64, U2>,
                b: &VectorN<f64, U2>) -> MatrixMN<f64, Self::SolutionDim, Self::SolutionDim> {
        Vector1::new(a.dot(&b))
    }
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // TODO: Make it easy to construct triangle meshes as well.
    // Need to make it easy to convert between different meshes, such as Quad2d -> Tri2d
    let mesh: QuadMesh2d<f64> = create_unit_square_uniform_quad_mesh_2d(3);
    let op = PoissonOperator2d;

    let (weights, points) = quad_quadrature_strength_5_f64();
    // TODO: Use Point for quadratures
    let points = points.into_iter().map(Point::from).collect();
    let quadrature = UniformQuadratureTable::from_points_and_weights(
        points,
        weights
    );

    let u = DVector::<f64>::zeros(mesh.vertices().len());

    // TODO: Build API or something for the elliptic assembler?
    let element_assembler = ElementEllipticAssembler {
        space: &mesh,
        op: &op,
        qtable: &quadrature,
        u: DVectorSlice::from(&u)
    };

    // TODO: CsrAssembler is not able to assemble patterns atm. So we use par assembler for the
    // pattern
    let matrix_assembler = CsrParAssembler::<f64>::default();
    let pattern = matrix_assembler.assemble_pattern(&element_assembler);
    let mut matrix_assembler = CsrAssembler::default();
    let nnz = pattern.nnz();
    let mut a = CsrMatrix::from_pattern_and_values(Arc::new(pattern), vec![0.0; nnz]);
    // TODO: Need a method that does the whole business of assembling a matrix without
    // needing storage first
    // TODO: Doesn't need to be mutable, does it?
    matrix_assembler.assemble_into_csr(&mut a, &element_assembler)?;

    println!("{}", a.build_dense());

    // TODO: Need vector assembler for source term

    Ok(())

    // let local_assembler = ElementEllipticAssembler
}
use std::convert::TryInto;

use itertools::Itertools;
use numeric_literals::replace_float_literals;

use crate::connectivity::{Hex20Connectivity, Hex27Connectivity, Hex8Connectivity};
use crate::element;
use crate::element::{ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement};
use crate::nalgebra::{distance, Matrix3, OMatrix, OPoint, Point3, Scalar, Vector3, U1, U20, U27, U3, U8};
use crate::Real;

impl<T> ElementConnectivity<T> for Hex8Connectivity
where
    T: Real,
{
    type Element = Hex8Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[OPoint<T, Self::GeometryDim>]) -> Option<Self::Element> {
        Some(Hex8Element::from_vertices([
            vertices.get(self.0[0])?.clone(),
            vertices.get(self.0[1])?.clone(),
            vertices.get(self.0[2])?.clone(),
            vertices.get(self.0[3])?.clone(),
            vertices.get(self.0[4])?.clone(),
            vertices.get(self.0[5])?.clone(),
            vertices.get(self.0[6])?.clone(),
            vertices.get(self.0[7])?.clone(),
        ]))
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Hex8Element<T>
where
    T: Real,
{
    type ReferenceDim = U3;
    type NodalDim = U8;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point3<T>) -> OMatrix<T, U1, U8> {
        // We define the shape functions as N_{alpha, beta, gamma} evaluated at xi such that
        //  N_{alpha, beta, gamma}([alpha, beta, gamma]) = 1,
        let phi_1d = element::phi_linear_1d;
        let phi = |alpha, beta, gamma, xi: &Point3<T>|
            phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]);
        OMatrix::<_, U1, U8>::from_row_slice(&[
            phi(-1.0, -1.0, -1.0, xi),
            phi( 1.0, -1.0, -1.0, xi),
            phi( 1.0,  1.0, -1.0, xi),
            phi(-1.0,  1.0, -1.0, xi),
            phi(-1.0, -1.0,  1.0, xi),
            phi( 1.0, -1.0,  1.0, xi),
            phi( 1.0,  1.0,  1.0, xi),
            phi(-1.0,  1.0,  1.0, xi),
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point3<T>) -> OMatrix<T, U3, U8> {
        let phi_1d = element::phi_linear_1d;
        let grad_1d = element::phi_linear_1d_grad;
        let phi_grad = |alpha, beta, gamma, xi: &Point3<T>|
            Vector3::new(
                grad_1d(alpha) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * grad_1d(beta) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * grad_1d(gamma)
            );

        OMatrix::from_columns(&[
            phi_grad(-1.0, -1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, -1.0, xi),
            phi_grad(-1.0,  1.0, -1.0, xi),
            phi_grad(-1.0, -1.0,  1.0, xi),
            phi_grad( 1.0, -1.0,  1.0, xi),
            phi_grad( 1.0,  1.0,  1.0, xi),
            phi_grad(-1.0,  1.0,  1.0, xi),
        ])
    }
}

impl<T> FiniteElement<T> for Hex8Element<T>
where
    T: Real,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        // TODO: Store this X matrix directly in Self...?
        let X = OMatrix::<_, U3, U8>::from_fn(|i, j| self.vertices[j][i]);
        let N = self.evaluate_basis(xi);
        OPoint::from(&X * &N.transpose())
    }

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        // TODO: Could store this matrix directly in the element, in order
        // to avoid repeated computation
        let X = OMatrix::<_, U3, U8>::from_fn(|i, j| self.vertices[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.vertices
            .iter()
            .tuple_combinations()
            .map(|(x, y)| distance(x, y))
            .fold(T::zero(), |a, b| a.max(b.clone()))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Hex8Element<T: Scalar> {
    vertices: [Point3<T>; 8],
}

impl<T> Hex8Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 8]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point3<T>; 8] {
        &self.vertices
    }
}

impl<T> Hex8Element<T>
where
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
        ])
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Hex27Element<T: Scalar> {
    // Store a hex8 element for trilinear transformations from reference element
    hex8: Hex8Element<T>,
    vertices: [Point3<T>; 27],
}

impl<T: Scalar + Copy> Hex27Element<T> {
    pub fn from_vertices(vertices: [Point3<T>; 27]) -> Self {
        Self {
            hex8: Hex8Element::from_vertices(vertices[0..8].try_into().unwrap()),
            vertices,
        }
    }

    pub fn vertices(&self) -> &[Point3<T>] {
        &self.vertices
    }
}

impl<T: Real> Hex27Element<T> {
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
            // Edge nodes
            Point3::new(0.0, -1.0, -1.0),
            Point3::new(-1.0, 0.0, -1.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, -1.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, -1.0, 1.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            // Face nodes
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            // Center node
            Point3::new(0.0, 0.0, 0.0),
        ])
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Hex27Element<T>
where
    T: Real,
{
    type ReferenceDim = U3;
    type NodalDim = U27;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point3<T>) -> OMatrix<T, U1, U27> {
        // We define the shape functions as N_{alpha, beta, gamma} evaluated at xi such that
        //  N_{alpha, beta, gamma}([alpha, beta, gamma]) = 1,
        let phi_1d = element::phi_quadratic_1d;
        let phi = |alpha, beta, gamma, xi: &Point3<T>|
            phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]);
        OMatrix::<_, U1, U27>::from_row_slice(&[
            // Vertex nodes
            phi(-1.0, -1.0, -1.0, xi),
            phi( 1.0, -1.0, -1.0, xi),
            phi( 1.0,  1.0, -1.0, xi),
            phi(-1.0,  1.0, -1.0, xi),
            phi(-1.0, -1.0,  1.0, xi),
            phi( 1.0, -1.0,  1.0, xi),
            phi( 1.0,  1.0,  1.0, xi),
            phi(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi(0.0, -1.0, -1.0, xi),
            phi(-1.0, 0.0, -1.0, xi),
            phi(-1.0, -1.0, 0.0, xi),
            phi(1.0, 0.0, -1.0, xi),
            phi(1.0, -1.0, 0.0, xi),
            phi(0.0, 1.0, -1.0, xi),
            phi(1.0, 1.0, 0.0, xi),
            phi(-1.0, 1.0, 0.0, xi),
            phi(0.0, -1.0, 1.0, xi),
            phi(-1.0, 0.0, 1.0, xi),
            phi(1.0, 0.0, 1.0, xi),
            phi(0.0, 1.0, 1.0, xi),

            // Face nodes
            phi(0.0, 0.0, -1.0, xi),
            phi(0.0, -1.0, 0.0, xi),
            phi(-1.0, 0.0, 0.0, xi),
            phi(1.0, 0.0, 0.0, xi),
            phi(0.0, 1.0, 0.0, xi),
            phi(0.0, 0.0, 1.0, xi),

            // Center node
            phi(0.0, 0.0, 0.0, xi)
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point3<T>) -> OMatrix<T, U3, U27> {
        let phi_1d = element::phi_quadratic_1d;
        let grad_1d = element::phi_quadratic_1d_grad;
        let phi_grad = |alpha, beta, gamma, xi: &Point3<T>|
            Vector3::new(
                grad_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * grad_1d(beta, xi[1]) * phi_1d(gamma, xi[2]),
                phi_1d(alpha, xi[0]) * phi_1d(beta, xi[1]) * grad_1d(gamma, xi[2])
            );

        OMatrix::from_columns(&[
            // Vertex nodes
            phi_grad(-1.0, -1.0, -1.0, xi),
            phi_grad( 1.0, -1.0, -1.0, xi),
            phi_grad( 1.0,  1.0, -1.0, xi),
            phi_grad(-1.0,  1.0, -1.0, xi),
            phi_grad(-1.0, -1.0,  1.0, xi),
            phi_grad( 1.0, -1.0,  1.0, xi),
            phi_grad( 1.0,  1.0,  1.0, xi),
            phi_grad(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi_grad(0.0, -1.0, -1.0, xi),
            phi_grad(-1.0, 0.0, -1.0, xi),
            phi_grad(-1.0, -1.0, 0.0, xi),
            phi_grad(1.0, 0.0, -1.0, xi),
            phi_grad(1.0, -1.0, 0.0, xi),
            phi_grad(0.0, 1.0, -1.0, xi),
            phi_grad(1.0, 1.0, 0.0, xi),
            phi_grad(-1.0, 1.0, 0.0, xi),
            phi_grad(0.0, -1.0, 1.0, xi),
            phi_grad(-1.0, 0.0, 1.0, xi),
            phi_grad(1.0, 0.0, 1.0, xi),
            phi_grad(0.0, 1.0, 1.0, xi),

            // Face nodes
            phi_grad(0.0, 0.0, -1.0, xi),
            phi_grad(0.0, -1.0, 0.0, xi),
            phi_grad(-1.0, 0.0, 0.0, xi),
            phi_grad(1.0, 0.0, 0.0, xi),
            phi_grad(0.0, 1.0, 0.0, xi),
            phi_grad(0.0, 0.0, 1.0, xi),

            // Center node
            phi_grad(0.0, 0.0, 0.0, xi)
        ])
    }
}

impl<T> FiniteElement<T> for Hex27Element<T>
where
    T: Real,
{
    type GeometryDim = U3;

    fn reference_jacobian(&self, reference_coords: &Point3<T>) -> Matrix3<T> {
        self.hex8.reference_jacobian(reference_coords)
    }

    fn map_reference_coords(&self, reference_coords: &OPoint<T, Self::ReferenceDim>) -> Point3<T> {
        self.hex8.map_reference_coords(reference_coords)
    }

    fn diameter(&self) -> T {
        self.hex8.diameter()
    }
}

impl<T> ElementConnectivity<T> for Hex27Connectivity
where
    T: Real,
{
    type Element = Hex27Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, global_vertices: &[Point3<T>]) -> Option<Self::Element> {
        let mut hex_vertices = [OPoint::origin(); 27];

        for (local_idx, global_idx) in self.0.iter().enumerate() {
            hex_vertices[local_idx] = global_vertices.get(*global_idx)?.clone();
        }

        Some(Hex27Element::from_vertices(hex_vertices))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Hex20Element<T: Scalar> {
    // Store a hex8 element for trilinear transformations from reference element
    hex8: Hex8Element<T>,
    vertices: [Point3<T>; 20],
}

impl<T: Scalar + Copy> Hex20Element<T> {
    pub fn from_vertices(vertices: [Point3<T>; 20]) -> Self {
        Self {
            hex8: Hex8Element::from_vertices(vertices[0..8].try_into().unwrap()),
            vertices,
        }
    }

    pub fn vertices(&self) -> &[Point3<T>] {
        &self.vertices
    }
}

impl<T: Real> Hex20Element<T> {
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
            Point3::new(1.0, -1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, 1.0, 1.0),
            // Edge nodes
            Point3::new(0.0, -1.0, -1.0),
            Point3::new(-1.0, 0.0, -1.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, 0.0, -1.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, -1.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(0.0, -1.0, 1.0),
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ])
    }
}

impl<T> FixedNodesReferenceFiniteElement<T> for Hex20Element<T>
where
    T: Real,
{
    type ReferenceDim = U3;
    type NodalDim = U20;

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn evaluate_basis(&self, xi: &Point3<T>) -> OMatrix<T, U1, U20> {
        // We define the shape functions as N_{alpha, beta, gamma} evaluated at xi such that
        //  N_{alpha, beta, gamma}([alpha, beta, gamma]) = 1,
        // but we define corner and edge nodes separately.

        // Formulas are adapted from the following website:
        // http://www.softeng.rl.ac.uk/st/projects/felib4/Docs/html/Level-0/brk20/brk20.html

        let phi_corner = |alpha, beta, gamma, xi: &Point3<T>|
            (1.0 / 8.0) * (1.0 + alpha * xi[0])
                * (1.0 + beta * xi[1])
                * (1.0 + gamma * xi[2])
                * (alpha * xi[0] + beta * xi[1] + gamma * xi[2] - 2.0);

        let phi_edge = |alpha, beta, gamma, xi: &Point3<T>| {
            let alpha2 = alpha * alpha;
            let beta2 = beta * beta;
            let gamma2 = gamma * gamma;
            (1.0 / 4.0) * (1.0 - (1.0 - alpha2) * xi[0]*xi[0])
                * (1.0 - (1.0 - beta2) * xi[1]*xi[1])
                * (1.0 - (1.0 - gamma2) * xi[2]*xi[2])
                * (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2])
        };

        OMatrix::<_, U1, U20>::from_row_slice(&[
            // Corner nodes
            phi_corner(-1.0, -1.0, -1.0, xi),
            phi_corner( 1.0, -1.0, -1.0, xi),
            phi_corner( 1.0,  1.0, -1.0, xi),
            phi_corner(-1.0,  1.0, -1.0, xi),
            phi_corner(-1.0, -1.0,  1.0, xi),
            phi_corner( 1.0, -1.0,  1.0, xi),
            phi_corner( 1.0,  1.0,  1.0, xi),
            phi_corner(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi_edge(0.0, -1.0, -1.0, xi),
            phi_edge(-1.0, 0.0, -1.0, xi),
            phi_edge(-1.0, -1.0, 0.0, xi),
            phi_edge(1.0, 0.0, -1.0, xi),
            phi_edge(1.0, -1.0, 0.0, xi),
            phi_edge(0.0, 1.0, -1.0, xi),
            phi_edge(1.0, 1.0, 0.0, xi),
            phi_edge(-1.0, 1.0, 0.0, xi),
            phi_edge(0.0, -1.0, 1.0, xi),
            phi_edge(-1.0, 0.0, 1.0, xi),
            phi_edge(1.0, 0.0, 1.0, xi),
            phi_edge(0.0, 1.0, 1.0, xi),
        ])
    }

    #[rustfmt::skip]
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn gradients(&self, xi: &Point3<T>) -> OMatrix<T, U3, U20> {
        let phi_grad_corner = |alpha, beta, gamma, xi: &Point3<T>| {
            // Decompose shape function as phi(xi) = (1/8) * f(xi) * g(xi),
            // with
            //  f(xi) = sum_i (alpha_i xi_i) - 2
            //  g(xi) = product_i (1 + alpha_i xi_i)
            // and use product rule to arrive at the below expression
            let f = alpha * xi[0] + beta * xi[1] + gamma * xi[2] - 2.0;
            let g = (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2]);
            let s = 1.0 / 8.0;
            Vector3::new(
                s * (alpha * g + f * alpha * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2])),
                s * (beta * g + f * beta * (1.0 + alpha * xi[0]) * (1.0 + gamma * xi[2])),
                s * (gamma * g + f * gamma * (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]))
            )
        };

        let phi_grad_edge = |alpha, beta, gamma, xi: &Point3<T>| {
            // Decompose shape function as phi(xi) = (1/8) * h(xi) * g(xi),
            // with
            //  h(xi) = product_i (1.0 - (1.0 - alpha_i^2) xi_i^2)
            //  g(xi) = product_i (1 + alpha_i xi_i)
            // and use product rule to arrive at the below expression
            let alpha2 = alpha * alpha;
            let beta2 = beta * beta;
            let gamma2 = gamma * gamma;
            let h = (1.0 - (1.0 - alpha2) * xi[0]*xi[0])
                * (1.0 - (1.0 - beta2) * xi[1]*xi[1])
                * (1.0 - (1.0 - gamma2) * xi[2]*xi[2]);
            let g = (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2]);
            let s = 1.0 / 4.0;

            // Note: we hope that the optimizer is able to optimize away most of these operations,
            // since alpha2, beta2, gamma2 should be known at compile-time, which
            // makes many of the terms here zero.
            let dh_xi0 = -2.0 * (1.0 - alpha2) * xi[0]
                * (1.0 - (1.0 - beta2) * xi[1]*xi[1])
                * (1.0 - (1.0 - gamma2) * xi[2]*xi[2]);
            let dh_xi1 = -2.0 * (1.0 - beta2) * xi[1]
                * (1.0 - (1.0 - alpha2) * xi[0] * xi[0])
                * (1.0 - (1.0 - gamma2) * xi[2] * xi[2]);
            let dh_xi2 = -2.0 * (1.0 - gamma2) * xi[2]
                * (1.0 - (1.0 - alpha2) * xi[0] * xi[0])
                * (1.0 - (1.0 - beta2) * xi[1] * xi[1]);
            Vector3::new(
                s * (dh_xi0 * g + h * alpha * (1.0 + beta * xi[1]) * (1.0 + gamma * xi[2])),
                s * (dh_xi1 * g + h * beta * (1.0 + alpha * xi[0]) * (1.0 + gamma * xi[2])),
                s * (dh_xi2 * g + h * gamma * (1.0 + alpha * xi[0]) * (1.0 + beta * xi[1]))
            )
        };

        OMatrix::from_columns(&[
            // Corner nodes
            phi_grad_corner(-1.0, -1.0, -1.0, xi),
            phi_grad_corner( 1.0, -1.0, -1.0, xi),
            phi_grad_corner( 1.0,  1.0, -1.0, xi),
            phi_grad_corner(-1.0,  1.0, -1.0, xi),
            phi_grad_corner(-1.0, -1.0,  1.0, xi),
            phi_grad_corner( 1.0, -1.0,  1.0, xi),
            phi_grad_corner( 1.0,  1.0,  1.0, xi),
            phi_grad_corner(-1.0,  1.0,  1.0, xi),

            // Edge nodes
            phi_grad_edge(0.0, -1.0, -1.0, xi),
            phi_grad_edge(-1.0, 0.0, -1.0, xi),
            phi_grad_edge(-1.0, -1.0, 0.0, xi),
            phi_grad_edge(1.0, 0.0, -1.0, xi),
            phi_grad_edge(1.0, -1.0, 0.0, xi),
            phi_grad_edge(0.0, 1.0, -1.0, xi),
            phi_grad_edge(1.0, 1.0, 0.0, xi),
            phi_grad_edge(-1.0, 1.0, 0.0, xi),
            phi_grad_edge(0.0, -1.0, 1.0, xi),
            phi_grad_edge(-1.0, 0.0, 1.0, xi),
            phi_grad_edge(1.0, 0.0, 1.0, xi),
            phi_grad_edge(0.0, 1.0, 1.0, xi),
        ])
    }
}

impl<T> FiniteElement<T> for Hex20Element<T>
where
    T: Real,
{
    type GeometryDim = U3;

    fn reference_jacobian(&self, reference_coords: &Point3<T>) -> Matrix3<T> {
        self.hex8.reference_jacobian(reference_coords)
    }

    fn map_reference_coords(&self, reference_coords: &OPoint<T, Self::ReferenceDim>) -> Point3<T> {
        self.hex8.map_reference_coords(reference_coords)
    }

    fn diameter(&self) -> T {
        self.hex8.diameter()
    }
}

impl<T> ElementConnectivity<T> for Hex20Connectivity
where
    T: Real,
{
    type Element = Hex20Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, global_vertices: &[Point3<T>]) -> Option<Self::Element> {
        let mut hex_vertices = [OPoint::origin(); 20];

        for (local_idx, global_idx) in self.0.iter().enumerate() {
            hex_vertices[local_idx] = global_vertices.get(*global_idx)?.clone();
        }

        Some(Hex20Element::from_vertices(hex_vertices))
    }
}

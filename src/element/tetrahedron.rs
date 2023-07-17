use std::cmp::Ordering;
use numeric_literals::replace_float_literals;

use crate::connectivity::{Connectivity, Tet10Connectivity, Tet20Connectivity, Tet4Connectivity};
use crate::element::{BoundsForElement, ClosestPoint, ClosestPointInElement, ElementConnectivity, FiniteElement, FixedNodesReferenceFiniteElement};
use crate::nalgebra::{
    distance, Matrix1x4, Matrix3, Matrix3x4, OMatrix, OPoint, Point3, Scalar, Vector3, U1, U10, U20, U3, U4,
};
use crate::Real;
use itertools::Itertools;
use nalgebra::distance_squared;
use fenris_geometry::AxisAlignedBoundingBox;

impl<T> ElementConnectivity<T> for Tet4Connectivity
where
    T: Real,
{
    type Element = Tet4Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[OPoint<T, Self::GeometryDim>]) -> Option<Self::Element> {
        Some(Tet4Element {
            vertices: [
                vertices.get(self.0[0])?.clone(),
                vertices.get(self.0[1])?.clone(),
                vertices.get(self.0[2])?.clone(),
                vertices.get(self.0[3])?.clone(),
            ],
        })
    }
}

impl<T> ElementConnectivity<T> for Tet10Connectivity
where
    T: Real,
{
    type Element = Tet10Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[OPoint<T, Self::GeometryDim>]) -> Option<Self::Element> {
        let mut tet10_vertices = [Point3::origin(); 10];
        for (i, v) in tet10_vertices.iter_mut().enumerate() {
            *v = vertices.get(self.0[i])?.clone();
        }

        let mut tet4_vertices = [Point3::origin(); 4];
        tet4_vertices.copy_from_slice(&tet10_vertices[0..4]);

        Some(Tet10Element {
            tet4: Tet4Element::from_vertices(tet4_vertices),
            vertices: tet10_vertices,
        })
    }
}

impl<T> ElementConnectivity<T> for Tet20Connectivity
where
    T: Real,
{
    type Element = Tet20Element<T>;
    type GeometryDim = U3;
    type ReferenceDim = U3;

    fn element(&self, vertices: &[OPoint<T, Self::GeometryDim>]) -> Option<Self::Element> {
        let mut tet20_vertices = [Point3::origin(); 20];
        for (i, v) in tet20_vertices.iter_mut().enumerate() {
            *v = vertices.get(self.0[i])?.clone();
        }

        let mut tet4_vertices = [Point3::origin(); 4];
        tet4_vertices.copy_from_slice(&tet20_vertices[0..4]);

        Some(Tet20Element {
            tet4: Tet4Element::from_vertices(tet4_vertices),
            vertices: tet20_vertices,
        })
    }
}

/// A Tet10 element, see documentation of [`Tet10Connectivity`](fenris::connectivity::Tet10Connectivity).
///
/// We currently assume that the reference-to-physical transformation is affine, meaning
/// that the geometry of the element is assumed to be affine.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Tet10Element<T>
where
    T: Scalar,
{
    tet4: Tet4Element<T>,
    vertices: [Point3<T>; 10],
}

impl<T> Tet10Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 10]) -> Self {
        let tet4_v = [
            vertices[0].clone(),
            vertices[1].clone(),
            vertices[2].clone(),
            vertices[3].clone(),
        ];
        Self {
            tet4: Tet4Element::from_vertices(tet4_v),
            vertices,
        }
    }

    pub fn vertices(&self) -> &[Point3<T>; 10] {
        &self.vertices
    }
}

impl<'a, T> From<&'a Tet4Element<T>> for Tet10Element<T>
where
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    fn from(tet4_element: &'a Tet4Element<T>) -> Self {
        let midpoint = |x: &Point3<_>, y: &Point3<_>| OPoint::from((x.coords + y.coords) * 0.5);

        let [a, b, c, d] = tet4_element.vertices;

        // TODO: Provide method for converting from Tet4 to Tet10 (From impl?)
        Tet10Element::from_vertices([
            a,
            b,
            c,
            d,
            midpoint(&a, &b),
            midpoint(&b, &c),
            midpoint(&a, &c),
            midpoint(&a, &d),
            midpoint(&c, &d),
            midpoint(&b, &d),
        ])
    }
}

impl<T> Tet10Element<T>
where
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self {
            tet4: Tet4Element::reference(),
            vertices: [
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, -1.0, -1.0),
                Point3::new(-1.0, 1.0, -1.0),
                Point3::new(-1.0, -1.0, 1.0),
                Point3::new(0.0, -1.0, -1.0),
                Point3::new(0.0, 0.0, -1.0),
                Point3::new(-1.0, 0.0, -1.0),
                Point3::new(-1.0, -1.0, 0.0),
                Point3::new(-1.0, 0.0, 0.0),
                Point3::new(0.0, -1.0, 0.0),
            ],
        }
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
impl<T> FixedNodesReferenceFiniteElement<T> for Tet10Element<T>
where
    T: Real,
{
    type ReferenceDim = U3;
    type NodalDim = U10;

    #[rustfmt::skip]
    fn evaluate_basis(&self, xi: &Point3<T>) -> OMatrix<T, U1, U10> {
        // We express the basis functions of Tet10 as products of
        // the Tet4 basis functions.
        let psi = self.tet4.evaluate_basis(xi);
        OMatrix::from([
            psi[0] * (2.0 * psi[0] - 1.0),
            psi[1] * (2.0 * psi[1] - 1.0),
            psi[2] * (2.0 * psi[2] - 1.0),
            psi[3] * (2.0 * psi[3] - 1.0),
            4.0 * psi[0] * psi[1],
            4.0 * psi[1] * psi[2],
            4.0 * psi[0] * psi[2],
            4.0 * psi[0] * psi[3],
            4.0 * psi[2] * psi[3],
            4.0 * psi[1] * psi[3]
        ])
    }

    #[rustfmt::skip]
    fn gradients(&self, xi: &Point3<T>) -> OMatrix<T, U3, U10> {
        // Similarly to `evaluate_basis`, we may implement the gradients of
        // Tet10 with the help of the function values and gradients of Tet4
        let psi = self.tet4.evaluate_basis(xi);
        let g = self.tet4.gradients(xi);

        // Gradient of vertex node i
        let vertex_gradient = |i| g.index((.., i)) * (4.0 * psi[i] - 1.0);

        // Gradient of edge node on the edge between vertex i and j
        let edge_gradient = |i, j|
            g.index((.., i)) * (4.0 * psi[j]) + g.index((.., j)) * (4.0 * psi[i]);

        OMatrix::from_columns(&[
            vertex_gradient(0),
            vertex_gradient(1),
            vertex_gradient(2),
            vertex_gradient(3),
            edge_gradient(0, 1),
            edge_gradient(1, 2),
            edge_gradient(0, 2),
            edge_gradient(0, 3),
            edge_gradient(2, 3),
            edge_gradient(1, 3)
        ])
    }
}

impl<T> FiniteElement<T> for Tet10Element<T>
where
    T: Real,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        self.tet4.reference_jacobian(xi)
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        self.tet4.map_reference_coords(xi)
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.tet4.diameter()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Tet20Element<T>
where
    T: Scalar,
{
    tet4: Tet4Element<T>,
    vertices: [Point3<T>; 20],
}

impl<T> Tet20Element<T>
where
    T: Real,
{
    pub fn from_tet4_vertices(vertices: [Point3<T>; 4]) -> Self {
        // TODO: Test this method
        let tet4_element = Tet4Element::from_vertices(vertices);
        let tet20_ref = Tet20Element::reference();
        let mut vertices = [OPoint::origin(); 20];
        // The reference element has the correct placement of nodes in the reference element.
        // We can obtain the vertex positions in physical space by mapping coordinates
        // with the Tet4 element that we have constructed. This is currently just a quick
        // way to avoid having to write down the vertices manually, which is error prone
        // TODO: Find a more canonical way of doing these things so that we only have
        // the canonical description in a single location
        for (v_ref, v_physical) in tet20_ref.vertices().iter().zip(&mut vertices) {
            *v_physical = tet4_element.map_reference_coords(v_ref);
        }
        Self::from_vertices(vertices)
    }

    // TODO: Remove this method so that it's not possible to create curved Tet20Elements
    // (we do *not* use isoparametric transformations at the moment). Same with Tet10Element
    pub fn from_vertices(vertices: [Point3<T>; 20]) -> Self {
        let tet4_v = [
            vertices[0].clone(),
            vertices[1].clone(),
            vertices[2].clone(),
            vertices[3].clone(),
        ];
        Self {
            tet4: Tet4Element::from_vertices(tet4_v),
            vertices,
        }
    }

    pub fn vertices(&self) -> &[Point3<T>; 20] {
        &self.vertices
    }
}

impl<T> Tet20Element<T>
where
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self {
            tet4: Tet4Element::reference(),
            vertices: [
                // Vertex nodes
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, -1.0, -1.0),
                Point3::new(-1.0, 1.0, -1.0),
                Point3::new(-1.0, -1.0, 1.0),
                // Edge nodes
                // Between node 0 and 1
                Point3::new(-1.0 / 3.0, -1.0, -1.0),
                Point3::new(1.0 / 3.0, -1.0, -1.0),
                // Between node 0 and 2
                Point3::new(-1.0, -1.0 / 3.0, -1.0),
                Point3::new(-1.0, 1.0 / 3.0, -1.0),
                // Between node 0 and 3
                Point3::new(-1.0, -1.0, -1.0 / 3.0),
                Point3::new(-1.0, -1.0, 1.0 / 3.0),
                // Between node 1 and 2
                Point3::new(1.0 / 3.0, -1.0 / 3.0, -1.0),
                Point3::new(-1.0 / 3.0, 1.0 / 3.0, -1.0),
                // Between node 1 and 3
                Point3::new(1.0 / 3.0, -1.0, -1.0 / 3.0),
                Point3::new(-1.0 / 3.0, -1.0, 1.0 / 3.0),
                // Between node 2 and 3
                Point3::new(-1.0, 1.0 / 3.0, -1.0 / 3.0),
                Point3::new(-1.0, -1.0 / 3.0, 1.0 / 3.0),
                // On face {0, 1, 2}
                Point3::new(-1.0 / 3.0, -1.0 / 3.0, -1.0),
                // On face {0, 1, 3}
                Point3::new(-1.0 / 3.0, -1.0, -1.0 / 3.0),
                // On face {0, 2, 3}
                Point3::new(-1.0, -1.0 / 3.0, -1.0 / 3.0),
                // On face {1, 2, 3}
                Point3::new(-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0),
            ],
        }
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
impl<T> FixedNodesReferenceFiniteElement<T> for Tet20Element<T>
where
    T: Real,
{
    type ReferenceDim = U3;
    type NodalDim = U20;

    #[rustfmt::skip]
    fn evaluate_basis(&self, xi: &Point3<T>) -> OMatrix<T, U1, U20> {
        // We express the basis functions of Tet10 as products of
        // the Tet4 basis functions. See Zienkiewicz et al., Finite Element Method
        // for the basis functions
        let psi = self.tet4.evaluate_basis(xi);

        // We define the edge functions by associating a particular edge node
        // with its closest vertex.
        let phi_edge = |closest: usize, other: usize|
            (9.0 / 2.0) * psi[closest] * psi[other] * (3.0 * psi[closest] - 1.0);
        // The face functions are associated with the three vertex nodes
        // that make up each facec
        let phi_face = |a: usize, b: usize, c: usize|
            27.0 * psi[a] * psi[b] * psi[c];

        OMatrix::<_, U1, U20>::from_row_slice(&[
            // Corner nodes
            0.5 * psi[0] * (3.0 * psi[0] - 1.0) * (3.0 * psi[0] - 2.0),
            0.5 * psi[1] * (3.0 * psi[1] - 1.0) * (3.0 * psi[1] - 2.0),
            0.5 * psi[2] * (3.0 * psi[2] - 1.0) * (3.0 * psi[2] - 2.0),
            0.5 * psi[3] * (3.0 * psi[3] - 1.0) * (3.0 * psi[3] - 2.0),

            // Edge nodes
            // Between node 0 and 1
            phi_edge(0, 1),
            phi_edge(1, 0),
            // Between node 0 and 2
            phi_edge(0, 2),
            phi_edge(2, 0),
            // Between node 0 and 3
            phi_edge(0, 3),
            phi_edge(3, 0),
            // Between node 1 and 2
            phi_edge(1, 2),
            phi_edge(2, 1),
            // Between node 1 and 3
            phi_edge(1, 3),
            phi_edge(3, 1),
            // Between node 2 and 3
            phi_edge(2, 3),
            phi_edge(3, 2),

            // Faces nodes
            // On face {0, 1, 2}
            phi_face(0, 1, 2),
            // On face {0, 1, 3}
            phi_face(0, 1, 3),
            // On face {0, 2, 3}
            phi_face(0, 2, 3),
            // On face {1, 2, 3}
            phi_face(1, 2, 3),
        ])
    }

    #[rustfmt::skip]
    fn gradients(&self, xi: &Point3<T>) -> OMatrix<T, U3, U20> {
        // Similarly to `evaluate_basis`, we may implement the gradients of
        // Tet10 with the help of the function values and gradients of Tet4
        let psi = self.tet4.evaluate_basis(xi);
        let tet4_gradients = self.tet4.gradients(xi);
        let g = |i| tet4_gradients.index((.., i));

        // Gradient of vertex node i
        let vertex_gradient = |i| -> Vector3<T> {
            let p = psi[i];
            g(i) * 0.5 * (27.0 * p * p - 18.0 * p + 2.0)
        };

        // Gradient of edge node on the edge between vertex a and b
        let edge_gradient = |a, b| -> Vector3<T> {
            let pa = psi[a];
            let pb = psi[b];
            ( g(a) * (pb * (6.0 * pa - 1.0)) + g(b) * (pa * (3.0 * pa - 1.0))) * (9.0 / 2.0)
        };

        let face_gradient = |a, b, c| -> Vector3<T> {
            (g(a) * psi[b] * psi[c] + g(b) * psi[a] * psi[c] + g(c) * psi[a] * psi[b]) * 27.0
        };

        OMatrix::from_columns(&[
            // Vertex nodes
            vertex_gradient(0),
            vertex_gradient(1),
            vertex_gradient(2),
            vertex_gradient(3),

            // Edge nodes
            // Between node 0 and 1
            edge_gradient(0, 1),
            edge_gradient(1, 0),
            // Between node 0 and 2
            edge_gradient(0, 2),
            edge_gradient(2, 0),
            // Between node 0 and 3
            edge_gradient(0, 3),
            edge_gradient(3, 0),
            // Between node 1 and 2
            edge_gradient(1, 2),
            edge_gradient(2, 1),
            // Between node 1 and 3
            edge_gradient(1, 3),
            edge_gradient(3, 1),
            // Between node 2 and 3
            edge_gradient(2, 3),
            edge_gradient(3, 2),

            // Faces nodes
            // On face {0, 1, 2}
            face_gradient(0, 1, 2),
            // On face {0, 1, 3}
            face_gradient(0, 1, 3),
            // On face {0, 2, 3}
            face_gradient(0, 2, 3),
            // On face {1, 2, 3}
            face_gradient(1, 2, 3),
        ])
    }
}

impl<T> FiniteElement<T> for Tet20Element<T>
where
    T: Real,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        self.tet4.reference_jacobian(xi)
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        self.tet4.map_reference_coords(xi)
    }

    // TODO: Write tests for diameter
    fn diameter(&self) -> T {
        self.tet4.diameter()
    }
}

impl<'a, T> From<&'a Tet4Element<T>> for Tet20Element<T>
where
    T: Real,
{
    fn from(tet4: &'a Tet4Element<T>) -> Self {
        // TODO: Test this!
        Self::from_tet4_vertices(tet4.vertices().clone())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Tet4Element<T>
where
    T: Scalar,
{
    vertices: [Point3<T>; 4],
}

impl<T> Tet4Element<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 4]) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Point3<T>; 4] {
        &self.vertices
    }
}

impl<T> Tet4Element<T>
where
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self {
            vertices: [
                Point3::new(-1.0, -1.0, -1.0),
                Point3::new(1.0, -1.0, -1.0),
                Point3::new(-1.0, 1.0, -1.0),
                Point3::new(-1.0, -1.0, 1.0),
            ],
        }
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
impl<T> FixedNodesReferenceFiniteElement<T> for Tet4Element<T>
where
    T: Real,
{
    type ReferenceDim = U3;
    type NodalDim = U4;

    #[rustfmt::skip]
    fn evaluate_basis(&self, xi: &Point3<T>) -> Matrix1x4<T> {
        Matrix1x4::from_row_slice(&[
            -0.5 * xi.x - 0.5 * xi.y - 0.5 * xi.z - 0.5,
            0.5 * xi.x + 0.5,
            0.5 * xi.y + 0.5,
            0.5 * xi.z + 0.5
        ])
    }

    #[rustfmt::skip]
    fn gradients(&self, _reference_coords: &Point3<T>) -> Matrix3x4<T> {
        Matrix3x4::from_columns(&[
            Vector3::new(-0.5, -0.5, -0.5),
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.0, 0.0, 0.5)
        ])
    }
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn is_likely_in_tet_ref_interior<T: Real>(xi: &Point3<T>) -> bool {
    let eps = 4.0 * T::default_epsilon();
    xi.x >= -1.0 - eps
        && xi.y >= -1.0 - eps
        && xi.z >= -1.0 - eps
        && xi.x + xi.y + xi.z <= eps
}

impl<T> FiniteElement<T> for Tet4Element<T>
where
    T: Real,
{
    type GeometryDim = U3;

    #[allow(non_snake_case)]
    fn reference_jacobian(&self, xi: &Point3<T>) -> Matrix3<T> {
        // TODO: Could store this matrix directly in the element, in order
        // to avoid repeated computation
        let X = Matrix3x4::from_fn(|i, j| self.vertices[j][i]);
        let G = self.gradients(xi);
        X * G.transpose()
    }

    #[allow(non_snake_case)]
    fn map_reference_coords(&self, xi: &Point3<T>) -> Point3<T> {
        // TODO: Store this X matrix directly in Self...?
        let X = Matrix3x4::from_fn(|i, j| self.vertices[j][i]);
        let N = self.evaluate_basis(xi);
        OPoint::from(&X * &N.transpose())
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

impl<T: Real> BoundsForElement<T> for Tet4Element<T> {
    fn element_bounds(&self) -> AxisAlignedBoundingBox<T, Self::GeometryDim> {
        AxisAlignedBoundingBox::from_points(self.vertices()).unwrap()
    }
}

impl<T: Real> ClosestPointInElement<T> for Tet4Element<T> {
    #[allow(non_snake_case)]
    fn closest_point(&self, p: &OPoint<T, Self::GeometryDim>) -> ClosestPoint<T, Self::ReferenceDim> {
        let xi_interior = {
            // Transformation is affine, so Jacobian is constant:
            //  p = A xi + p0
            // for some p0 which we can determine by evaluating at xi = 0
            let A = self.reference_jacobian(&Point3::origin());
            A.try_inverse()
                .map(|a_inv| {
                    let p0 = self.map_reference_coords(&Point3::origin());
                    Point3::from(a_inv * (p - p0))
                })
                // If the inverse transformation doesn't lead to a point clearly inside
                // the reference domain, we assume that the closest point is on the boundary
                .filter(is_likely_in_tet_ref_interior)
        };

        if !p.coords.norm_squared().is_finite() {
            panic!("p not finite");
        }

        let conn = Tet4Connectivity([0, 1, 2, 3]);
        let face_elements_iter = (0 .. 4)
            .map(|face_idx| conn.get_face_connectivity(face_idx).unwrap())
            .map(|face_conn| face_conn.element(self.vertices()).unwrap());

        let (face_idx, _, xi_face, dist2_face) = face_elements_iter
            .enumerate()
            .map(|(face_idx, tri_element)| {
                let xi_closest = tri_element.closest_point(p).point().clone();
                let x_closest = tri_element.map_reference_coords(&xi_closest);
                let dist2 = distance_squared(&x_closest, p);
                if !dist2.is_finite() {
                    panic!("not finite");
                }

                (face_idx, tri_element, xi_closest, dist2)
            })
            .min_by(|(_, _, _, d1), (_, _, _, d2)| d1.partial_cmp(d2).unwrap_or(Ordering::Less))
            .expect("Always have 4 > 0 faces");

        if let Some(xi_interior) = xi_interior {
            let x_interior = self.map_reference_coords(&xi_interior);
            let dist2_interior = distance_squared(p, &x_interior);
            if dist2_interior < dist2_face {
                return ClosestPoint::InElement(xi_interior);
            }
        }

        // Next, we need to obtain the coordinates for the point on the face in the
        // tetrahedron reference element.
        // We can do this by considering the corresponding tetrahedron reference element
        // and use the same face index to map into "physical space" which will in fact
        // be the reference coordinates that we need
        let reference_element = Self::reference();
        let xi = conn
            .get_face_connectivity(face_idx).unwrap()
            .element(reference_element.vertices()).unwrap()
            .map_reference_coords(&xi_face);
        ClosestPoint::ClosestPoint(xi)
    }
}

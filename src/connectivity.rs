use crate::geometry::{Hexahedron, LineSegment2d, Quad2d, Tetrahedron, Triangle, Triangle2d, Triangle3d};
use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Point2, Point3, RealField, Scalar, U2, U3};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// Represents the type of the faces for a given cell connectivity.
pub type CellFace<T, Cell> = <<Cell as Connectivity>::FaceConnectivity as CellConnectivity<T, U2>>::Cell;

pub trait Connectivity: Clone {
    type FaceConnectivity: Connectivity;

    fn num_faces(&self) -> usize;
    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity>;

    fn vertex_indices(&self) -> &[usize];
}

impl Connectivity for () {
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        const EMPTY_SLICE: &[usize] = &[];
        &EMPTY_SLICE
    }
}

pub trait ConnectivityMut: Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize];
}

pub trait CellConnectivity<T, D>: Connectivity
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    type Cell;

    /// Legacy method
    ///
    /// TODO: Remove in favor of using `num_faces` and `get_face_Connectivity`
    fn for_each_face<F>(&self, mut f: F)
    where
        F: FnMut(Self::FaceConnectivity),
    {
        let num_faces = self.num_faces();
        for i in 0..num_faces {
            let face = self
                .get_face_connectivity(i)
                .expect("Since index is in bounds, connectivity must exist.");
            f(face)
        }
    }

    fn cell(&self, vertices: &[OPoint<T, D>]) -> Option<Self::Cell>;
}

/// Connectivity for a two-dimensional Quad9 element.
///
/// A Quad9 element has a quadrilateral geometry, with 9 nodes evenly distributed across
/// the surface of the reference element [-1, 1]^2.
///
/// Note that the element is not completely isoparametric: The element itself is assumed to have
/// straight faces, i.e. the same as a bilinear quad element.
///
/// The schematic below demonstrates the node numbering.
///
/// ```text
/// 3____6____2
/// |         |
/// 7    8    5
/// |         |
/// 0____4____1
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quad9d2Connectivity(pub [usize; 9]);

impl<'a> From<&'a Quad9d2Connectivity> for Quad4d2Connectivity {
    fn from(quad9: &'a Quad9d2Connectivity) -> Self {
        let Quad9d2Connectivity(indices) = quad9;
        Quad4d2Connectivity([indices[0], indices[1], indices[2], indices[3]])
    }
}

impl Deref for Quad9d2Connectivity {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Segment2d1Connectivity(pub [usize; 2]);

impl Connectivity for Segment2d1Connectivity {
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Segment2d1Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Segment2d2Connectivity(pub [usize; 2]);

impl Connectivity for Segment2d2Connectivity {
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Segment2d2Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U2> for Segment2d2Connectivity
where
    T: Scalar,
{
    type Cell = LineSegment2d<T>;

    fn cell(&self, vertices: &[Point2<T>]) -> Option<Self::Cell> {
        let a = vertices.get(self.0[0]).cloned()?;
        let b = vertices.get(self.0[1]).cloned()?;
        Some(LineSegment2d::new(a, b))
    }
}

/// Connectivity for a two-dimensional Quad4 element.
///
/// A Quad4 element has a quadrilateral geometry, with 4 nodes distributed across
/// the corners of the reference element [-1, 1]^2.
///
/// The schematic below demonstrates the node numbering.
///
/// ```text
/// 3_________2
/// |         |
/// |         |
/// |         |
/// 0_________1
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Quad4d2Connectivity(pub [usize; 4]);

impl Deref for Quad4d2Connectivity {
    type Target = [usize; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Quad4d2Connectivity {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Connectivity for Quad4d2Connectivity {
    type FaceConnectivity = Segment2d2Connectivity;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let idx = &self.0;
        if index < 4 {
            Some(Segment2d2Connectivity([idx[index], idx[(index + 1) % 4]]))
        } else {
            None
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Quad4d2Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U2> for Quad4d2Connectivity
where
    T: Scalar,
{
    type Cell = Quad2d<T>;

    fn cell(&self, vertices: &[Point2<T>]) -> Option<Self::Cell> {
        Some(Quad2d([
            vertices.get(self.0[0]).cloned()?,
            vertices.get(self.0[1]).cloned()?,
            vertices.get(self.0[2]).cloned()?,
            vertices.get(self.0[3]).cloned()?,
        ]))
    }
}

/// Connectivity for a two-dimensional Tri3 element.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Tri3d2Connectivity(pub [usize; 3]);

impl Connectivity for Tri3d2Connectivity {
    type FaceConnectivity = Segment2d2Connectivity;

    fn num_faces(&self) -> usize {
        3
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let idx = &self.0;
        if index < 3 {
            Some(Segment2d2Connectivity([idx[index], idx[(index + 1) % 3]]))
        } else {
            None
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Tri3d2Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U2> for Tri3d2Connectivity
where
    T: Scalar,
{
    type Cell = Triangle2d<T>;

    fn cell(&self, vertices: &[Point2<T>]) -> Option<Self::Cell> {
        Some(Triangle([
            vertices.get(self.0[0]).cloned()?,
            vertices.get(self.0[1]).cloned()?,
            vertices.get(self.0[2]).cloned()?,
        ]))
    }
}

impl Deref for Tri3d2Connectivity {
    type Target = [usize; 3];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tri3d2Connectivity {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Connectivity for a two-dimensional Tri6 element.
////
///
/// The schematic below demonstrates the node numbering.
///
/// ```text
/// 2
/// |`\
/// |  `\
/// 5    `4
/// |      `\
/// |        `\
/// 0-----3----1
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Tri6d2Connectivity(pub [usize; 6]);

impl<'a> From<&'a Tri6d2Connectivity> for Tri3d2Connectivity {
    fn from(tri6: &'a Tri6d2Connectivity) -> Self {
        let Tri6d2Connectivity(indices) = tri6;
        Tri3d2Connectivity([indices[0], indices[1], indices[2]])
    }
}

impl Deref for Tri6d2Connectivity {
    type Target = [usize; 6];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Connectivity for Tri6d2Connectivity {
    type FaceConnectivity = Segment3d2Connectivity;

    fn num_faces(&self) -> usize {
        3
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let idx = &self.0;
        if index < 3 {
            Some(Segment3d2Connectivity([
                idx[index],
                idx[index + 3],
                idx[(index + 1) % 3],
            ]))
        } else {
            None
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Tri6d2Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U2> for Tri6d2Connectivity
where
    T: Scalar,
{
    type Cell = Triangle2d<T>;

    fn cell(&self, vertices: &[Point2<T>]) -> Option<Self::Cell> {
        Some(Triangle([
            vertices.get(self.0[0]).cloned()?,
            vertices.get(self.0[1]).cloned()?,
            vertices.get(self.0[2]).cloned()?,
        ]))
    }
}

/// Connectivity for a 2D segment element of polynomial degree 2.
///
/// This connectivity is used e.g. to represent the faces of a Quad9 element.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Segment3d2Connectivity(pub [usize; 3]);

impl Connectivity for Segment3d2Connectivity {
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Segment3d2Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl Connectivity for Quad9d2Connectivity {
    type FaceConnectivity = Segment3d2Connectivity;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;
        match index {
            0 => Some(Segment3d2Connectivity([v[0], v[4], v[1]])),
            1 => Some(Segment3d2Connectivity([v[1], v[5], v[2]])),
            2 => Some(Segment3d2Connectivity([v[2], v[6], v[3]])),
            3 => Some(Segment3d2Connectivity([v[3], v[7], v[0]])),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Quad9d2Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

/// TODO: Move this somewhere else. Also figure out a better way to deal with Cell/Element
/// distinctions
impl<T> CellConnectivity<T, U2> for Quad9d2Connectivity
where
    T: Scalar,
{
    type Cell = <Quad4d2Connectivity as CellConnectivity<T, U2>>::Cell;

    fn cell(&self, vertices: &[Point2<T>]) -> Option<Self::Cell> {
        let quad4 = Quad4d2Connectivity::from(self);
        quad4.cell(vertices)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quad8d3Connectivity(pub [usize; 8]);

impl Connectivity for Quad8d3Connectivity {
    type FaceConnectivity = Segment3d3Connectivity;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;
        let segment = |a, b, c| Some(Segment3d3Connectivity([v[a], v[b], v[c]]));

        match index {
            // TODO: We need to fix this later. We're currently using a kind of bogus
            // ordering for segments
            0 => segment(0, 4, 1),
            1 => segment(1, 5, 2),
            2 => segment(2, 6, 3),
            3 => segment(3, 7, 0),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Quad8d3Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quad9d3Connectivity(pub [usize; 9]);

impl Connectivity for Quad9d3Connectivity {
    type FaceConnectivity = Segment3d3Connectivity;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;
        let segment = |a, b, c| Some(Segment3d3Connectivity([v[a], v[b], v[c]]));

        match index {
            // TODO: We need to fix this later. We're currently using a kind of bogus
            // ordering for segments
            0 => segment(0, 4, 1),
            1 => segment(1, 5, 2),
            2 => segment(2, 6, 3),
            3 => segment(3, 7, 0),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Quad9d3Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tet4Connectivity(pub [usize; 4]);

impl Connectivity for Tet4Connectivity {
    type FaceConnectivity = Tri3d3Connectivity;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;
        // Note: need to carefully choose faces such that their normal point outwards,
        // otherwise extracted surface meshes have normals the wrong way around
        match index {
            0 => Some(Tri3d3Connectivity([v[0], v[2], v[1]])),
            1 => Some(Tri3d3Connectivity([v[0], v[1], v[3]])),
            2 => Some(Tri3d3Connectivity([v[1], v[2], v[3]])),
            3 => Some(Tri3d3Connectivity([v[0], v[3], v[2]])),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Tet4Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Tet4Connectivity
where
    T: RealField,
{
    type Cell = Tetrahedron<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        let mut tet_vertices = [Point3::origin(); 4];
        for (tet_v, idx) in izip!(&mut tet_vertices, &self.0) {
            *tet_v = vertices[*idx];
        }
        Some(Tetrahedron::from_vertices(tet_vertices))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Quad4d3Connectivity(pub [usize; 4]);

impl Connectivity for Quad4d3Connectivity {
    type FaceConnectivity = Segment2d3Connectivity;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;

        let segment = |a, b| Some(Segment2d3Connectivity([v[a], v[b]]));

        match index {
            0 => segment(0, 1),
            1 => segment(1, 2),
            2 => segment(2, 3),
            3 => segment(3, 0),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Quad4d3Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hex8Connectivity(pub [usize; 8]);

impl Connectivity for Hex8Connectivity {
    type FaceConnectivity = Quad4d3Connectivity;

    fn num_faces(&self) -> usize {
        6
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;

        let quad = |i, j, k, l| Some(Quad4d3Connectivity([v[i], v[j], v[k], v[l]]));

        // Must choose faces carefully so that they point towards the *exterior*,
        // in order to get proper surface normals for the boundary
        // Note: This is just the oppositely oriented choices of the current (at time of writing)
        // implementation of ConvexPolyhedron for Hexahedron
        match index {
            0 => quad(3, 2, 1, 0),
            1 => quad(0, 1, 5, 4),
            2 => quad(1, 2, 6, 5),
            3 => quad(2, 3, 7, 6),
            4 => quad(4, 7, 3, 0),
            5 => quad(5, 6, 7, 4),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Hex8Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Hex8Connectivity
where
    T: RealField,
{
    type Cell = Hexahedron<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        let mut hex_vertices = [Point3::origin(); 8];
        for (v, idx) in izip!(&mut hex_vertices, &self.0) {
            *v = vertices[*idx];
        }
        Some(Hexahedron::from_vertices(hex_vertices))
    }
}

/// Connectivity for a 3D tri-quadratic Hex element.
///
/// The node ordering is the same as defined by gmsh, see
/// <http://gmsh.info/doc/texinfo/gmsh.html#Low-order-elements> for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hex27Connectivity(pub [usize; 27]);

impl Connectivity for Hex27Connectivity {
    type FaceConnectivity = Quad9d3Connectivity;

    fn num_faces(&self) -> usize {
        6
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;

        // The macro just takes care of the boilerplate from mapping local
        // indices to global indices in the resulting Quad9d3Connectivity
        macro_rules! quad9 {
            ($($idx:expr),+) => { Some(Quad9d3Connectivity([$(v[$idx],)+])) }
        }

        // Must choose faces carefully so that they point towards the *exterior*,
        // in order to get proper surface normals for the boundary
        match index {
            0 => quad9!(0, 3, 2, 1, 9, 13, 11, 8, 20),
            1 => quad9!(0, 1, 5, 4, 8, 12, 16, 10, 21),
            2 => quad9!(1, 2, 6, 5, 11, 14, 18, 12, 23),
            3 => quad9!(2, 3, 7, 6, 13, 15, 19, 14, 24),
            4 => quad9!(0, 4, 7, 3, 10, 17, 15, 9, 22),
            5 => quad9!(4, 5, 6, 7, 16, 18, 19, 17, 25),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Hex27Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Hex27Connectivity
where
    T: RealField,
{
    type Cell = Hexahedron<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        let mut hex_vertices = [Point3::origin(); 8];
        // The first 8 vertices are the same as the linear hex element
        for (v, idx) in izip!(&mut hex_vertices, &self.0) {
            *v = vertices[*idx];
        }
        Some(Hexahedron::from_vertices(hex_vertices))
    }
}

/// Connectivity for a 3D 20-node Hex element.
///
/// The node ordering is the same as defined by gmsh, see
/// <http://gmsh.info/doc/texinfo/gmsh.html#Low-order-elements> for more information.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hex20Connectivity(pub [usize; 20]);

impl Connectivity for Hex20Connectivity {
    // TODO: Implement FaceConnectivity for Hex27
    type FaceConnectivity = Quad8d3Connectivity;

    fn num_faces(&self) -> usize {
        6
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;

        // The macro just takes care of the boilerplate from mapping local
        // indices to global indices in the resulting Quad9d3Connectivity
        macro_rules! quad8 {
            ($($idx:expr),+) => { Some(Quad8d3Connectivity([$(v[$idx],)+])) }
        }

        // Must choose faces carefully so that they point towards the *exterior*,
        // in order to get proper surface normals for the boundary

        // Note: This is identical to Hex27, except for the lack of the midpoint on each face
        match index {
            0 => quad8!(0, 3, 2, 1, 9, 13, 11, 8),
            1 => quad8!(0, 1, 5, 4, 8, 12, 16, 10),
            2 => quad8!(1, 2, 6, 5, 11, 14, 18, 12),
            3 => quad8!(2, 3, 7, 6, 13, 15, 19, 14),
            4 => quad8!(0, 4, 7, 3, 10, 17, 15, 9),
            5 => quad8!(4, 5, 6, 7, 16, 18, 19, 17),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Hex20Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Hex20Connectivity
where
    T: RealField,
{
    type Cell = Hexahedron<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        let mut hex_vertices = [Point3::origin(); 8];
        // The first 8 vertices are the same as the linear hex element
        for (v, idx) in izip!(&mut hex_vertices, &self.0) {
            *v = vertices[*idx];
        }
        Some(Hexahedron::from_vertices(hex_vertices))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Tri3d3Connectivity(pub [usize; 3]);

impl Connectivity for Tri3d3Connectivity {
    type FaceConnectivity = Segment2d3Connectivity;

    fn num_faces(&self) -> usize {
        3
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let segment = |i, j| Some(Segment2d3Connectivity([self.0[i], self.0[j]]));
        match index {
            0 => segment(0, 1),
            1 => segment(1, 2),
            2 => segment(2, 0),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Tri3d3Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Tri3d3Connectivity
where
    T: Scalar,
{
    type Cell = Triangle3d<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        Some(Triangle([
            vertices.get(self.0[0]).cloned()?,
            vertices.get(self.0[1]).cloned()?,
            vertices.get(self.0[2]).cloned()?,
        ]))
    }
}

impl Deref for Tri3d3Connectivity {
    type Target = [usize; 3];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tri3d3Connectivity {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Tri6d3Connectivity(pub [usize; 6]);

impl Deref for Tri6d3Connectivity {
    type Target = [usize; 6];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Connectivity for Tri6d3Connectivity {
    type FaceConnectivity = Segment3d3Connectivity;

    fn num_faces(&self) -> usize {
        3
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let idx = &self.0;
        if index < 3 {
            Some(Segment3d3Connectivity([
                idx[index],
                idx[index + 3],
                idx[(index + 1) % 3],
            ]))
        } else {
            None
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Tri6d3Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Tri6d3Connectivity
where
    T: Scalar,
{
    type Cell = Triangle3d<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        Some(Triangle([
            vertices.get(self.0[0]).cloned()?,
            vertices.get(self.0[1]).cloned()?,
            vertices.get(self.0[2]).cloned()?,
        ]))
    }
}

/// Connectivity for a 10-node tetrahedron element.
///
/// See GMSH documentation for node ordering.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tet10Connectivity(pub [usize; 10]);

impl<'a> From<&'a Tet10Connectivity> for Tet4Connectivity {
    fn from(tet10: &'a Tet10Connectivity) -> Self {
        let Tet10Connectivity(indices) = tet10;
        Tet4Connectivity([indices[0], indices[1], indices[2], indices[3]])
    }
}

impl Connectivity for Tet10Connectivity {
    type FaceConnectivity = Tri6d3Connectivity;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face_connectivity(&self, index: usize) -> Option<Self::FaceConnectivity> {
        let v = &self.0;
        match index {
            0 => Some(Tri6d3Connectivity([v[0], v[2], v[1], v[6], v[5], v[4]])),
            1 => Some(Tri6d3Connectivity([v[0], v[1], v[3], v[4], v[9], v[7]])),
            2 => Some(Tri6d3Connectivity([v[1], v[2], v[3], v[5], v[8], v[9]])),
            3 => Some(Tri6d3Connectivity([v[0], v[3], v[2], v[7], v[8], v[6]])),
            _ => None,
        }
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Tet10Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Tet10Connectivity
where
    T: RealField,
{
    type Cell = Tetrahedron<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        let mut tet4_v = [0, 0, 0, 0];
        tet4_v.clone_from_slice(&self.0[0..4]);
        let tet4 = Tet4Connectivity(tet4_v);
        tet4.cell(vertices)
    }
}

/// Connectivity for a 20-node tetrahedron element.
///
/// See GMSH documentation for node ordering.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tet20Connectivity(pub [usize; 20]);

impl<'a> From<&'a Tet20Connectivity> for Tet4Connectivity {
    fn from(tet20: &'a Tet20Connectivity) -> Self {
        let Tet20Connectivity(indices) = tet20;
        Tet4Connectivity([indices[0], indices[1], indices[2], indices[3]])
    }
}

impl Connectivity for Tet20Connectivity {
    // TODO: Connectivity?
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Tet20Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

impl<T> CellConnectivity<T, U3> for Tet20Connectivity
where
    T: RealField,
{
    type Cell = Tetrahedron<T>;

    fn cell(&self, vertices: &[Point3<T>]) -> Option<Self::Cell> {
        let mut tet4_v = [0, 0, 0, 0];
        tet4_v.clone_from_slice(&self.0[0..4]);
        let tet4 = Tet4Connectivity(tet4_v);
        tet4.cell(vertices)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Segment2d3Connectivity(pub [usize; 2]);

impl Connectivity for Segment2d3Connectivity {
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Segment2d3Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Segment3d3Connectivity(pub [usize; 3]);

impl Connectivity for Segment3d3Connectivity {
    type FaceConnectivity = ();

    fn num_faces(&self) -> usize {
        0
    }

    fn get_face_connectivity(&self, _index: usize) -> Option<Self::FaceConnectivity> {
        None
    }

    fn vertex_indices(&self) -> &[usize] {
        &self.0
    }
}

impl ConnectivityMut for Segment3d3Connectivity {
    fn vertex_indices_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }
}

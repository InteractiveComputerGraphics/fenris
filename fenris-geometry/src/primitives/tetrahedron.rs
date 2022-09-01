use crate::{
    AxisAlignedBoundingBox, BoundedGeometry, ConvexPolyhedron, Distance, OrientationTestResult, Triangle, Triangle3d,
};
use fenris_traits::Real;
use nalgebra::{OPoint, Point3, Scalar, U3};
use numeric_literals::replace_float_literals;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Copy, Clone, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound(serialize = "Point3<T>: Serialize"))]
#[serde(bound(deserialize = "Point3<T>: Deserialize<'de>"))]
pub struct Tetrahedron<T>
where
    T: Scalar,
{
    // Ordering uses same conventions as Tet4Connectivity
    vertices: [Point3<T>; 4],
}

impl<T> Tetrahedron<T>
where
    T: Scalar,
{
    /// Construct tetrahedron from the given points.
    ///
    /// Ordering is the same as for `Tet4Connectivity`.
    pub fn from_vertices(vertices: [Point3<T>; 4]) -> Self {
        Self { vertices }
    }
}

impl<T> Tetrahedron<T>
where
    T: Real,
{
    /// Reference tetrahedron.
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn reference() -> Self {
        Self::from_vertices([
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, -1.0, -1.0),
            Point3::new(-1.0, 1.0, -1.0),
            Point3::new(-1.0, -1.0, 1.0),
        ])
    }
}

impl<T: Real> BoundedGeometry<T> for Tetrahedron<T> {
    type Dimension = U3;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, U3> {
        AxisAlignedBoundingBox::from_points(&self.vertices).unwrap()
    }
}

impl<'a, T> ConvexPolyhedron<'a, T> for Tetrahedron<T>
where
    T: Real,
{
    type Face = Triangle3d<T>;

    fn num_faces(&self) -> usize {
        4
    }

    fn get_face(&self, index: usize) -> Option<Self::Face> {
        let v = &self.vertices;
        let tri = |i, j, k| Some(Triangle([v[i], v[j], v[k]]));

        // Must choose faces carefully so that they point towards the interior
        match index {
            0 => tri(0, 1, 2),
            1 => tri(0, 3, 1),
            2 => tri(1, 3, 2),
            3 => tri(0, 2, 3),
            _ => None,
        }
    }
}

impl<T> Distance<T, Point3<T>> for Tetrahedron<T>
where
    T: Real,
{
    fn distance(&self, point: &OPoint<T, U3>) -> T {
        let triangle = |i, j, k| Triangle([self.vertices[i], self.vertices[j], self.vertices[k]]);

        let tri_faces = [
            // We must carefully choose the ordering of vertices so that the
            // resulting faces have outwards-pointing normals
            triangle(2, 1, 0),
            triangle(1, 2, 3),
            triangle(0, 1, 3),
            triangle(2, 0, 3),
        ];

        let mut point_inside = true;
        // TODO: Fix unwrap
        let mut min_dist = T::max_value().unwrap();

        for tri_face in &tri_faces {
            // Remember that the triangles are oriented such that *outwards* is the positive
            // direction, so for the point to be inside of the cell, we need its orientation
            // with respect to each face to be *negative*
            if tri_face.point_orientation(point) == OrientationTestResult::Positive {
                point_inside = false;
            }

            min_dist = T::min(min_dist, tri_face.distance(point));
        }

        if point_inside {
            T::zero()
        } else {
            min_dist
        }
    }
}

use crate::{
    AxisAlignedBoundingBox, BoundedGeometry, ConvexPolyhedron, Distance, Quad3d, SignedDistance, SignedDistanceResult,
};
use fenris_traits::Real;
use nalgebra::{OPoint, Point3, Scalar, U3};
use numeric_literals::replace_float_literals;

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub struct Hexahedron<T>
where
    T: Scalar,
{
    // Ordering uses same conventions as Hex8Connectivity
    vertices: [Point3<T>; 8],
}

impl<T> BoundedGeometry<T> for Hexahedron<T>
where
    T: Real,
{
    type Dimension = U3;

    fn bounding_box(&self) -> AxisAlignedBoundingBox<T, U3> {
        AxisAlignedBoundingBox::from_points(&self.vertices).unwrap()
    }
}

impl<T> Hexahedron<T>
where
    T: Scalar,
{
    pub fn from_vertices(vertices: [Point3<T>; 8]) -> Self {
        Self { vertices }
    }
}

impl<T> Hexahedron<T>
where
    T: Real,
{
    #[replace_float_literals(T::from_f64(literal).unwrap())]
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

impl<T> Distance<T, Point3<T>> for Hexahedron<T>
where
    T: Real,
{
    fn distance(&self, point: &Point3<T>) -> T {
        let signed_dist = self.compute_signed_distance(point).signed_distance;
        T::max(signed_dist, T::zero())
    }
}

impl<T> SignedDistance<T, U3> for Hexahedron<T>
where
    T: Real,
{
    fn query_signed_distance(&self, point: &OPoint<T, U3>) -> Option<SignedDistanceResult<T, U3>> {
        Some(self.compute_signed_distance(point))
    }
}

impl<'a, T> ConvexPolyhedron<'a, T> for Hexahedron<T>
where
    T: Real,
{
    type Face = Quad3d<T>;

    fn num_faces(&self) -> usize {
        6
    }

    fn get_face(&self, index: usize) -> Option<Self::Face> {
        let v = &self.vertices;
        let quad = |i, j, k, l| Some(Quad3d::from_vertices([v[i], v[j], v[k], v[l]]));

        // Must choose faces carefully so that they point towards the interior
        match index {
            0 => quad(0, 1, 2, 3),
            1 => quad(4, 5, 1, 0),
            2 => quad(5, 6, 2, 1),
            3 => quad(6, 7, 3, 2),
            4 => quad(0, 3, 7, 4),
            5 => quad(4, 7, 6, 5),
            _ => None,
        }
    }
}

use crate::{
    AxisAlignedBoundingBox2d, BoundedGeometry, ConvexPolygon3d, Distance, SimplePolygon2d, Triangle, Triangle2d,
};
use fenris_traits::Real;
use itertools::izip;
use nalgebra::{Point2, Point3, Scalar, U2};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Quad3d<T: Scalar> {
    vertices: [Point3<T>; 4],
}

impl<T: Scalar> Quad3d<T> {
    pub fn from_vertices(vertices: [Point3<T>; 4]) -> Self {
        Self { vertices }
    }
}

impl<'a, T> ConvexPolygon3d<'a, T> for Quad3d<T>
where
    T: Scalar,
{
    fn num_vertices(&self) -> usize {
        4
    }

    fn get_vertex(&self, index: usize) -> Option<Point3<T>> {
        self.vertices.get(index).cloned()
    }
}

impl<T> BoundedGeometry<T> for Quad2d<T>
where
    T: Real,
{
    type Dimension = U2;

    fn bounding_box(&self) -> AxisAlignedBoundingBox2d<T> {
        AxisAlignedBoundingBox2d::from_points(&self.0).expect("Triangle always has > 0 vertices")
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// A quadrilateral consisting of four vertices, assumed to be specified in counter-clockwise
/// winding order.
pub struct Quad2d<T: Scalar>(pub [Point2<T>; 4]);

impl<T> Quad2d<T>
where
    T: Real,
{
    /// Returns the index of a concave corner of the quadrilateral, if there is any.
    pub fn concave_corner(&self) -> Option<usize> {
        for i in 0..4 {
            let x_next = self.0[(i + 2) % 4];
            let x_curr = self.0[(i + 1) % 4];
            let x_prev = self.0[(i + 1) % 4];

            let a = x_next - x_curr;
            let b = x_prev - x_curr;
            // perp gives "2d cross product", which when negative means that the interior angle
            // is creater than 180 degrees, and so the corner must be concave
            if a.perp(&b) < T::zero() {
                return Some(i + 1);
            }
        }

        None
    }

    /// Splits the quad into two triangles represented by local indices { 0, 1, 2, 3 }
    /// which correspond to the quad's vertices.
    ///
    /// While the quad may be concave, it is assumed that it has no self-intersections and that
    /// all vertices are unique.
    pub fn split_into_triangle_connectivities(&self) -> ([usize; 3], [usize; 3]) {
        if let Some(concave_corner_index) = self.concave_corner() {
            let i = concave_corner_index;
            let triangle1 = [(i + 2) % 4, (i + 3) % 4, (i + 0) % 4];
            let triangle2 = [(i + 2) % 4, (i + 0) % 4, (i + 1) % 4];
            (triangle1, triangle2)
        } else {
            // Split arbitrarily, but in a regular fashion
            let triangle1 = [0, 1, 2];
            let triangle2 = [0, 2, 3];
            (triangle1, triangle2)
        }
    }

    pub fn split_into_triangles(&self) -> (Triangle2d<T>, Triangle2d<T>) {
        let (conn1, conn2) = self.split_into_triangle_connectivities();
        let mut vertices1 = [Point2::origin(); 3];
        let mut vertices2 = [Point2::origin(); 3];

        for (v, idx) in izip!(&mut vertices1, &conn1) {
            *v = self.0[*idx];
        }

        for (v, idx) in izip!(&mut vertices2, &conn2) {
            *v = self.0[*idx];
        }

        let tri1 = Triangle(vertices1);
        let tri2 = Triangle(vertices2);

        (tri1, tri2)
    }

    pub fn area(&self) -> T {
        let (tri1, tri2) = self.split_into_triangles();
        tri1.area() + tri2.area()
    }
}

impl<T> Distance<T, Point2<T>> for Quad2d<T>
where
    T: Real,
{
    fn distance(&self, point: &Point2<T>) -> T {
        // TODO: Avoid heap allocation
        SimplePolygon2d::from_vertices(self.0.to_vec()).distance(point)
    }
}

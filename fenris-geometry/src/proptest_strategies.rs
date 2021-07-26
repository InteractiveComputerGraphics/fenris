// use crate::procedural::create_rectangular_uniform_quad_mesh_2d;
use crate::{LineSegment2d, Orientation, Quad2d, Triangle, Triangle2d, Triangle3d};

use crate::Orientation::Counterclockwise;
use nalgebra::{Point2, Point3, Vector2};
use proptest::prelude::*;

// TODO: This is just copied from fenris to prevent unneeded coupling for the time being
fn point2() -> impl Strategy<Value = Point2<f64>> {
    // Pick a reasonably small range to pick coordinates from,
    // otherwise we can easily get floating point numbers that are
    // so ridiculously large as to break anything we might want to do with them
    let range = -10.0..10.0;
    [range.clone(), range.clone()].prop_map(|[x, y]| Point2::new(x, y))
}

// TODO: This is just copied from fenris to prevent unneeded coupling for the time being
fn point3() -> impl Strategy<Value = Point3<f64>> {
    // Pick a reasonably small range to pick coordinates from,
    // otherwise we can easily get floating point numbers that are
    // so ridiculously large as to break anything we might want to do with them
    let range = -10.0..10.0;
    [range.clone(), range.clone(), range.clone()].prop_map(|[x, y, z]| Point3::new(x, y, z))
}

#[derive(Debug, Clone)]
pub struct Triangle3dParams {
    orientation: Orientation,
}

impl Triangle3dParams {
    pub fn with_orientation(self, orientation: Orientation) -> Self {
        Self { orientation, ..self }
    }
}

impl Default for Triangle3dParams {
    fn default() -> Self {
        Self {
            orientation: Counterclockwise,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Triangle2dParams {
    orientation: Orientation,
}

impl Triangle2dParams {
    pub fn with_orientation(self, orientation: Orientation) -> Self {
        Self { orientation, ..self }
    }
}

impl Default for Triangle2dParams {
    fn default() -> Self {
        Self {
            orientation: Counterclockwise,
        }
    }
}

impl Arbitrary for Triangle3d<f64> {
    // TODO: Parameter for extents (i.e. bounding box or so)
    type Parameters = Triangle3dParams; // TODO: Avoid boxing for performance...?
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let points = [point3(), point3(), point3()];
        points
            .prop_map(|points| Triangle(points))
            .prop_map(move |mut triangle| {
                if triangle.orientation() != args.orientation {
                    triangle.swap_vertices(0, 1);
                }
                triangle
            })
            .boxed()
    }
}

impl Arbitrary for Triangle2d<f64> {
    // TODO: Parameter for extents (i.e. bounding box or so)
    type Parameters = Triangle2dParams; // TODO: Avoid boxing for performance...?
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let points = [point2(), point2(), point2()];
        points
            .prop_map(|points| Triangle(points))
            .prop_map(move |mut triangle| {
                if triangle.orientation() != args.orientation {
                    triangle.swap_vertices(0, 1);
                }
                triangle
            })
            .boxed()
    }
}

pub fn triangle2d_strategy_f64() -> impl Strategy<Value = Triangle2d<f64>> {
    [point2(), point2(), point2()].prop_map(|points| Triangle(points))
}

pub fn clockwise_triangle2d_strategy_f64() -> impl Strategy<Value = Triangle2d<f64>> {
    triangle2d_strategy_f64().prop_map(|mut triangle| {
        if triangle.orientation() != Orientation::Clockwise {
            triangle.swap_vertices(0, 2);
        }
        triangle
    })
}

pub fn nondegenerate_line_segment2d_strategy_f64() -> impl Strategy<Value = LineSegment2d<f64>> {
    // Make sure to construct the second point from non-zero components
    let gen = prop_oneof![0.5..3.5, -0.5..3.5, 1e-6..10.0, -10.0..-1e-6];
    (point2(), gen.clone(), gen).prop_map(|(a, x, y)| {
        let d = Vector2::new(x, y);
        let b = a + d;
        LineSegment2d::new(a, b)
    })
}

/// A strategy for triangles that are oriented clockwise and not degenerate
/// (i.e. collapsed to a line, area
pub fn nondegenerate_triangle2d_strategy_f64() -> impl Strategy<Value = Triangle2d<f64>> {
    let segment = nondegenerate_line_segment2d_strategy_f64();
    let t1_gen = prop_oneof![-3.0..3.0, -10.0..10.0];
    let t2_gen = prop_oneof![0.5..3.0, 1e-6..10.0];
    (segment, t1_gen, t2_gen).prop_map(|(segment, t1, t2)| {
        let a = segment.from();
        let b = segment.to();
        let ab = b - a;
        let n = Vector2::new(-ab.y, ab.x);
        let c = Point2::from(a + t1 * ab + t2 * n);
        Triangle([*a, *b, c])
    })
}

fn extrude_triangle_to_convex_quad(triangle: &Triangle2d<f64>, t1: f64, t3: f64) -> Quad2d<f64> {
    // In order to generate a convex quad, we first generate one triangle,
    // then we "extrude" a vertex from one of the sides of the triangle, in such a way
    // that the vertex is contained in the convex cone defined by the two other sides,
    // constrained to lie on the side itself or further away.
    // The result is a convex quad.
    let t2 = 1.0 - t1;
    assert!(t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t3 >= 0.0);
    let a = &triangle.0[0];
    let b = &triangle.0[1];
    let c = &triangle.0[2];
    let d1 = b - a;
    let d2 = c - a;
    // Define a vector d3 pointing from a to a point on the opposite edge
    let d3_hat = t1 * d1 + t2 * d2;
    // Choose a parameter t3 >= 0. Then (1 + t3) * d3_hat is a vector pointing from a to the new
    // point
    let d3 = (1.0 + t3) * d3_hat;

    Quad2d([*a, *b, a + d3, *c])
}

pub fn convex_quad2d_strategy_f64() -> impl Strategy<Value = Quad2d<f64>> {
    let t1_gen = 0.0..=1.0;
    let t3_gen = 0.0..10.0;
    (t1_gen, t3_gen, clockwise_triangle2d_strategy_f64())
        .prop_map(|(t1, t3, triangle)| extrude_triangle_to_convex_quad(&triangle, t1, t3))
}

pub fn nondegenerate_convex_quad2d_strategy_f64() -> impl Strategy<Value = Quad2d<f64>> {
    let t1_gen = prop_oneof![0.25..=0.75, 1e-6..=(1.0 - 1e-6)];
    let t3_gen = prop_oneof![0.5..3.0, 1e-6..10.0];
    (t1_gen, t3_gen, nondegenerate_triangle2d_strategy_f64())
        .prop_map(|(t1, t3, triangle)| extrude_triangle_to_convex_quad(&triangle, t1, t3))
}

pub fn parallelogram_strategy_f64() -> impl Strategy<Value = Quad2d<f64>> {
    nondegenerate_triangle2d_strategy_f64().prop_map(|triangle| {
        let a = &triangle.0[0];
        let b = &triangle.0[1];
        let d = &triangle.0[2];
        let ab = b - a;
        let ad = d - a;
        let c = a + ab + ad;
        Quad2d([*a, *b, c, *d])
    })
}

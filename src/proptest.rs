use crate::element::{Tet4Element, Tri3d2Element};
use crate::geometry::proptest::Triangle3dParams;
use crate::geometry::Orientation::Counterclockwise;
use crate::mesh::procedural::create_rectangular_uniform_quad_mesh_2d;
use crate::mesh::QuadMesh2d;
use ::proptest::prelude::*;
use fenris_geometry::proptest::Triangle2dParams;
use fenris_geometry::{Triangle2d, Triangle3d};
use nalgebra::{Point2, Point3, Vector2};
use std::cmp::max;

pub fn point2() -> impl Strategy<Value = Point2<f64>> {
    // Pick a reasonably small range to pick coordinates from,
    // otherwise we can easily get floating point numbers that are
    // so ridiculously large as to break anything we might want to do with them
    let range = -10.0..10.0;
    [range.clone(), range.clone()].prop_map(|[x, y]| Point2::new(x, y))
}

pub fn point3() -> impl Strategy<Value = Point3<f64>> {
    // Pick a reasonably small range to pick coordinates from,
    // otherwise we can easily get floating point numbers that are
    // so ridiculously large as to break anything we might want to do with them
    let range = -10.0..10.0;
    [range.clone(), range.clone(), range.clone()].prop_map(|[x, y, z]| Point3::new(x, y, z))
}

impl Arbitrary for Tri3d2Element<f64> {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        any_with::<Triangle2d<f64>>(Triangle2dParams::default().with_orientation(Counterclockwise))
            .prop_map(|triangle| Self::from(triangle))
            .boxed()
    }
}

impl Arbitrary for Tet4Element<f64> {
    // TODO: Reasonable parameters?
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        any_with::<Triangle3d<f64>>(Triangle3dParams::default().with_orientation(Counterclockwise))
            .prop_flat_map(|triangle| {
                // To create an arbitrary tetrahedron element, we take a counter-clockwise oriented
                // triangle, and pick a point somewhere on the "positive" side of the
                // triangle plane. We do this by associating a parameter with each
                // tangent vector defined by the sides of the triangle,
                // plus a non-negative parameter that scales along the normal direction
                let range = -10.0..10.0;
                let tangent_params = [range.clone(), range.clone(), range.clone()];
                let normal_param = 0.0..=10.0;
                (Just(triangle), tangent_params, normal_param)
            })
            .prop_map(|(triangle, tangent_params, normal_param)| {
                let mut tangent_pos = triangle.centroid();

                for (side, param) in triangle.sides().iter().zip(&tangent_params) {
                    tangent_pos.coords += *param * side;
                }
                let coord = tangent_pos + normal_param * triangle.normal_dir().normalize();
                Tet4Element::from_vertices([triangle.0[0], triangle.0[1], triangle.0[2], coord])
            })
            .boxed()
    }
}

// Returns a strategy in which each value is a triplet (cells_per_unit, units_x, units_y)
// such that cells_per_unit^2 * units_x * units_y <= max_cells
fn rectangular_uniform_mesh_cell_distribution_strategy(
    max_cells: usize,
) -> impl Strategy<Value = (usize, usize, usize)> {
    let max_cells_per_unit = f64::floor(f64::sqrt(max_cells as f64)) as usize;
    (1..=max(1, max_cells_per_unit))
        .prop_flat_map(move |cells_per_unit| (Just(cells_per_unit), 0..=max_cells / (cells_per_unit * cells_per_unit)))
        .prop_flat_map(move |(cells_per_unit, units_x)| {
            let units_y_strategy = 0..=max_cells / (cells_per_unit * cells_per_unit * max(1, units_x));
            (Just(cells_per_unit), Just(units_x), units_y_strategy)
        })
}

pub fn rectangular_uniform_mesh_strategy(unit_length: f64, max_cells: usize) -> impl Strategy<Value = QuadMesh2d<f64>> {
    rectangular_uniform_mesh_cell_distribution_strategy(max_cells).prop_map(
        move |(cells_per_unit, units_x, units_y)| {
            create_rectangular_uniform_quad_mesh_2d(
                unit_length,
                units_x,
                units_y,
                cells_per_unit,
                &Vector2::new(0.0, 0.0),
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::rectangular_uniform_mesh_cell_distribution_strategy;
    use crate::geometry::proptest::{
        convex_quad2d_strategy_f64, nondegenerate_convex_quad2d_strategy_f64, nondegenerate_triangle2d_strategy_f64,
    };
    use crate::geometry::Orientation;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn rectangular_uniform_mesh_cell_distribution_strategy_respects_max_cells(
            (max_cells, cells_per_unit, units_x, units_y)
             in (0..20usize).prop_flat_map(|max_cells| {
                rectangular_uniform_mesh_cell_distribution_strategy(max_cells)
                    .prop_map(move |(cells_per_unit, units_x, units_y)| {
                    (max_cells, cells_per_unit, units_x, units_y)
                })
             })
        ) {
            // Test that the distribution strategy for rectangular meshes
            // respects the maximum number of cells given
            prop_assert!(cells_per_unit * cells_per_unit * units_x * units_y <= max_cells);
        }

        #[test]
        fn convex_quads_are_convex(quad in convex_quad2d_strategy_f64()) {
            prop_assert!(quad.concave_corner().is_none());
        }

        #[test]
        fn nondegenerate_triangles_have_positive_area(
            triangle in nondegenerate_triangle2d_strategy_f64()
        ){
            prop_assert!(triangle.area() > 0.0);
            prop_assert!(triangle.orientation() == Orientation::Counterclockwise);
        }

        #[test]
        fn nondegenerate_quads_have_positive_area(
            quad in nondegenerate_convex_quad2d_strategy_f64()
        ){
            prop_assert!(quad.area() > 0.0);
        }
    }
}

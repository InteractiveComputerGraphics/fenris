//! Quadrature rules constructed by subdividing the reference domain.
use crate::element::{FiniteElement, Tri3d2Element};
use crate::integrate::volume_form;
use crate::quadrature::{
    BorrowedQuadratureParts, Quadrature, Quadrature1d, Quadrature2d, QuadraturePair1d, QuadraturePair2d,
};
use itertools::izip;
use nalgebra::{point, vector, Point1, Point2, RealField, U2};
use numeric_literals::replace_float_literals;

/// Construct a univariate quadrature rule by subdivision.
///
/// This function constructs a quadrature rule for the univariate reference domain by subdividing the domain
/// into the prescribed number of pieces and applying the given quadrature rule in each subdivision.
/// The resulting weights and points are transformed back to the reference domain, thereby constructing
/// an aggregate quadrature rule from the individual pieces.
pub fn subdivide_univariate<T>(quadrature: impl Quadrature1d<T>, subdivision_pieces: usize) -> QuadraturePair1d<T>
where
    T: RealField,
{
    subdivide_univariate_(quadrature.weights(), quadrature.points(), subdivision_pieces)
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn subdivide_univariate_<T>(
    reference_weights: &[T],
    reference_points: &[Point1<T>],
    subdivision_pieces: usize,
) -> QuadraturePair1d<T>
where
    T: RealField,
{
    let mut points = Vec::new();
    let mut weights = Vec::new();

    let pieces_as_scalar = T::from_usize(subdivision_pieces)
        // This should never panic, because in the worst case, it gets truncated. However, it is
        // possible that for a custom/niche "real" type, the method returns `None`.
        .expect("Internal error: Failed to convert usize to scalar type");

    let subdivision_size = 2.0 / pieces_as_scalar;
    for i in 0..subdivision_pieces {
        // This should never panic, because we would likely have already panicked in the previous integer -> scalar
        // conversion.
        let i = T::from_usize(i).expect("Internal error: Failed to convert usize to scalar type.");
        let a = i * subdivision_size - 1.0;
        let b = a + subdivision_size;
        let jacobian = (b - a) / 2.0;

        for (ref_weight, ref_point) in reference_weights.iter().zip(reference_points) {
            let weight = ref_weight.clone() * jacobian;
            let point = ((b - a) * ref_point[0] + (b + a)) / 2.0;
            weights.push(weight);
            points.push(Point1::new(point));
        }
    }

    (weights, points)
}

/// Constructs a new quadrature rule for the reference triangle by subdividing the given
/// base quadrature.
///
/// The domain $[-1, 1]^2$ is subdivided into a regular grid with the supplied number of
/// subdivisions per axis. Each cell is divided into two triangles, and only the triangles
/// that make up part of the reference triangle are kept. The base quadrature is then
/// applied to each individual triangle in the subdivision and combined to form a single quadrature
/// rule for the reference element.
///
/// # Panics
///
/// Panics if `subdivisions == 0`.
pub fn subdivide_triangle<T>(quadrature: impl Quadrature2d<T, Data = ()>, subdivisions: usize) -> QuadraturePair2d<T>
where
    T: RealField,
{
    subdivide_triangle_(quadrature.to_parts(), subdivisions)
}

#[replace_float_literals(T::from_f64(literal).unwrap())]
fn subdivide_triangle_<T>(
    base_quadrature: BorrowedQuadratureParts<T, U2, ()>,
    subdivisions: usize,
) -> QuadraturePair2d<T>
where
    T: RealField,
{
    assert!(
        subdivisions > 0,
        "Number of subdivisions must be greater or equal to 1."
    );
    let cell_size = 2.0 / T::from_usize(subdivisions).unwrap();
    // TODO: Reserve space since we know size of final quadrature rule
    let mut quadrature = QuadraturePair2d::default();
    // Loop over cells in an implicit regular grid. Each (i, j) coordinate corresponds to a cell
    // in the grid, where i denotes a row and j denotes a column in a matrix-like numbering system
    for i in 0..subdivisions {
        // Loop over lower triangle of the implicit grid
        for j in 0..=i {
            let t_i = T::from_usize(i).unwrap();
            let t_j = T::from_usize(j).unwrap();
            let cell_center = point![-1.0 + cell_size * (t_j + 0.5), 1.0 - cell_size * (t_i + 0.5)];

            let cell_vertices = [
                &cell_center + vector![-1.0, -1.0] * 0.5 * cell_size,
                &cell_center + vector![1.0, -1.0] * 0.5 * cell_size,
                &cell_center + vector![1.0, 1.0] * 0.5 * cell_size,
                &cell_center + vector![-1.0, 1.0] * 0.5 * cell_size,
            ];

            let lower_verts = [cell_vertices[0], cell_vertices[1], cell_vertices[3]];
            let upper_verts = [cell_vertices[1], cell_vertices[2], cell_vertices[3]];

            add_triangle_quadrature(&mut quadrature, lower_verts, base_quadrature.to_parts());

            // For the implicit cells on the diagonal, we only want to keep the lower half
            // that is actually contained inside the reference triangle
            if i != j {
                add_triangle_quadrature(&mut quadrature, upper_verts, base_quadrature.to_parts());
            }
        }
    }
    quadrature
}

fn add_triangle_quadrature<T: RealField>(
    quadrature: &mut QuadraturePair2d<T>,
    triangle_vertices: [Point2<T>; 3],
    base_quadrature: BorrowedQuadratureParts<T, U2, ()>,
) {
    // Use the transformation functionality of the linear triangle finite element
    let triangle = Tri3d2Element::from_vertices(triangle_vertices);
    for (w_base, xi_base) in izip!(base_quadrature.weights(), base_quadrature.points()) {
        let x = triangle.map_reference_coords(xi_base);
        let j = triangle.reference_jacobian(xi_base);
        let w = volume_form(&j) * w_base.clone();
        quadrature.0.push(w);
        quadrature.1.push(x);
    }
}

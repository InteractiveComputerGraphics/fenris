//! Basic procedural mesh generation routines.
use crate::connectivity::{Hex8Connectivity, Quad4d2Connectivity, Tet4Connectivity};
use crate::geometry::polymesh::PolyMesh3d;
use crate::geometry::sdf::BoundedSdf;
use crate::geometry::{AxisAlignedBoundingBox2d, HalfSpace};
use crate::mesh::{HexMesh, Mesh, QuadMesh2d, Tet4Mesh, TriangleMesh2d};
use crate::Real;
use itertools::{iproduct, Itertools};
use nalgebra::{convert, point, try_convert, vector, Point2, Point3, Unit, Vector2, Vector3};
use numeric_literals::replace_float_literals;
use ordered_float::NotNan;
use std::cmp::min;
use std::f64::consts::PI;

pub fn create_unit_square_uniform_quad_mesh_2d<T>(cells_per_dim: usize) -> QuadMesh2d<T>
where
    T: Real,
{
    create_rectangular_uniform_quad_mesh_2d(T::one(), 1, 1, cells_per_dim, &Vector2::new(T::zero(), T::one()))
}

pub fn create_unit_square_uniform_tri_mesh_2d<T>(cells_per_dim: usize) -> TriangleMesh2d<T>
where
    T: Real,
{
    create_rectangular_uniform_quad_mesh_2d(T::one(), 1, 1, cells_per_dim, &Vector2::new(T::zero(), T::one()))
        .split_into_triangles()
}

pub fn create_unit_box_uniform_hex_mesh_3d<T>(cells_per_dim: usize) -> HexMesh<T>
where
    T: Real,
{
    create_rectangular_uniform_hex_mesh(T::one(), 1, 1, 1, cells_per_dim)
}

pub fn create_unit_box_uniform_tet_mesh_3d<T>(cells_per_dim: usize) -> Tet4Mesh<T>
where
    T: Real,
{
    let hex_mesh = create_unit_box_uniform_hex_mesh_3d(cells_per_dim);
    Tet4Mesh::from(&hex_mesh)
}

/// Generates an axis-aligned rectangular uniform mesh given a unit length,
/// dimensions as multipliers of the unit length and the number of cells per unit length.
pub fn create_rectangular_uniform_quad_mesh_2d<T>(
    unit_length: T,
    units_x: usize,
    units_y: usize,
    cells_per_unit: usize,
    top_left: &Vector2<T>,
) -> QuadMesh2d<T>
where
    T: Real,
{
    if cells_per_unit == 0 || units_x == 0 || units_y == 0 {
        QuadMesh2d::from_vertices_and_connectivity(Vec::new(), Vec::new())
    } else {
        let mut vertices = Vec::new();
        let mut cells = Vec::new();

        let cell_size = T::from_f64(unit_length.to_subset().unwrap() / cells_per_unit as f64).unwrap();
        let num_cells_x = units_x * cells_per_unit;
        let num_cells_y = units_y * cells_per_unit;
        let num_vertices_x = num_cells_x + 1;
        let num_vertices_y = num_cells_y + 1;

        let to_global_vertex_index = |i, j| (num_cells_x + 1) * j + i;

        for j in 0..num_vertices_y {
            for i in 0..num_vertices_x {
                let i_as_t = T::from_usize(i).expect("Must be able to fit usize in T");
                let j_as_t = T::from_usize(j).expect("Must be able to fit usize in T");
                let v = top_left + Vector2::new(i_as_t, -j_as_t) * cell_size;
                vertices.push(Point2::from(v));
            }
        }

        for j in 0..num_cells_y {
            for i in 0..num_cells_x {
                let quad = Quad4d2Connectivity([
                    to_global_vertex_index(i, j + 1),
                    to_global_vertex_index(i + 1, j + 1),
                    to_global_vertex_index(i + 1, j),
                    to_global_vertex_index(i, j),
                ]);
                cells.push(quad);
            }
        }

        QuadMesh2d::from_vertices_and_connectivity(vertices, cells)
    }
}

#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
pub fn voxelize_bounding_box_2d<T>(bounds: &AxisAlignedBoundingBox2d<T>, max_cell_size: T) -> QuadMesh2d<T>
where
    T: Real,
{
    assert!(max_cell_size > T::zero(), "Max cell size must be positive.");

    let extents = bounds.extents();
    let enlarged_bounds = AxisAlignedBoundingBox2d::new(bounds.min() - extents * 0.01, bounds.max() + extents * 0.01);
    let enlarged_extents = enlarged_bounds.extents();

    // Determine the minimum number of cells needed in each direction to completely cover
    // the enlarged bounding box. We do this in double precision
    let enlarged_extents_f64: Vector2<f64> = try_convert(enlarged_extents).expect("Must be able to fit extents in f64");
    let resolution_f64: f64 = try_convert(max_cell_size).expect("Must be able to fit resolution in f64");

    let candidate_num_cells_x = (enlarged_extents_f64.x / resolution_f64).ceil();
    let candidate_num_cells_y = (enlarged_extents_f64.y / resolution_f64).ceil();
    let candidate_cell_size_x = enlarged_extents_f64.x / candidate_num_cells_x;
    let candidate_cell_size_y = enlarged_extents_f64.y / candidate_num_cells_y;
    let cell_size_f64 = min(
        NotNan::new(candidate_cell_size_x).unwrap(),
        NotNan::new(candidate_cell_size_y).unwrap(),
    )
    .into_inner();

    let num_cells_x = (enlarged_extents_f64.x / cell_size_f64).ceil();
    let num_cells_y = (enlarged_extents_f64.y / cell_size_f64).ceil();
    let final_extents_x = num_cells_x * cell_size_f64;
    let final_extents_y = num_cells_y * cell_size_f64;
    let final_extents: Vector2<T> = Vector2::new(convert(final_extents_x), convert(final_extents_y));

    let center = bounds.center();
    let top_left = Vector2::new(center.x - final_extents.x / 2.0, center.y + final_extents.y / 2.0);

    create_rectangular_uniform_quad_mesh_2d(
        convert(cell_size_f64),
        num_cells_x as usize,
        num_cells_y as usize,
        1,
        &top_left,
    )
}

#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
pub fn voxelize_sdf_2d<T>(sdf: &impl BoundedSdf<T>, max_cell_size: T) -> QuadMesh2d<T>
where
    T: Real,
{
    let rectangular_mesh: QuadMesh2d<T> = voxelize_bounding_box_2d(&sdf.bounding_box(), max_cell_size);
    let desired_cell_indices: Vec<_> = rectangular_mesh
        .cell_iter()
        .enumerate()
        .filter(|(_, quad)| quad.0.iter().any(|v| sdf.eval(v) <= T::zero()))
        .map(|(cell_index, _)| cell_index)
        .collect();

    rectangular_mesh.keep_cells(&desired_cell_indices)
}

#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
pub fn approximate_quad_mesh_for_sdf_2d<T>(sdf: &impl BoundedSdf<T>, max_cell_size: T) -> QuadMesh2d<T>
where
    T: Real,
{
    let mut mesh = voxelize_sdf_2d(sdf, max_cell_size);

    mesh.transform_vertices(|vertex| {
        let phi = sdf.eval(vertex);
        if phi > 0.0 {
            let grad = sdf.gradient(vertex).expect("TODO: Fix when no gradient");
            let new_vertex = &*vertex - grad * phi;
            *vertex = new_vertex;
        }
    });

    mesh
}

pub fn approximate_triangle_mesh_for_sdf_2d<T>(sdf: &impl BoundedSdf<T>, max_cell_size: T) -> TriangleMesh2d<T>
where
    T: Real,
{
    // TODO: This is not the most efficient way to do this, since we compute SDFs for the same points
    // several times, but it's at least simple
    let mut mesh = voxelize_sdf_2d(sdf, max_cell_size).split_into_triangles();

    // Remove triangle that fall completely outside (this is also done by voxelize_sdf for quads,
    // but it may be that after splitting that some triangles still fall completely outside
    let desired_cell_indices: Vec<_> = mesh
        .cell_iter()
        .enumerate()
        .filter(|(_, triangle)| {
            // TODO: Implement better criteria for filtering triangles. By only checking
            // values at SDF, we may discard triangles that are intersected by the SDF but
            // no vertices of the triangle are inside of the shape
            triangle.0.iter().any(|v| sdf.eval(v) <= T::zero())
        })
        .map(|(cell_index, _)| cell_index)
        .collect();

    mesh = mesh.keep_cells(&desired_cell_indices);

    mesh.transform_vertices(|vertex| {
        let phi = sdf.eval(vertex);
        if phi > T::zero() {
            let grad = sdf.gradient(vertex).expect("TODO: Fix when no gradient");
            let new_vertex = &*vertex - grad * phi;
            *vertex = new_vertex;
        }
    });

    mesh
}

/// Generates an axis-aligned rectangular uniform three-dimensional hex mesh given a unit length,
/// dimensions as multipliers of the unit length and the number of cells per unit length.
///
/// The resulting box is given by the set `[0, u * ux] x [0, u * uy] x [0, u * uz]`
/// where u denotes the unit length, ux, uy and uz denote the number of units along each
/// coordinate axis.
pub fn create_rectangular_uniform_hex_mesh<T>(
    unit_length: T,
    units_x: usize,
    units_y: usize,
    units_z: usize,
    cells_per_unit: usize,
) -> HexMesh<T>
where
    T: Real,
{
    if cells_per_unit == 0 || units_x == 0 || units_y == 0 {
        HexMesh::from_vertices_and_connectivity(Vec::new(), Vec::new())
    } else {
        let mut vertices = Vec::new();
        let mut cells = Vec::new();

        let cell_size = T::from_f64(unit_length.to_subset().unwrap() / cells_per_unit as f64).unwrap();

        let num_cells_x = units_x * cells_per_unit;
        let num_cells_y = units_y * cells_per_unit;
        let num_cells_z = units_z * cells_per_unit;
        let num_vertices_x = num_cells_x + 1;
        let num_vertices_y = num_cells_y + 1;
        let num_vertices_z = num_cells_z + 1;

        let to_global_vertex_index =
            |i: usize, j: usize, k: usize| (num_vertices_x * num_vertices_y) * k + (num_vertices_x) * j + i;

        for k in 0..num_vertices_z {
            for j in 0..num_vertices_y {
                for i in 0..num_vertices_x {
                    let v = Point3::new(
                        T::from_usize(i).unwrap() * cell_size,
                        T::from_usize(j).unwrap() * cell_size,
                        T::from_usize(k).unwrap() * cell_size,
                    );
                    vertices.push(v);
                }
            }
        }

        for k in 0..num_cells_z {
            for j in 0..num_cells_y {
                for i in 0..num_cells_x {
                    let idx = &to_global_vertex_index;
                    cells.push(Hex8Connectivity([
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j + 1, k),
                        idx(i, j + 1, k),
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i + 1, j + 1, k + 1),
                        idx(i, j + 1, k + 1),
                    ]));
                }
            }
        }

        Mesh::from_vertices_and_connectivity(vertices, cells)
    }
}

/// Creates a rectangular uniform tetrahedral mesh.
///
/// The implementation uses a BCC lattice, where each pair of adjacent cell centers
/// along each coordinate direction is connected by an octahedron, which is further subdivided
/// into four tetrahedra along the edge between the two cell centers. The boundaries of the
/// cuboidal domain are finally filled with pyramids that are subdivided into two tetrahedra.
#[replace_float_literals(T::from_f64(literal).unwrap())]
pub fn create_rectangular_uniform_tet_mesh<T>(
    unit_length: T,
    units_x: usize,
    units_y: usize,
    units_z: usize,
    cells_per_unit: usize,
) -> Tet4Mesh<T>
where
    T: Real,
{
    if units_x == 0 || units_y == 0 || units_z == 0 || cells_per_unit == 0 {
        return Mesh::from_vertices_and_connectivity(vec![], vec![]);
    }

    let cell_size = unit_length / T::from_usize(cells_per_unit).unwrap();
    // Cell, vertex counts along each dimension
    let [cx, cy, cz] = [units_x, units_y, units_z].map(|units| units * cells_per_unit);
    let [vx, vy, vz] = [cx, cy, cz].map(|num_cells| num_cells + 1);

    // Construct all vertices first: first all vertices of the (implicit) uniform hex mesh,
    // then all vertices that correspond to cell centers.
    let mut vertices = Vec::new();
    for (k, j, i) in iproduct!(0..vz, 0..vy, 0..vx) {
        vertices.push(point![
            cell_size * T::from_usize(i).unwrap(),
            cell_size * T::from_usize(j).unwrap(),
            cell_size * T::from_usize(k).unwrap()
        ]);
    }
    let cell_center_offset = vertices.len();
    for (k, j, i) in iproduct!(0..cz, 0..cy, 0..cx) {
        vertices.push(point![
            cell_size * (0.5 + T::from_usize(i).unwrap()),
            cell_size * (0.5 + T::from_usize(j).unwrap()),
            cell_size * (0.5 + T::from_usize(k).unwrap())
        ])
    }

    let vertex_to_global_idx = |[i, j, k]: [usize; 3]| (vx * vy) * k + vx * j + i;
    let cell_to_global_midpoint_idx = |[i, j, k]: [usize; 3]| (cx * cy) * k + cx * j + i + cell_center_offset;

    let mut connectivity = Vec::new();

    // Offsets to [i, j, k] coordinates to obtain the vertices of the face connecting [i, j, k] and
    // its neighbor in the positive direction along each axis 0, 1, 2
    let positive_face_deltas_for_each_axis = [
        [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]].map(Vector3::from),
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]].map(Vector3::from),
        [[0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]].map(Vector3::from),
    ];

    let connect_centers_with_tets = |connectivity: &mut Vec<_>, [i, j, k]: [usize; 3], axis: usize| {
        // Make four tets connecting (i, j, k) and (i + di, j + dj, k + dk).
        // The octahedron formed by the two cell centers and the common face vertices
        // is split into four tetrahedra along the edge between the cell centers.
        let cell = Vector3::from([i, j, k]);
        let cell_delta = Vector3::from_fn(|idx, _| (idx == axis) as usize);

        let shared_face_vertices = positive_face_deltas_for_each_axis[axis]
            .map(|delta| cell + delta)
            .map(|v| vertex_to_global_idx(v.into()));
        let c1 = cell_to_global_midpoint_idx([i, j, k]);
        let c2 = cell_to_global_midpoint_idx((cell + cell_delta).into());
        for (v1, v2) in shared_face_vertices
            .into_iter()
            .cycle()
            .take(5)
            .tuple_windows()
        {
            connectivity.push(Tet4Connectivity([c1, c2, v2, v1]));
        }
    };

    let make_pyramid = |connectivity: &mut Vec<_>, [i, j, k]: [usize; 3], axis: usize, positive_dir: bool| {
        let positive_face_deltas = positive_face_deltas_for_each_axis[axis];
        let mut face_vertices = positive_face_deltas.map(|delta_coord| delta_coord + vector![i, j, k]);
        if !positive_dir {
            // Face vertices are oriented such that they are only correct for the positive
            // direction, need to flip otherwise.
            face_vertices.reverse();
            // Pick the faces one coordinate unit lower
            for coord in &mut face_vertices {
                coord[axis] -= 1;
            }
        }

        let [a, b, c, d] = face_vertices.map(|v| vertex_to_global_idx(v.into()));
        let center = cell_to_global_midpoint_idx([i, j, k]);

        // Ensure that the diagonal choice alternates along the boundary,
        // to prevent excessive diagonal bias along the surface
        if (i + j + k) % 2 == 0 {
            connectivity.push(Tet4Connectivity([a, b, c, center]));
            connectivity.push(Tet4Connectivity([a, c, d, center]));
        } else {
            connectivity.push(Tet4Connectivity([a, b, d, center]));
            connectivity.push(Tet4Connectivity([b, c, d, center]));
        }
    };

    for (k, j, i) in iproduct!(0..cz, 0..cy, 0..cx) {
        let cell = [i, j, k];
        let num_cells = [cx, cy, cz];
        for axis in [0, 1, 2] {
            if cell[axis] + 1 < num_cells[axis] {
                connect_centers_with_tets(&mut connectivity, cell, axis);
            }
            if cell[axis] == 0 {
                make_pyramid(&mut connectivity, cell, axis, false);
            }
            if cell[axis] + 1 == num_cells[axis] {
                make_pyramid(&mut connectivity, cell, axis, true);
            }
        }
    }

    Mesh::from_vertices_and_connectivity(vertices, connectivity)
}

pub fn create_simple_stupid_sphere(center: &Point3<f64>, radius: f64, num_sweeps: usize) -> PolyMesh3d<f64> {
    assert!(num_sweeps > 0);

    // Create cube centered at origin
    let mut mesh = create_rectangular_uniform_hex_mesh(2.0 * radius, 1, 1, 1, 1);
    mesh.translate(&Vector3::new(-radius, -radius, -radius));
    let mut mesh = PolyMesh3d::from(&mesh);

    for i_theta in 0..num_sweeps {
        for j_phi in 0..num_sweeps {
            let theta = PI * (i_theta as f64) / (num_sweeps as f64);
            let phi = 2.0 * PI * (j_phi as f64) / (num_sweeps as f64);
            let r = radius;

            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();

            let x = Point3::new(x, y, z);
            // Normal must point inwards
            let n = -x.coords.normalize();
            let half_space = HalfSpace::from_point_and_normal(x, Unit::new_normalize(n));
            mesh = mesh.intersect_half_space(&half_space);
        }
    }

    // Move sphere from origin to desired center
    mesh.translate(&center.coords);
    mesh
}

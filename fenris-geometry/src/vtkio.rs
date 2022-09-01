//! Currently we've commented out all the functionality here.
//!
//! This is to facilitate easier upgrade to a new version of vtkio. We need to re-evaluate
//! what we actually need from here, and in what form.

// use crate::polymesh::PolyMesh;
// use crate::{ConvexPolygon, GeneralPolygon};
// use ::vtkio::model::{Attributes, Cells, DataSet, PolyDataTopology};
// use nalgebra::allocator::Allocator;
// use nalgebra::{DefaultAllocator, DimName, OPoint, Real, Scalar};
// use num::Zero;
// use std::fs::create_dir_all;
// use std::path::Path;
// use vtkio::model::{Version, Vtk};
// use vtkio::{export_be, Error};
//
// pub fn create_vtk_data_set_from_polygons<T>(polygons: &[ConvexPolygon<T>]) -> DataSet
// where
//     T: Scalar + Zero,
// {
//     let mut points = Vec::new();
//     let mut cells = Cells {
//         num_cells: polygons.len() as u32,
//         vertices: Vec::new(),
//     };
//
//     for polygon in polygons {
//         let point_start = (points.len() / 3) as u32;
//         let num_points = polygon.vertices().len() as u32;
//
//         cells.vertices.push(num_points);
//
//         for (i, vertex) in polygon.vertices().iter().enumerate() {
//             points.push(vertex.x.clone());
//             points.push(vertex.y.clone());
//             points.push(T::zero());
//             cells.vertices.push(point_start + i as u32);
//         }
//     }
//
//     DataSet::PolyData {
//         points: points.into(),
//         topo: vec![PolyDataTopology::Polygons(cells)],
//         data: Attributes::new(),
//     }
// }
//
// pub fn create_vtk_data_set_from_points<T, D>(points: &[OPoint<T, D>]) -> DataSet
// where
//     T: Scalar + Zero,
//     D: DimName,
//     DefaultAllocator: Allocator<T, D>,
// {
//     assert!(D::dim() <= 3, "Only support dimensions up to 3.");
//
//     let mut vtk_points = Vec::new();
//     let mut cells = Cells {
//         num_cells: points.len() as u32,
//         vertices: Vec::new(),
//     };
//
//     for (i, point) in points.iter().enumerate() {
//         for j in 0..D::dim() {
//             vtk_points.push(point.coords[j].clone());
//         }
//
//         for _ in D::dim()..3 {
//             vtk_points.push(T::zero());
//         }
//
//         cells.vertices.push(1);
//         cells.vertices.push(i as u32);
//     }
//
//     DataSet::PolyData {
//         points: vtk_points.into(),
//         topo: vec![PolyDataTopology::Vertices(cells)],
//         data: Attributes::new(),
//     }
// }
//
// /// Convenience method for easily writing polygons to VTK files
// pub fn write_vtk_polygons<T>(
//     polygons: &[ConvexPolygon<T>],
//     filename: &str,
//     title: &str,
// ) -> Result<(), Error>
// where
//     T: Scalar + Zero,
// {
//     let data = create_vtk_data_set_from_polygons(polygons);
//     write_vtk(data, filename, title)
// }
//
// // TODO: This really doesn't belong in this crate. Fix!
// pub fn write_vtk<P: AsRef<Path>>(
//     data: impl Into<DataSet>,
//     filename: P,
//     title: &str,
// ) -> Result<(), Error> {
//     let vtk_file = Vtk {
//         version: Version::new((4, 1)),
//         title: title.to_string(),
//         data: data.into(),
//     };
//
//     let filename = filename.as_ref();
//
//     if let Some(dir) = filename.parent() {
//         create_dir_all(dir)?;
//     }
//     export_be(vtk_file, filename)
// }
//
// impl<'a, T, D> From<&'a PolyMesh<T, D>> for DataSet
// where
//     T: Scalar + Zero,
//     D: DimName,
//     DefaultAllocator: Allocator<T, D>,
// {
//     fn from(mesh: &'a PolyMesh<T, D>) -> Self {
//         assert!(
//             D::dim() == 2 || D::dim() == 3,
//             "Only dimensions 2 and 3 supported."
//         );
//
//         let points: Vec<_> = {
//             let mut points: Vec<T> = Vec::new();
//             for v in mesh.vertices() {
//                 points.extend_from_slice(v.coords.as_slice());
//
//                 if D::dim() == 2 {
//                     points.push(T::zero());
//                 }
//             }
//             points
//         };
//
//         // Vertices is laid out as follows: N, i_1, i_2, ... i_N,
//         // so for quads this becomes 4 followed by the four indices making up the quad
//         let mut vertices = Vec::new();
//         for face in mesh.face_connectivity_iter() {
//             vertices.push(face.len() as u32);
//             for idx in face {
//                 // TODO: Safer cast? How to handle this? TryFrom instead of From?
//                 vertices.push(*idx as u32);
//             }
//         }
//
//         let cells = Cells {
//             num_cells: mesh.num_faces() as u32,
//             vertices,
//         };
//
//         DataSet::PolyData {
//             points: points.into(),
//             topo: vec![PolyDataTopology::Polygons(cells)],
//             data: Attributes::new(),
//         }
//     }
// }
//
// impl<'a, T> From<&'a GeneralPolygon<T>> for DataSet
// where
//     T: Real,
// {
//     fn from(polygon: &'a GeneralPolygon<T>) -> Self {
//         let mut points = Vec::with_capacity(polygon.num_vertices() * 3);
//         let mut cells = Cells {
//             num_cells: polygon.num_edges() as u32,
//             vertices: Vec::new(),
//         };
//
//         for v in polygon.vertices() {
//             points.push(v.x);
//             points.push(v.y);
//             points.push(T::zero());
//         }
//
//         for i in 0..polygon.num_edges() {
//             cells.vertices.push(2);
//             // Edge points from vertex i to i + 1 (modulo)
//             cells.vertices.push(i as u32);
//             cells.vertices.push(((i + 1) % polygon.num_edges()) as u32);
//         }
//
//         DataSet::PolyData {
//             points: points.into(),
//             topo: vec![PolyDataTopology::Lines(cells)],
//             data: Attributes::new(),
//         }
//     }
// }

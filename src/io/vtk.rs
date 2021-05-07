use crate::mesh::{Mesh};
use nalgebra::{DefaultAllocator, DimName, RealField, Scalar};
use vtkio::model::{CellType, Cells, DataSet, UnstructuredGridPiece, VertexNumbers};

use crate::connectivity::{
    Connectivity, Hex20Connectivity, Hex27Connectivity, Hex8Connectivity, Quad4d2Connectivity,
    Quad9d2Connectivity, Segment2d2Connectivity, Tet10Connectivity, Tet4Connectivity,
    Tri3d2Connectivity, Tri3d3Connectivity, Tri6d2Connectivity,
};

use nalgebra::allocator::Allocator;

use std::convert::{TryInto};

// TODO: This is kind of a dirty hack to get around the fact that some VTK things are in
// the geometry crate and some are in this crate. Need to clean this up!
pub use fenris_geometry::vtkio::*;
use crate::vtkio::model::{Piece, Version, ByteOrder, Vtk};
use num::ToPrimitive;
use std::path::Path;

/// Represents connectivity that is supported by VTK.
pub trait VtkCellConnectivity: Connectivity {
    fn num_nodes(&self) -> usize {
        self.vertex_indices().len()
    }

    fn cell_type(&self) -> vtkio::model::CellType;

    /// Write connectivity and return number of nodes.
    ///
    /// Panics if `connectivity.len() != self.num_nodes()`.
    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.vertex_indices().len());
        connectivity.clone_from_slice(self.vertex_indices());
    }
}

impl VtkCellConnectivity for Segment2d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Line
    }
}

impl VtkCellConnectivity for Tri3d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Triangle
    }
}

impl VtkCellConnectivity for Tri6d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticTriangle
    }
}

impl VtkCellConnectivity for Quad4d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Quad
    }
}

impl VtkCellConnectivity for Quad9d2Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticQuad
    }
}

impl VtkCellConnectivity for Tet4Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Tetra
    }
}

impl VtkCellConnectivity for Hex8Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Hexahedron
    }
}

impl VtkCellConnectivity for Tri3d3Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::Triangle
    }
}

impl VtkCellConnectivity for Tet10Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticTetra
    }

    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.vertex_indices().len());
        connectivity.clone_from_slice(self.vertex_indices());

        // Gmsh ordering and ParaView have different conventions for quadratic tets,
        // so we must adjust for that. In particular, nodes 8 and 9 are switched
        connectivity.swap(8, 9);
    }
}

impl VtkCellConnectivity for Hex20Connectivity {
    fn cell_type(&self) -> CellType {
        CellType::QuadraticHexahedron
    }

    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.num_nodes());

        let v = self.vertex_indices();
        // The first 8 entries are the same
        connectivity[0..8].clone_from_slice(&v[0..8]);
        connectivity[8] = v[8];
        connectivity[9] = v[11];
        connectivity[10] = v[13];
        connectivity[11] = v[9];
        connectivity[12] = v[16];
        connectivity[13] = v[18];
        connectivity[14] = v[19];
        connectivity[15] = v[17];
        connectivity[16] = v[10];
        connectivity[17] = v[12];
        connectivity[18] = v[14];
        connectivity[19] = v[15];
    }
}

impl VtkCellConnectivity for Hex27Connectivity {
    fn num_nodes(&self) -> usize {
        20
    }

    // There is no tri-quadratic Hex in legacy VTK, so use Hex20 instead
    fn cell_type(&self) -> CellType {
        CellType::QuadraticHexahedron
    }

    fn write_vtk_connectivity(&self, connectivity: &mut [usize]) {
        assert_eq!(connectivity.len(), self.num_nodes());

        let v = self.vertex_indices();
        // The first 8 entries are the same
        connectivity[0..8].clone_from_slice(&v[0..8]);
        connectivity[8] = v[8];
        connectivity[9] = v[11];
        connectivity[10] = v[13];
        connectivity[11] = v[9];
        connectivity[12] = v[16];
        connectivity[13] = v[18];
        connectivity[14] = v[19];
        connectivity[15] = v[17];
        connectivity[16] = v[10];
        connectivity[17] = v[12];
        connectivity[18] = v[14];
        connectivity[19] = v[15];
    }
}

// impl<'a, T, D, C> From<&'a Mesh<T, D, C>> for DataSet
// where
//     T: Scalar + Zero,
//     D: DimName,
//     C: VtkCellConnectivity,
//     DefaultAllocator: Allocator<T, D>,
// {
//     fn from(mesh: &'a Mesh<T, D, C>) -> Self {
//         // TODO: Create a "SmallDim" trait or something for this case...?
//         // Or just implement the trait directly for U1/U2/U3?
//         assert!(D::dim() <= 3, "Unable to support dimensions larger than 3.");
//         let points: Vec<_> = {
//             let mut points: Vec<T> = Vec::new();
//             for v in mesh.vertices() {
//                 points.extend_from_slice(v.coords.as_slice());
//
//                 for _ in v.coords.len()..3 {
//                     points.push(T::zero());
//                 }
//             }
//             points
//         };
//
//         // Vertices is laid out as follows: N, i_1, i_2, ... i_N,
//         // so for quads this becomes 4 followed by the four indices making up the quad
//         let mut vertices = Vec::new();
//         let mut cell_types = Vec::new();
//         let mut vertex_indices = Vec::new();
//         for cell in mesh.connectivity() {
//             // TODO: Return Result or something
//             vertices.push(cell.num_nodes() as u32);
//
//             vertex_indices.clear();
//             vertex_indices.resize(cell.num_nodes(), 0);
//             cell.write_vtk_connectivity(&mut vertex_indices);
//
//             // TODO: Safer cast? How to handle this? TryFrom instead of From?
//             vertices.extend(vertex_indices.iter().copied().map(|i| i as u32));
//             cell_types.push(cell.cell_type());
//         }
//
//         DataSet::UnstructuredGrid {
//             points: points.into(),
//             cells: Cells {
//                 num_cells: mesh.connectivity().len() as u32,
//                 vertices,
//             },
//             cell_types,
//             data: Attributes::new(),
//         }
//     }
// }

// pub fn create_vtk_data_set_from_quadratures<T, C, D>(
//     vertices: &[Point<T, D>],
//     connectivity: &[C],
//     quadrature_rules: impl IntoIterator<Item = impl Quadrature<T, C::ReferenceDim>>,
// ) -> DataSet
// where
//     T: RealField,
//     D: DimName + DimMin<D, Output = D>,
//     C: ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
//     DefaultAllocator: Allocator<T, D> + ElementConnectivityAllocator<T, C>,
// {
//     let quadrature_rules = quadrature_rules.into_iter();
//
//     // Quadrature weights and points mapped to physical domain
//     let mut physical_weights = Vec::new();
//     let mut physical_points = Vec::new();
//     // Cell indices map each individual quadrature point to its original cell
//     let mut cell_indices = Vec::new();
//
//     for ((cell_idx, conn), quadrature) in zip_eq(connectivity.iter().enumerate(), quadrature_rules)
//     {
//         let element = conn.element(vertices).unwrap();
//         for (w_ref, xi) in zip_eq(quadrature.weights(), quadrature.points()) {
//             let j = element.reference_jacobian(xi);
//             let x = element.map_reference_coords(xi);
//             let w_physical = j.determinant().abs() * *w_ref;
//             physical_points.push(Point::from(x));
//             physical_weights.push(w_physical);
//             cell_indices.push(cell_idx as u64);
//         }
//     }
//
//     let mut dataset = create_vtk_data_set_from_points(&physical_points);
//     let weight_point_attributes = Attribute::Scalars {
//         num_comp: 1,
//         lookup_table: None,
//         data: physical_weights.into(),
//     };
//
//     let cell_idx_point_attributes = Attribute::Scalars {
//         num_comp: 1,
//         lookup_table: None,
//         data: cell_indices.into(),
//     };
//
//     match dataset {
//         DataSet::PolyData { ref mut data, .. } => {
//             data.point
//                 .push(("weight".to_string(), weight_point_attributes));
//             data.point
//                 .push(("cell_index".to_string(), cell_idx_point_attributes));
//         }
//         _ => panic!("Unexpected enum variant from data set."),
//     }
//
//     dataset
// }
//
// /// Convenience function for writing meshes to VTK files.
// pub fn write_vtk_mesh<'a, T, Connectivity>(
//     mesh: &'a Mesh2d<T, Connectivity>,
//     filename: &str,
//     title: &str,
// ) -> Result<(), Error>
// where
//     T: Scalar + Zero,
//     &'a Mesh2d<T, Connectivity>: Into<DataSet>,
// {
//     let data = mesh.into();
//     write_vtk(data, filename, title)
// }

pub struct FiniteElementMeshDataSetBuilder<'a, T, D, C>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    mesh: &'a Mesh<T, D, C>,

    // Only used for exporting directly to file
    title: Option<String>

    // TODO: How to represent attributes?
}

impl<'a, T, D, C> FiniteElementMeshDataSetBuilder<'a, T, D, C>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    pub fn from_mesh(mesh: &'a Mesh<T, D, C>) -> Self {
        Self {
            mesh,
            title: None
        }
    }
}

impl <'a, T, D, C> FiniteElementMeshDataSetBuilder<'a, T, D, C>
where
    T: RealField + ToPrimitive,
    D: DimName,
    DefaultAllocator: Allocator<T, D>
{
    pub fn with_title(self, title: impl Into<String>) -> Self {
        Self {
            mesh: self.mesh,
            title: Some(title.into())
        }
    }

    // TODO: Different error type
    pub fn try_build(&self) -> Result<DataSet, Box<dyn std::error::Error>>
    where
        C: VtkCellConnectivity
    {
        // TODO: Create a "SmallDim" trait or something for this case...?
        // Or just implement the trait directly for U1/U2/U3?
        assert!(D::dim() <= 3, "Unable to support dimensions larger than 3.");
        let points: Vec<_> = {
            let mut points: Vec<T> = Vec::new();
            for v in self.mesh.vertices() {
                points.extend_from_slice(v.coords.as_slice());

                for _ in v.coords.len()..3 {
                    points.push(T::zero());
                }
            }
            points
        };

        // Vertices is laid out as follows: N, i_1, i_2, ... i_N,
        // so for e.g. quads this becomes 4 followed by the four indices making up the quad
        let mut vertices = Vec::new();
        let mut cell_types = Vec::new();
        let mut vertex_indices = Vec::new();
        for cell in self.mesh.connectivity() {
            // TODO: Return better error
            vertices.push(cell.num_nodes().try_into()?);

            vertex_indices.clear();
            vertex_indices.resize(cell.num_nodes(), 0);
            cell.write_vtk_connectivity(&mut vertex_indices);

            for &idx in &vertex_indices {
                // TODO: Return better error
                vertices.push(idx.try_into()?);
            }
            cell_types.push(cell.cell_type());
        }

        // TODO: Attributes

        let piece = UnstructuredGridPiece {
            points: points.into(),
            cells: Cells {
                // TODO: Use XML instead of Legacy?
                cell_verts: VertexNumbers::Legacy {
                    num_cells: self.mesh.connectivity().len() as u32,
                    vertices
                },
                types: cell_types
            },
            data: Default::default()
        };

        Ok(DataSet::UnstructuredGrid {
            meta: None,
            pieces: vec![Piece::Inline(Box::new(piece))]
        })
    }

    /// Convenience function for directly exporting the dataset to a file.
    pub fn try_export(&self, filename: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>>
    where
        C: VtkCellConnectivity
    {
        let filepath = filename.as_ref();
        let fallback_title = filepath.file_stem()
            .map(|os_str| os_str.to_string_lossy().to_string())
            .unwrap_or_else(|| "untitled".to_string());
        let dataset = self.try_build()?;
        Vtk {
            // TODO: What to choose here? Depends on format?
            version: Version { major: 4, minor: 1 },
            // If we don't have a title then just make the filepath the title
            title: self.title.clone().unwrap_or(fallback_title),
            byte_order: ByteOrder::BigEndian,
            data: dataset,
            file_path: None
        }.export(filepath)?;
        Ok(())
    }
}
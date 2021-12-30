use crate::connectivity::{Tet4Connectivity, Tri3d2Connectivity, Tri3d3Connectivity};
use crate::mesh::Mesh;
use eyre::{eyre, Context};
use log::warn;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Point2, Point3, RealField, U2, U3};
use std::path::Path;

/// Loads a [`Mesh`] from a Gmsh MSH file at the given path.
pub fn load_msh_from_file<T, D, C, P: AsRef<Path>>(file_path: P) -> eyre::Result<Mesh<T, D, C>>
where
    T: RealField,
    D: DimName,
    C: MshConnectivity,
    OPoint<T, D>: TryVertexFromMshNode<T, D, f64>,
    DefaultAllocator: Allocator<T, D>,
{
    let msh_bytes = std::fs::read(file_path).wrap_err("failed to read file")?;
    load_msh_from_bytes(&msh_bytes).wrap_err("failed to load mesh from msh file")
}

/// Loads a [`Mesh`] by parsing the given bytes as a Gmsh MSH file.
pub fn load_msh_from_bytes<T, D, C>(bytes: &[u8]) -> eyre::Result<Mesh<T, D, C>>
where
    T: RealField,
    D: DimName,
    C: MshConnectivity,
    OPoint<T, D>: TryVertexFromMshNode<T, D, f64>,
    DefaultAllocator: Allocator<T, D>,
{
    let mut msh_file = mshio::parse_msh_bytes(bytes).map_err(|e| eyre!("failed to parse msh file: {}", e))?;

    let msh_nodes = msh_file
        .data
        .nodes
        .take()
        .ok_or(eyre!("MSH file does not contain nodes"))?;
    let msh_elements = msh_file
        .data
        .elements
        .take()
        .ok_or(eyre!("MSH file does not contain elements"))?;

    let mut vertices = Vec::new();
    let mut connectivity = Vec::new();

    // Ensure that at least one element block matches the target mesh connectivity
    if !msh_elements
        .element_blocks
        .iter()
        .any(|block| element_block_matches_connectivity::<C, _>(block))
    {
        return Err(eyre!(
            "MSH file does not contain an element block of the requested type ({:?} of dim {})",
            C::msh_element_type(),
            C::reference_dim()
        ));
    }

    // Collect all mesh vertices
    for node_block in &msh_nodes.node_blocks {
        let block_vertices = vertices_from_node_block(node_block)?;
        vertices.extend(block_vertices);
    }

    // Collect all connectivity matching the target connectivity
    for element_block in &msh_elements.element_blocks {
        let block_connectivity = connectivity_from_element_block(element_block)?;
        connectivity.extend(block_connectivity);
    }

    Ok(Mesh::from_vertices_and_connectivity(vertices, connectivity))
}

/// Tries to convert a `mshio::NodeBlock` to a `Vec<OPoint<T, D>>`.
fn vertices_from_node_block<T, D, F, I>(node_block: &mshio::NodeBlock<u64, I, F>) -> eyre::Result<Vec<OPoint<T, D>>>
where
    T: RealField,
    D: DimName,
    F: mshio::MshFloatT,
    I: mshio::MshIntT,
    OPoint<T, D>: TryVertexFromMshNode<T, D, F>,
    DefaultAllocator: Allocator<T, D>,
{
    // Ensure that node tags are consecutive
    if node_block.node_tags.is_some() {
        return Err(eyre!("node block tags are not consecutive in msh file"));
    }

    // Check dimension of node block vertices
    if node_block
        .entity_dim
        .to_usize()
        .ok_or_else(|| eyre!("error converting node block entity dimension to usize"))?
        != D::dim()
    {
        // TODO: When can this happen?
        warn!("Node block entity does not have the right dimension for this mesh. Will be read as if they were of the same dimension.");
        //return Err(eyre!("node block entity does not have the right dimension for this mesh"));
    }

    let mut vertices = Vec::with_capacity(node_block.nodes.len());

    // Convert MSH vertices to points
    for node in &node_block.nodes {
        vertices.push(OPoint::try_vertex_from_msh_node(node)?);
    }

    Ok(vertices)
}

/// Tries to convert a `mshio::ElementBlock` to a `Vec<Connectivity>`.
fn connectivity_from_element_block<C, I>(element_block: &mshio::ElementBlock<u64, I>) -> eyre::Result<Vec<C>>
where
    C: MshConnectivity,
    I: mshio::MshIntT,
{
    // Ensure that element tags are consecutive
    if element_block.element_tags.is_some() {
        return Err(eyre!("element block tags are not consecutive in msh file"));
    }

    if !element_block_matches_connectivity::<C, _>(element_block) {
        // Just ignore blocks that don't match the requested connectivity
        return Ok(Vec::new());
    } else {
        let mut connectivity = Vec::with_capacity(element_block.elements.len());
        let requested_nodes = C::msh_element_type()
            .nodes()
            .map_err(|_| eyre!("unimplemented element type requested"))?;

        for element in &element_block.elements {
            if element.nodes.len() < requested_nodes {
                return Err(eyre!("not enough nodes to initialize connectivity"));
            }
            connectivity.push(C::try_connectivity_from_msh_element(element)?);
        }

        return Ok(connectivity);
    }
}

/// Returns whether the given element block contains elements corresponding to the specified connectivity.
fn element_block_matches_connectivity<C, I>(element_block: &mshio::ElementBlock<u64, I>) -> bool
where
    C: MshConnectivity,
    I: mshio::MshIntT,
{
    element_block.element_type == C::msh_element_type()
        && element_block
            .entity_dim
            .to_usize()
            .expect("failed to convert element block dimension to usize")
            == C::reference_dim()
}

/// Allows conversion from `mshio::Node`s to `OPoint`s which are used as vertices in `fenris`.
pub trait TryVertexFromMshNode<T, D, F>
where
    T: RealField,
    D: DimName,
    F: mshio::MshFloatT,
    DefaultAllocator: Allocator<T, D>,
{
    fn try_vertex_from_msh_node(node: &mshio::Node<F>) -> eyre::Result<OPoint<T, D>>;
}

macro_rules! f_to_t {
    ($component:expr) => {
        T::from_f64(
            $component
                .to_f64()
                .ok_or_else(|| eyre!("failed to convert coordinate to f64"))?,
        )
        .ok_or_else(|| eyre!("failed to convert node coordinate from f64 to target mesh real type"))?
    };
}

impl<T, F> TryVertexFromMshNode<T, U2, F> for Point2<T>
where
    T: RealField,
    F: mshio::MshFloatT,
{
    fn try_vertex_from_msh_node(node: &mshio::Node<F>) -> eyre::Result<Self> {
        // TODO: Ensure that node.z is zero?
        Ok(Self::new(f_to_t!(node.x), f_to_t!(node.y)))
    }
}

impl<T, F> TryVertexFromMshNode<T, U3, F> for Point3<T>
where
    T: RealField,
    F: mshio::MshFloatT,
{
    fn try_vertex_from_msh_node(node: &mshio::Node<F>) -> eyre::Result<Self> {
        Ok(Self::new(f_to_t!(node.x), f_to_t!(node.y), f_to_t!(node.z)))
    }
}

/// Allows conversion from `mshio::Element`s to connectivity types used in `fenris`.
pub trait MshConnectivity
where
    Self: Sized,
{
    /// Returns the MSH element type corresponding to this connectivity.
    fn msh_element_type() -> mshio::ElementType;
    /// Returns the reference dimension of this connectivity (corresponds to MSH entity dimension).
    fn reference_dim() -> usize;
    /// Tries to construct the element connectivity from the given MSH element.
    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> eyre::Result<Self>;
}

impl MshConnectivity for Tri3d2Connectivity {
    fn msh_element_type() -> mshio::ElementType {
        mshio::ElementType::Tri3
    }

    fn reference_dim() -> usize {
        2
    }

    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> eyre::Result<Self> {
        Ok(Self([
            element.nodes[0] as usize - 1,
            element.nodes[1] as usize - 1,
            element.nodes[2] as usize - 1,
        ]))
    }
}

impl MshConnectivity for Tri3d3Connectivity {
    fn msh_element_type() -> mshio::ElementType {
        mshio::ElementType::Tri3
    }

    fn reference_dim() -> usize {
        3
    }

    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> eyre::Result<Self> {
        Ok(Self([
            element.nodes[0] as usize - 1,
            element.nodes[1] as usize - 1,
            element.nodes[2] as usize - 1,
        ]))
    }
}

impl MshConnectivity for Tet4Connectivity {
    fn msh_element_type() -> mshio::ElementType {
        mshio::ElementType::Tet4
    }

    fn reference_dim() -> usize {
        3
    }

    fn try_connectivity_from_msh_element(element: &mshio::Element<u64>) -> eyre::Result<Self> {
        Ok(Self([
            element.nodes[0] as usize - 1,
            element.nodes[1] as usize - 1,
            element.nodes[2] as usize - 1,
            element.nodes[3] as usize - 1,
        ]))
    }
}

#[cfg(test)]
mod msh_tests {
    use crate::connectivity::{Tet4Connectivity, Tri3d2Connectivity, Tri3d3Connectivity};
    use crate::io::msh::load_msh_from_file;
    use nalgebra::{U2, U3};

    #[test]
    fn load_msh_sphere_tet4() -> eyre::Result<()> {
        let mesh = load_msh_from_file::<f64, U3, Tet4Connectivity, _>("assets/meshes/sphere_593.msh")?;

        assert_eq!(mesh.vertices().len(), 183);
        assert_eq!(mesh.connectivity().len(), 593);
        Ok(())
    }

    #[test]
    fn load_msh_rect_tri3d2() -> eyre::Result<()> {
        let mesh = load_msh_from_file::<f64, U2, Tri3d2Connectivity, _>("assets/meshes/rectangle_110.msh")?;

        assert_eq!(mesh.vertices().len(), 70);
        assert_eq!(mesh.connectivity().len(), 110);
        Ok(())
    }
}

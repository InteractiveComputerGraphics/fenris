use crate::allocators::ElementConnectivityAllocator;
use crate::assembly::color_nodes;
use crate::connectivity::{
    CellConnectivity, Connectivity, Quad4d2Connectivity, Quad9d2Connectivity, Tet4Connectivity,
    Tri3d2Connectivity, Tri6d2Connectivity,
};
use crate::element::{map_physical_coordinates, ElementConnectivity, ReferenceFiniteElement};
use crate::geometry::{Distance, DistanceQuery, GeometryCollection};
use crate::mesh::Mesh;
use crate::quadrature::QuadraturePair;
use crate::space::GeometricFiniteElementSpace;
use itertools::izip;
use nalgebra::allocator::Allocator;
use nalgebra::{
    DVector, DefaultAllocator, DimMin, DimName, Dynamic, MatrixMN, MatrixSliceMut, Point,
    RealField, Scalar, VectorN, U1, U2, U3,
};
use paradis::DisjointSubsets;
use serde::{Deserialize, Serialize};

/// A finite element model consisting of vertices (physical nodes) and physical elements
/// that are defined by their connectivity to the vertices.
///
/// This generalizes the usual finite element bases, such as standard Lagrangian polynomial
/// finite elements on quads/hex/tri/tet meshes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize,\
                 <DefaultAllocator as Allocator<T, D>>::Buffer: Serialize,\
                 Connectivity: Serialize",
    deserialize = "T: Deserialize<'de>,\
                   <DefaultAllocator as Allocator<T, D>>::Buffer: Deserialize<'de>,\
                   Connectivity: Deserialize<'de>"
))]
pub struct NodalModel<T, D, Connectivity>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    mesh: Mesh<T, D, Connectivity>,

    mass_quadrature: Option<QuadraturePair<T, D>>,
    stiffness_quadrature: Option<QuadraturePair<T, D>>,
    elliptic_quadrature: Option<QuadraturePair<T, D>>,

    // Colors for parallel assembly
    colors: Vec<DisjointSubsets>,
}

pub type NodalModel2d<T, C> = NodalModel<T, U2, C>;
pub type NodalModel3d<T, C> = NodalModel<T, U3, C>;

impl<T, D, C> NodalModel<T, D, C>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn mesh(&self) -> &Mesh<T, D, C> {
        &self.mesh
    }

    pub fn connectivity(&self) -> &[C] {
        self.mesh.connectivity()
    }

    pub fn vertices(&self) -> &[Point<T, D>] {
        self.mesh.vertices()
    }

    pub fn mass_quadrature(&self) -> Option<&QuadraturePair<T, D>> {
        self.mass_quadrature.as_ref()
    }

    pub fn stiffness_quadrature(&self) -> Option<&QuadraturePair<T, D>> {
        self.stiffness_quadrature.as_ref()
    }

    pub fn elliptic_quadrature(&self) -> Option<&QuadraturePair<T, D>> {
        self.elliptic_quadrature.as_ref()
    }

    pub fn colors(&self) -> &[DisjointSubsets] {
        &self.colors
    }

    /// Constructs a new model from the given mesh and quadrature.
    ///
    /// The same quadrature is used for all quadrature kinds.
    ///
    /// TODO: Remove/deprecate this method. It is currently only here for legacy reasons.
    pub fn from_mesh_and_quadrature(
        mesh: Mesh<T, D, C>,
        quadrature: (Vec<T>, Vec<VectorN<T, D>>),
    ) -> Self
    where
        C: Connectivity,
    {
        let colors = color_nodes(mesh.connectivity());
        Self {
            mesh,
            mass_quadrature: Some(quadrature.clone()),
            stiffness_quadrature: Some(quadrature.clone()),
            elliptic_quadrature: Some(quadrature.clone()),
            colors,
        }
    }

    /// Constructs a new model from the given mesh, without attaching any quadrature.
    pub fn from_mesh(mesh: Mesh<T, D, C>) -> Self
    where
        C: Connectivity,
    {
        let colors = color_nodes(mesh.connectivity());
        Self {
            mesh,
            mass_quadrature: None,
            stiffness_quadrature: None,
            elliptic_quadrature: None,
            colors,
        }
    }

    pub fn with_mass_quadrature(self, mass_quadrature: QuadraturePair<T, D>) -> Self {
        Self {
            mass_quadrature: Some(mass_quadrature),
            ..self
        }
    }

    pub fn with_stiffness_quadrature(self, stiffness_quadrature: QuadraturePair<T, D>) -> Self {
        Self {
            stiffness_quadrature: Some(stiffness_quadrature),
            ..self
        }
    }

    pub fn with_elliptic_quadrature(self, elliptic_quadrature: QuadraturePair<T, D>) -> Self {
        Self {
            elliptic_quadrature: Some(elliptic_quadrature),
            ..self
        }
    }
}

pub type Quad4Model<T> = NodalModel2d<T, Quad4d2Connectivity>;
pub type Tri3d2Model<T> = NodalModel2d<T, Tri3d2Connectivity>;
pub type Tri6d2Model<T> = NodalModel2d<T, Tri6d2Connectivity>;
pub type Quad9Model<T> = NodalModel2d<T, Quad9d2Connectivity>;
pub type Tet4Model<T> = NodalModel3d<T, Tet4Connectivity>;

impl<'a, T, D, C> GeometryCollection<'a> for NodalModel<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: CellConnectivity<T, D>,
    DefaultAllocator: Allocator<T, D>,
{
    type Geometry = C::Cell;

    fn num_geometries(&self) -> usize {
        self.connectivity().len()
    }

    fn get_geometry(&'a self, index: usize) -> Option<Self::Geometry> {
        self.connectivity().get(index)?.cell(self.vertices())
    }
}

impl<'a, T, D, C, QueryGeometry> DistanceQuery<'a, QueryGeometry> for NodalModel<T, D, C>
where
    T: RealField,
    D: DimName,
    C: CellConnectivity<T, D>,
    Mesh<T, D, C>: DistanceQuery<'a, QueryGeometry>,
    DefaultAllocator: Allocator<T, D>,
{
    fn nearest(&'a self, query_geometry: &'a QueryGeometry) -> Option<usize> {
        self.mesh.nearest(query_geometry)
    }
}

/// Interpolates solution variables onto a fixed set of interpolation points.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteElementInterpolator<T> {
    // Store the highest node index in supported_nodes, so that we can
    // guarantee that we don't go out of bounds during interpolation.
    max_node_index: Option<usize>,

    // For a set of points X_I and solution variables u, a finite element interpolation can be written
    //  u_h(X_I) = sum_J N_J(X_I) * u_J,
    // where N_J is the basis function associated with node J, u_J is the solution variable
    // associated with node J (basis weight) and u_h is the interpolation solution. Since the basis
    // functions have local support, it suffices to consider nodes J for which X_I lies in the
    // support of N_J.
    // While the above is essentially a matrix-vector multiplication, we want to work with
    // low-dimensional point and vector types. Thus we implement a CSR-like custom format
    // that lets us compactly represent the weights.

    // Offsets into the node support vector. supported_node_offsets[I] gives the
    // index of the first node that is supported on X_I. The length of support_node_offsets is
    // m + 1, where m is the number of interpolation points X_I.
    // This way the number of supported bases for a given point I is given by
    // count = supported_node_offsets[I + 1] - supported_node_offsets[I].
    supported_node_offsets: Vec<usize>,

    /// Stores the value N and index of the basis function of each supported node
    node_values: Vec<(T, usize)>,
}

impl<T> FiniteElementInterpolator<T>
where
    T: RealField,
{
    pub fn interpolate<SolutionDim>(&self, u: &DVector<T>) -> Vec<VectorN<T, SolutionDim>>
    where
        SolutionDim: DimName,
        DefaultAllocator: Allocator<T, SolutionDim, U1>,
    {
        let num_sol_vectors = self.supported_node_offsets.len().saturating_sub(1);
        let mut sol_vectors = vec![VectorN::zeros(); num_sol_vectors];
        self.interpolate_into(&mut sol_vectors, u);
        sol_vectors
    }

    // TODO: Take "arbitrary" u, not just DVector
    pub fn interpolate_into<SolutionDim>(
        &self,
        result: &mut [VectorN<T, SolutionDim>],
        u: &DVector<T>,
    ) where
        SolutionDim: DimName,
        DefaultAllocator: Allocator<T, SolutionDim, U1>,
    {
        assert_eq!(
            result.len() + 1,
            self.supported_node_offsets.len(),
            "Number of interpolation points must match."
        );
        assert!(
            self.max_node_index.is_none()
                || SolutionDim::dim() * self.max_node_index.unwrap() < u.len(),
            "Cannot reference degrees of freedom not present in solution variables"
        );

        for i in 0..result.len() {
            let i_support_start = self.supported_node_offsets[i];
            let i_support_end = self.supported_node_offsets[i + 1];

            result[i].fill(T::zero());

            for (v, j) in &self.node_values[i_support_start..i_support_end] {
                let u_j = u.fixed_slice::<SolutionDim, U1>(SolutionDim::dim() * j, 0);
                result[i] += u_j * v.clone();
            }
        }
    }
}

impl<T> FiniteElementInterpolator<T> {
    pub fn from_compressed_values(
        node_values: Vec<(T, usize)>,
        supported_node_offsets: Vec<usize>,
    ) -> Self {
        assert!(
            supported_node_offsets
                .iter()
                .all(|i| *i < node_values.len() + 1),
            "Supported node offsets must be in bounds with respect to supported nodes."
        );

        Self {
            max_node_index: node_values.iter().map(|(_, i)| i).max().cloned(),
            node_values,
            supported_node_offsets,
        }
    }
}

impl<T> FiniteElementInterpolator<T> {
    pub fn interpolate_space<'a, Space, D>(
        mesh: &'a Space,
        interpolation_points: &'a [Point<T, D>],
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        T: RealField,
        D: DimName + DimMin<D, Output = D>,
        Space: GeometricFiniteElementSpace<'a, T> + DistanceQuery<'a, Point<T, D>>,
        Space::Connectivity: ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
        DefaultAllocator: ElementConnectivityAllocator<T, Space::Connectivity>,
    {
        let mut supported_node_offsets = Vec::new();
        let mut node_values = Vec::new();

        let mut basis_buffer = MatrixMN::<_, U1, Dynamic>::zeros(0);

        for point in interpolation_points {
            let point_node_support_begin = node_values.len();
            supported_node_offsets.push(point_node_support_begin);

            if mesh.num_connectivities() > 0 {
                let element_idx = mesh
                    .nearest(point)
                    .expect("Logic error: Mesh should have non-zero number of cells/elements.");
                let conn = mesh.get_connectivity(element_idx).unwrap();
                let element = mesh.get_element(element_idx).unwrap();

                let xi = map_physical_coordinates(&element, point)
                    .map_err(|_| "Failed to map physical coordinates to reference coordinates.")?;

                basis_buffer.resize_horizontally_mut(element.num_nodes(), T::zero());
                element.populate_basis(MatrixSliceMut::from(&mut basis_buffer), &xi.coords);
                for (index, v) in izip!(conn.vertex_indices(), basis_buffer.iter()) {
                    node_values.push((v.clone(), index.clone()));
                }
            }
        }

        supported_node_offsets.push(node_values.len());
        assert_eq!(interpolation_points.len() + 1, supported_node_offsets.len());

        Ok(FiniteElementInterpolator::from_compressed_values(
            node_values,
            supported_node_offsets,
        ))
    }
}

impl<T, D, Connectivity> NodalModel<T, D, Connectivity>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    Connectivity:
        CellConnectivity<T, D> + ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    Connectivity::Cell: Distance<T, Point<T, D>>,
    Mesh<T, D, Connectivity>:
        for<'a> GeometricFiniteElementSpace<'a, T, Connectivity = Connectivity>,
    DefaultAllocator: ElementConnectivityAllocator<T, Connectivity>,
{
    /// Creates an interpolator that interpolates solution variables at the given
    /// interpolation points.
    ///
    /// Returns an error if the elements can not be converted to convex polygons,
    /// or if an interpolation point is outside of the computational domain,
    /// or if mapping a physical coordinate to a reference coordinate for the given
    /// element fails.
    /// TODO: Return proper error differentiating the different failure cases.
    pub fn make_interpolator(
        &self,
        interpolation_points: &[Point<T, D>],
    ) -> Result<FiniteElementInterpolator<T>, Box<dyn std::error::Error>> {
        FiniteElementInterpolator::interpolate_space(&self.mesh, interpolation_points)
    }
}

pub trait MakeInterpolator<T, D>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    DefaultAllocator: Allocator<T, D>,
{
    fn make_interpolator(
        &self,
        interpolation_points: &[Point<T, D>],
    ) -> Result<FiniteElementInterpolator<T>, Box<dyn std::error::Error>>;
}

impl<T, D, Connectivity> MakeInterpolator<T, D> for NodalModel<T, D, Connectivity>
where
    T: RealField,
    D: DimName + DimMin<D, Output = D>,
    Connectivity:
        CellConnectivity<T, D> + ElementConnectivity<T, GeometryDim = D, ReferenceDim = D>,
    Connectivity::Cell: Distance<T, Point<T, D>>,
    Mesh<T, D, Connectivity>:
        for<'a> GeometricFiniteElementSpace<'a, T, Connectivity = Connectivity>,
    DefaultAllocator: ElementConnectivityAllocator<T, Connectivity>,
{
    fn make_interpolator(
        &self,
        interpolation_points: &[Point<T, D>],
    ) -> Result<FiniteElementInterpolator<T>, Box<dyn std::error::Error>> {
        self.make_interpolator(interpolation_points)
    }
}

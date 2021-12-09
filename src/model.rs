use crate::allocators::BiDimAllocator;
use crate::geometry::DistanceQuery;
use crate::space::GeometricFiniteElementSpace;
use nalgebra::allocator::Allocator;
use nalgebra::{DVector, DefaultAllocator, DimMin, DimName, OPoint, OVector, RealField, U1};
use serde::{Deserialize, Serialize};

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
    pub fn interpolate<SolutionDim>(&self, u: &DVector<T>) -> Vec<OVector<T, SolutionDim>>
    where
        SolutionDim: DimName,
        DefaultAllocator: Allocator<T, SolutionDim, U1>,
    {
        let num_sol_vectors = self.supported_node_offsets.len().saturating_sub(1);
        let mut sol_vectors = vec![OVector::zeros(); num_sol_vectors];
        self.interpolate_into(&mut sol_vectors, u);
        sol_vectors
    }

    // TODO: Take "arbitrary" u, not just DVector
    pub fn interpolate_into<SolutionDim>(&self, result: &mut [OVector<T, SolutionDim>], u: &DVector<T>)
    where
        SolutionDim: DimName,
        DefaultAllocator: Allocator<T, SolutionDim, U1>,
    {
        assert_eq!(
            result.len() + 1,
            self.supported_node_offsets.len(),
            "Number of interpolation points must match."
        );
        assert!(
            self.max_node_index.is_none() || SolutionDim::dim() * self.max_node_index.unwrap() < u.len(),
            "Cannot reference degrees of freedom not present in solution variables"
        );

        for i in 0..result.len() {
            let i_support_start = self.supported_node_offsets[i];
            let i_support_end = self.supported_node_offsets[i + 1];

            result[i].fill(T::zero());

            for (v, j) in &self.node_values[i_support_start..i_support_end] {
                let u_j = u.generic_slice((SolutionDim::dim() * j, 0), (SolutionDim::name(), U1::name()));
                result[i] += u_j * v.clone();
            }
        }
    }
}

impl<T> FiniteElementInterpolator<T> {
    pub fn from_compressed_values(node_values: Vec<(T, usize)>, supported_node_offsets: Vec<usize>) -> Self {
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
        _space: &'a Space,
        _interpolation_points: &'a [OPoint<T, D>],
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        T: RealField,
        D: DimName + DimMin<D, Output = D>,
        Space: GeometricFiniteElementSpace<'a, T, GeometryDim = D> + DistanceQuery<'a, OPoint<T, D>>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        todo!("Reimplement this function or scrap it in favor of a different design?");
        // let mut supported_node_offsets = Vec::new();
        // let mut node_values = Vec::new();
        //
        // let mut basis_buffer = OMatrix::<_, U1, Dynamic>::zeros(0);
        //
        // for point in interpolation_points {
        //     let point_node_support_begin = node_values.len();
        //     supported_node_offsets.push(point_node_support_begin);
        //
        //     if !mesh.connectivity().is_empty() > 0 {
        //         let element_idx = mesh
        //             .nearest(point)
        //             .expect("Logic error: Mesh should have non-zero number of cells/elements.");
        //         let conn = mesh.get_connectivity(element_idx).unwrap();
        //         let element = mesh.get_element(element_idx).unwrap();
        //
        //         let xi = map_physical_coordinates(&element, point)
        //             .map_err(|_| "Failed to map physical coordinates to reference coordinates.")?;
        //
        //         basis_buffer.resize_horizontally_mut(element.num_nodes(), T::zero());
        //         element.populate_basis(MatrixSliceMut::from(&mut basis_buffer), &xi.coords);
        //         for (index, v) in izip!(conn.vertex_indices(), basis_buffer.iter()) {
        //             node_values.push((v.clone(), index.clone()));
        //         }
        //     }
        // }
        //
        // supported_node_offsets.push(node_values.len());
        // assert_eq!(interpolation_points.len() + 1, supported_node_offsets.len());
        //
        // Ok(FiniteElementInterpolator::from_compressed_values(
        //     node_values,
        //     supported_node_offsets,
        // ))
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
        interpolation_points: &[OPoint<T, D>],
    ) -> Result<FiniteElementInterpolator<T>, Box<dyn std::error::Error>>;
}

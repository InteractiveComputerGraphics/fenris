use crate::allocators::DimAllocator;
use crate::assembly::buffers::{BasisFunctionBuffer, QuadratureBuffer};
use crate::assembly::local::{ElementConnectivityAssembler, ElementMatrixAssembler, QuadratureTable};
use crate::define_thread_local_workspace;
use crate::element::{ReferenceFiniteElement, VolumetricFiniteElement};
use crate::nalgebra::{DMatrixSliceMut, DefaultAllocator, DimName, OPoint};
use crate::space::{ElementInSpace, FiniteElementConnectivity, VolumetricFiniteElementSpace};
use crate::util::clone_upper_to_lower;
use crate::workspace::with_thread_local_workspace;
use itertools::izip;
use nalgebra::{RealField, Scalar};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

/// A wrapper type for a number that represents a *density*.
///
/// This is primarily used as a parameter for mass matrix construction.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Density<T>(pub T);

impl<T> Density<T> {
    pub fn as_inner_slice<'a>(slice: &'a [Density<T>]) -> &'a [T] {
        // SAFETY: This is sound because `Density` is `repr(transparent)`
        unsafe { std::mem::transmute(slice) }
    }

    pub fn from_inner_slice<'a>(slice: &'a [T]) -> &'a [Density<T>] {
        // SAFETY: This is sound because `Density` is `repr(transparent)`
        unsafe { std::mem::transmute(slice) }
    }
}

impl<T: RealField> Default for Density<T> {
    fn default() -> Self {
        Density(T::zero())
    }
}

impl<T: Display> Display for Density<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Density({})", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct ElementMassAssembler<'a, Space, QTable> {
    space: &'a Space,
    qtable: &'a QTable,
    solution_dim: usize,
}

impl<'a> ElementMassAssembler<'a, (), ()> {
    pub fn with_solution_dim(solution_dim: usize) -> Self {
        Self {
            space: &(),
            qtable: &(),
            solution_dim,
        }
    }
}

impl<'a, QTable> ElementMassAssembler<'a, (), QTable> {
    pub fn with_space<Space>(self, space: &'a Space) -> ElementMassAssembler<'a, Space, QTable> {
        ElementMassAssembler {
            space,
            qtable: self.qtable,
            solution_dim: self.solution_dim,
        }
    }
}

impl<'a, Space> ElementMassAssembler<'a, Space, ()> {
    pub fn with_quadrature_table<QTable>(self, table: &'a QTable) -> ElementMassAssembler<'a, Space, QTable> {
        ElementMassAssembler {
            space: self.space,
            qtable: table,
            solution_dim: self.solution_dim,
        }
    }
}

define_thread_local_workspace!(WORKSPACE);

impl<'a, Space, QTable> ElementConnectivityAssembler for ElementMassAssembler<'a, Space, QTable>
where
    Space: FiniteElementConnectivity,
{
    fn solution_dim(&self) -> usize {
        self.solution_dim
    }

    fn num_elements(&self) -> usize {
        self.space.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.space.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.space.populate_element_nodes(output, element_index)
    }
}

#[derive(Debug)]
struct MassAssemblerWorkspace<T: Scalar, D: DimName>
where
    DefaultAllocator: DimAllocator<T, D>,
{
    quadrature_buffer: QuadratureBuffer<T, D, Density<T>>,
    basis_buffer: BasisFunctionBuffer<T>,
}

impl<T: RealField, D: DimName> Default for MassAssemblerWorkspace<T, D>
where
    DefaultAllocator: DimAllocator<T, D>,
{
    fn default() -> Self {
        Self {
            quadrature_buffer: Default::default(),
            basis_buffer: Default::default(),
        }
    }
}

impl<'a, T, Space, QTable> ElementMatrixAssembler<T> for ElementMassAssembler<'a, Space, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::GeometryDim, Data = Density<T>>,
    DefaultAllocator: DimAllocator<T, Space::GeometryDim>,
{
    fn assemble_element_matrix_into(&self, element_index: usize, output: DMatrixSliceMut<T>) -> eyre::Result<()> {
        with_thread_local_workspace(&WORKSPACE, |ws: &mut MassAssemblerWorkspace<T, Space::GeometryDim>| {
            let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
            ws.basis_buffer
                .resize(element.num_nodes(), Space::ReferenceDim::dim());
            ws.basis_buffer
                .populate_element_nodes_from_space(element_index, self.space);
            ws.quadrature_buffer
                .populate_element_quadrature_from_table(element_index, self.qtable);

            assemble_element_mass_matrix(
                output,
                &element,
                ws.quadrature_buffer.weights(),
                ws.quadrature_buffer.points(),
                Density::as_inner_slice(ws.quadrature_buffer.data()),
                self.solution_dim,
                ws.basis_buffer.element_basis_values_mut(),
            )
        })
    }
}

/// Assembles the element mass matrix using the provided quadrature.
///
/// Given a finite element, a quadrature rule with density values associated to each quadrature point,
/// stores the resulting element mass matrix in the provided output vector.
///
/// Given a finite element with domain $K$ and $N$ nodes, the element mass matrix is the matrix
/// $M^K \in \mathbb{R}^{s N\times s N}$ whose $s \times s$ blocks $M_{IJ}$ are defined by
///
/// $$
/// M^K_{IJ} := I^s \int_{K} \rho(x) \\, \phi_I(x) \\, \phi_J(x) \\, \mathrm{d} V \qquad I, J = 1, \dots, N.
/// $$
///
/// Here $s$ is the dimension of the solution variable, $I^s$ is the $s \times s$ identity matrix,
/// $\phi_I$ is the basis function associated with node $I$ and $\rho: \mathbb{\Omega} \rightarrow \mathbb{R}^+$
/// is a (non-negative) *density function*.
///
/// For $s = 1$, we obtain the mass matrix associated with e.g. the scalar wave equation,
/// whereas for $s = d$ with $d$ being the dimension of the domain we obtain the mass matrix associated with e.g.
/// time-dependent elasticity equations.
///
/// The computation requires a buffer for evaluating basis functions. The buffer must be able to
/// store the function value for each node in the element.
///
/// # Panics
///
/// Panics if the quadrature arrays do not have the same lengths.
///
/// Panics if the number of elements in the basis value buffer is not equal to the number of nodes
/// in the element.
#[allow(non_snake_case)]
pub fn assemble_element_mass_matrix<T, Element>(
    mut output: DMatrixSliceMut<T>,
    element: &Element,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    quadrature_density: &[T],
    solution_dim: usize,
    basis_values_buffer: &mut [T],
) -> eyre::Result<()>
where
    T: RealField,
    // We only support volumetric elements atm
    Element: VolumetricFiniteElement<T>,
    DefaultAllocator: DimAllocator<T, Element::GeometryDim>,
{
    assert_eq!(quadrature_weights.len(), quadrature_points.len());
    assert_eq!(quadrature_points.len(), quadrature_density.len());
    assert_eq!(basis_values_buffer.len(), element.num_nodes());

    let s = solution_dim;
    let n = element.num_nodes();
    assert_eq!(output.nrows(), s * n, "Output matrix dimension mismatch");
    assert_eq!(output.ncols(), s * n, "Output matrix dimension mismatch");

    output.fill(T::zero());

    let phi = basis_values_buffer;

    let quadrature_iter = izip!(quadrature_weights, quadrature_points, quadrature_density);
    for (&weight, point, density) in quadrature_iter {
        let j_det = element.reference_jacobian(point).determinant();

        // First populate basis values with respect to reference coords
        element.populate_basis(phi, &point);

        let scale = weight * j_det.abs() * *density;

        // TODO: We could in fact compute the contribution from the current quadrature point
        // to the element mass matrix as
        //  (s p p^T) kron I = s (p kron I) (p kron I)^T = s p_s p_s^T,
        // where I is the s x s identity matrix, p_s = p kron I and p is the vector of basis function values
        // (p_I = phi_I). This is a (scaled) symmetric outer product for which nalgebra already provides the
        // syger kernel. However, to avoid allocation we'd need to store p_s first, and currently we don't have
        // a buffer big enough for this purpose.
        // TODO: Maybe make buffer opaque so that we could precompute p_s and use syger
        // TODO: Also contribute a kronecker routine to nalgebra that doesn't allocate, but instead stores the
        // result in a result array

        // Compute contribution for each basis function pair (phi_I, phi_J)
        for I in 0..n {
            // Fill only upper triangle, then copy over lower half at the end
            for J in I..n {
                // Scalar contribution to IJ block
                let m_IJ_contrib = scale * phi[I] * phi[J];

                // Block contribution: update diagonal entries belonging to M_IJ
                let mut M_IJ = output.slice_mut((s * I, s * J), (s, s));
                for i in 0..s {
                    M_IJ[(i, i)] += m_IJ_contrib;
                }
            }
        }
    }

    // So far we only computed the upper triangle, so fill lower triangle as well
    clone_upper_to_lower(&mut output);

    Ok(())
}

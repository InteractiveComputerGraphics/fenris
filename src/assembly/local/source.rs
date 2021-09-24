use crate::allocators::{BiDimAllocator, SmallDimAllocator, TriDimAllocator};
use crate::assembly::global::{BasisFunctionBuffer, QuadratureBuffer};
use crate::assembly::local::{ElementConnectivityAssembler, ElementVectorAssembler, QuadratureTable};
use crate::assembly::operators::Operator;
use crate::element::{MatrixSlice, ReferenceFiniteElement, VolumetricFiniteElement};
use crate::nalgebra::{
    DVectorSliceMut, DefaultAllocator, DimName, Dynamic, MatrixSliceMutMN, OPoint, OVector, RealField, Scalar, U1,
};
use crate::space::{ElementInSpace, VolumetricFiniteElementSpace};
use crate::workspace::{with_thread_local_workspace, Workspace};
use crate::SmallDim;
use itertools::izip;
use std::cell::RefCell;
use std::marker::PhantomData;

pub trait SourceFunction<T, GeometryDim>: Operator<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, Self::SolutionDim>,
{
    fn evaluate(&self, coords: &OPoint<T, GeometryDim>, data: &Self::Parameters) -> OVector<T, Self::SolutionDim>;
}

pub struct ElementSourceAssemblerBuilder<T, SpaceRef, SourceRef, QTableRef> {
    space: SpaceRef,
    source: SourceRef,
    qtable: QTableRef,
    marker: PhantomData<T>,
}

impl ElementSourceAssemblerBuilder<(), (), (), ()> {
    pub fn new() -> Self {
        Self {
            space: (),
            source: (),
            qtable: (),
            marker: PhantomData,
        }
    }
}

impl<SpaceRef, SourceRef, QTableRef> ElementSourceAssemblerBuilder<(), SpaceRef, SourceRef, QTableRef> {
    pub fn with_finite_element_space<Space>(
        self,
        space: &Space,
    ) -> ElementSourceAssemblerBuilder<(), &Space, SourceRef, QTableRef> {
        ElementSourceAssemblerBuilder {
            space,
            source: self.source,
            qtable: self.qtable,
            marker: PhantomData,
        }
    }

    pub fn with_source<Source>(
        self,
        source: &Source,
    ) -> ElementSourceAssemblerBuilder<(), SpaceRef, &Source, QTableRef> {
        ElementSourceAssemblerBuilder {
            space: self.space,
            source,
            qtable: self.qtable,
            marker: PhantomData,
        }
    }

    pub fn with_quadrature_table<QTable>(
        self,
        qtable: &QTable,
    ) -> ElementSourceAssemblerBuilder<(), SpaceRef, SourceRef, &QTable> {
        ElementSourceAssemblerBuilder {
            space: self.space,
            source: self.source,
            qtable,
            marker: PhantomData,
        }
    }
}

impl<'a, Space, Source, QTable> ElementSourceAssemblerBuilder<(), &'a Space, &'a Source, &'a QTable> {
    // TODO: It's totally weird to have T as a parameter on the function here. Can we design
    // this differently? Maybe FiniteElementSpace should actually have Scalar as an
    // associated type?
    pub fn build<T>(self) -> ElementSourceAssembler<'a, T, Space, Source, QTable> {
        ElementSourceAssembler {
            space: self.space,
            qtable: self.qtable,
            source: self.source,
            marker: PhantomData,
        }
    }
}

/// An element assembler for source functions.
///
/// TODO: Docs
pub struct ElementSourceAssembler<'a, T, Space, Source, QTable> {
    space: &'a Space,
    qtable: &'a QTable,
    source: &'a Source,
    marker: PhantomData<T>,
}

impl<'a, T, Space, Source, QTable> ElementConnectivityAssembler for ElementSourceAssembler<'a, T, Space, Source, QTable>
where
    T: Scalar,
    Space: VolumetricFiniteElementSpace<T>,
    Source: Operator<T, Space::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
{
    fn solution_dim(&self) -> usize {
        Source::SolutionDim::dim()
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

thread_local! { static SOURCE_WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default()) }

struct SourceTermWorkspace<T, D, Data>
where
    T: Scalar,
    D: SmallDim,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    quadrature_buffer: QuadratureBuffer<T, D, Data>,
    basis_buffer: BasisFunctionBuffer<T>,
}

impl<T, D, Data> Default for SourceTermWorkspace<T, D, Data>
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: SmallDimAllocator<T, D>,
{
    fn default() -> Self {
        Self {
            quadrature_buffer: QuadratureBuffer::default(),
            basis_buffer: BasisFunctionBuffer::default(),
        }
    }
}

impl<'a, T, Space, Source, QTable> ElementVectorAssembler<T> for ElementSourceAssembler<'a, T, Space, Source, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Source: SourceFunction<T, Space::ReferenceDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Source::Parameters>,
    DefaultAllocator: TriDimAllocator<T, Space::GeometryDim, Space::ReferenceDim, Source::SolutionDim>,
{
    fn assemble_element_vector_into(&self, element_index: usize, output: DVectorSliceMut<T>) -> eyre::Result<()> {
        with_thread_local_workspace(
            &SOURCE_WORKSPACE,
            |ws: &mut SourceTermWorkspace<T, Space::ReferenceDim, Source::Parameters>| {
                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                ws.basis_buffer
                    .resize(element.num_nodes(), Space::ReferenceDim::dim());
                ws.basis_buffer
                    .populate_element_nodes_from_space(element_index, self.space);
                ws.quadrature_buffer
                    .populate_element_quadrature_from_table(element_index, self.qtable);

                assemble_element_source_vector(
                    output,
                    &element,
                    self.source,
                    ws.quadrature_buffer.weights(),
                    ws.quadrature_buffer.points(),
                    ws.quadrature_buffer.data(),
                    ws.basis_buffer.element_basis_values_mut(),
                );

                Ok(())
            },
        )
    }
}

/// Assemble the local source term vector associated with a particular finite element and source.
///
/// Assembles the local vector for the provided element associated with the $(f, v)$ term in the
/// weak form of many PDEs using the given quadrature.
/// For example, the weak form of the Poisson equation (assuming suitable boundary conditions) is
/// $$ a(u, v) = (f, v) \qquad \forall v \in V,$$
/// where $f: \mathbb{R}^d \rightarrow \mathbb{R}^s$ is the *source function*.
///
/// A working array for storing basis function values must be provided.
///
/// **This is a low-level routine**. Most users will not need to call this function directly,
/// and are instead more likely to use [`ElementSourceAssembler`]. Refer to its documentation
/// for a more detailed account of source functions.
///
/// # Panics
///
/// The size of the output vector must be equal to `n * s`, where `n` is the number of
/// nodes in the element and `s` is the solution dimension.
///
/// Panics if the quadrature weights, points and data arrays do not have the same length.
///
/// The basis values buffer must have size `n`.
pub fn assemble_element_source_vector<T, Element, Source>(
    mut output: DVectorSliceMut<T>,
    element: &Element,
    source: &Source,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    quadrature_data: &[Source::Parameters],
    basis_values_buffer: &mut [T],
) where
    T: RealField,
    // We only support volumetric elements atm
    Element: VolumetricFiniteElement<T>,
    Source: SourceFunction<T, Element::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Element::GeometryDim, Source::SolutionDim>,
{
    assert_eq!(
        quadrature_weights.len(),
        quadrature_points.len(),
        "Number of quadrature weights must be equal to number of points."
    );
    assert_eq!(
        quadrature_points.len(),
        quadrature_data.len(),
        "Number of quadrature points must be equal to length of data"
    );
    assert_eq!(
        basis_values_buffer.len(),
        element.num_nodes(),
        "Number of basis functions in buffer must be equal to nodes in element."
    );

    // Reshape output into an `s x n` matrix, so that each column corresponds to the
    // output associated with a node
    let n = element.num_nodes();
    assert_eq!(
        output.len(),
        n * Source::SolutionDim::dim(),
        "Length of output vector must be consistent with number of nodes and solution dim"
    );
    let mut output =
        MatrixSliceMutMN::from_slice_generic(output.as_mut_slice(), Source::SolutionDim::name(), Dynamic::new(n));

    output.fill(T::zero());

    let quadrature_iter = izip!(quadrature_weights, quadrature_points, quadrature_data);
    for (weight, point, data) in quadrature_iter {
        element.populate_basis(&mut *basis_values_buffer, point);

        let x = element.map_reference_coords(point);
        let j = element.reference_jacobian(point);
        let f = source.evaluate(&x, data);

        // The output contribution for quadrature point q is
        //  w * |det J| * [ f_1 f_2 f_3, ... ]
        // where f_I = f * phi_I is the output associated with node I, and phi_I is the
        // basis values of node I.
        // Then the contribution is given by
        //  w * |det J| * [ f * phi_1, f * phi_2, ... ] = w * |det J| * f * phi,
        // where phi is a row vector of basis values
        let phi = MatrixSlice::from_slice_generic(&*basis_values_buffer, U1::name(), Dynamic::new(n));
        output.gemm(*weight * j.determinant().abs(), &f, &phi, T::one());
    }
}

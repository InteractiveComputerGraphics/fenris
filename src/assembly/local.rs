use crate::connectivity::Connectivity;
use crate::mesh::Mesh;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::DVectorSliceMut;
use crate::nalgebra::{DMatrixSliceMut, DefaultAllocator, DimName, Scalar};

mod elliptic;
mod mass;
mod quadrature_table;
mod source;

pub use elliptic::*;
pub use mass::*;
pub use quadrature_table::*;
pub use source::*;

pub trait ElementConnectivityAssembler {
    fn solution_dim(&self) -> usize;

    fn num_elements(&self) -> usize;

    fn num_nodes(&self) -> usize;

    fn element_node_count(&self, element_index: usize) -> usize;

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize);

    /// Returns an adapter that modifies element node indices according to the provided function.
    ///
    /// In general, changing the node indices is often accompanied by a change in the total number of nodes.
    /// Therefore the new total number of nodes has to be provided.
    ///
    /// This is often used to enlarge the index space and populate only parts of a matrix.
    /// For example, we might use an element assembler for a single body and offset its indices so that
    /// we can assemble directly into a larger matrix containing the results of multiple bodies.
    fn map_element_nodes<F>(self, new_num_nodes: usize, f: F) -> MapElementNodes<Self, F>
    where
        Self: Sized,
    {
        MapElementNodes {
            mapped: self,
            function: f,
            num_nodes: new_num_nodes
        }
    }
}

impl<T, D, C> ElementConnectivityAssembler for Mesh<T, D, C>
where
    T: Scalar,
    D: DimName,
    C: Connectivity,
    DefaultAllocator: Allocator<T, D>,
{
    fn solution_dim(&self) -> usize {
        1
    }

    fn num_elements(&self) -> usize {
        self.connectivity().len()
    }

    fn num_nodes(&self) -> usize {
        self.vertices().len()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.connectivity()[element_index].vertex_indices().len()
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        output.copy_from_slice(self.connectivity()[element_index].vertex_indices());
    }
}

pub trait ElementMatrixAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_matrix_into(&self, element_index: usize, output: DMatrixSliceMut<T>) -> eyre::Result<()>;
}

pub trait ElementVectorAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_vector_into(&self, element_index: usize, output: DVectorSliceMut<T>) -> eyre::Result<()>;
}

pub trait ElementScalarAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T>;
}

#[derive(Debug, Clone)]
pub struct MapElementNodes<Mapped, F> {
    mapped: Mapped,
    function: F,
    num_nodes: usize,
}

impl<Assembler, F> ElementConnectivityAssembler for MapElementNodes<Assembler, F>
where
    Assembler: ElementConnectivityAssembler,
    F: Fn(usize) -> usize,
{
    fn solution_dim(&self) -> usize {
        self.mapped.solution_dim()
    }

    fn num_elements(&self) -> usize {
        self.mapped.num_elements()
    }

    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.mapped.element_node_count(element_index)
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        self.mapped.populate_element_nodes(output, element_index);
        for idx in output {
            *idx = (self.function)(*idx);
        }
    }
}

impl<T, Assembler, F> ElementScalarAssembler<T> for MapElementNodes<Assembler, F>
where
    T: Scalar,
    Assembler: ElementScalarAssembler<T>,
    F: Fn(usize) -> usize,
{
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T> {
        self.mapped.assemble_element_scalar(element_index)
    }
}

impl<T, Assembler, F> ElementVectorAssembler<T> for MapElementNodes<Assembler, F>
where
    T: Scalar,
    Assembler: ElementVectorAssembler<T>,
    F: Fn(usize) -> usize,
{
    fn assemble_element_vector_into(&self, element_index: usize, output: DVectorSliceMut<T>) -> eyre::Result<()> {
        self.mapped
            .assemble_element_vector_into(element_index, output)
    }
}

impl<T, Assembler, F> ElementMatrixAssembler<T> for MapElementNodes<Assembler, F>
where
    T: Scalar,
    Assembler: ElementMatrixAssembler<T>,
    F: Fn(usize) -> usize,
{
    fn assemble_element_matrix_into(&self, element_index: usize, output: DMatrixSliceMut<T>) -> eyre::Result<()> {
        self.mapped
            .assemble_element_matrix_into(element_index, output)
    }
}

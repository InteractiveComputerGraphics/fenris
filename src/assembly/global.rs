use crate::allocators::{ElementConnectivityAllocator, FiniteElementMatrixAllocator};
use crate::assembly::local::{
    ElementConnectivityAssembler, ElementMatrixAssembler, ElementMatrixTransformation,
    GeneralizedEllipticContraction, GeneralizedEllipticOperator,
    GeneralizedStiffnessElementAssembler,
};
use crate::assembly::{local, NoTransformation, QuadratureTable};
use crate::connectivity::Connectivity;
use crate::element::{ElementConnectivity, MatrixSlice, MatrixSliceMut, ReferenceFiniteElement};
use crate::util::coerce_col_major_slice_mut;

use nalgebra::base::allocator::Allocator;
use nalgebra::base::storage::Storage;
use nalgebra::{
    ComplexField, DMatrix, DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut,
    DefaultAllocator, Dim, DimMin, DimName, Dynamic, Matrix, Point, RealField, Scalar,
    SquareMatrix, U1,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;
use serde::__private::PhantomData;
use thread_local::ThreadLocal;

use fenris_sparse::{CooMatrix, CsrMatrix, CsrRowMut, SparsityPattern};
use nested_vec::NestedVec;
use paradis::adapter::BlockAdapter;
use paradis::coloring::sequential_greedy_coloring;
use paradis::DisjointSubsets;

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::error::Error;
use std::sync::Arc;

/// An assembler for CSR matrices.
#[derive(Debug, Clone)]
pub struct CsrAssembler<T: Scalar> {
    // All members are buffers that help prevent unnecessary allocations
    // when assembling multiple matrices with the same assembler
    workspace: RefCell<CsrAssemblerWorkspace<T>>,
}

impl<T: Scalar> Default for CsrAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: RefCell::new(CsrAssemblerWorkspace::default()),
        }
    }
}

#[derive(Debug, Clone)]
struct CsrAssemblerWorkspace<T: Scalar> {
    // All members are buffers that help prevent unnecessary allocations
    // when assembling multiple matrices with the same assembler
    connectivity_permutation: Vec<usize>,
    element_global_nodes: Vec<usize>,
    element_matrix: DMatrix<T>,
}

impl<T: Scalar> Default for CsrAssemblerWorkspace<T> {
    fn default() -> Self {
        Self {
            connectivity_permutation: Vec::new(),
            element_global_nodes: Vec::new(),
            element_matrix: DMatrix::from_row_slice(0, 0, &[]),
        }
    }
}

impl<T: Scalar> CsrAssembler<T> {
    // TODO: Test this method!
    pub fn assemble_pattern(
        &self,
        element_assembler: &dyn ElementConnectivityAssembler,
    ) -> SparsityPattern {
        // Here we optimize for memory usage rather than performance: by collecting into a
        // BTreeSet we store each matrix entry exactly once. This is important, because depending
        // on the mesh, there may be a relatively large number of duplicate entries which would
        // need to be combined.
        let sdim = element_assembler.solution_dim();
        let mut matrix_entries = BTreeSet::new();
        let mut element_global_nodes = Vec::new();
        for i in 0..element_assembler.num_elements() {
            let element_node_count = element_assembler.element_node_count(i);
            element_global_nodes.resize(element_node_count, usize::MAX);
            element_assembler.populate_element_nodes(&mut element_global_nodes, i);

            for node_i in &element_global_nodes {
                for node_j in &element_global_nodes {
                    for s_i in 0..sdim {
                        for s_j in 0..sdim {
                            let idx_i = sdim * node_i + s_i;
                            let idx_j = sdim * node_j + s_j;
                            matrix_entries.insert((idx_i, idx_j));
                        }
                    }
                }
            }
        }

        let num_rows = sdim * element_assembler.num_nodes();
        let mut offsets = Vec::with_capacity(num_rows + 1);
        let mut column_indices = Vec::with_capacity(matrix_entries.len());

        offsets.push(0);
        for (i, j) in matrix_entries {
            while i + 1 > offsets.len() {
                // This condition indicates that we have reached a new row. We need to run this
                // in a while loop to correctly handle consecutive empty rows
                offsets.push(column_indices.len());
            }
            column_indices.push(j);
        }

        // Make sure we fill out the remaining offsets if the last rows are empty
        while offsets.len() < (num_rows + 1) {
            offsets.push(column_indices.len());
        }

        SparsityPattern::from_offsets_and_indices(num_rows, num_rows, offsets, column_indices)
    }
}

impl<T: RealField> CsrAssembler<T> {
    pub fn assemble(
        &self,
        element_assembler: &dyn ElementMatrixAssembler<T>,
    ) -> Result<CsrMatrix<T>, Box<dyn Error + Send + Sync>> {
        let pattern = self.assemble_pattern(element_assembler.as_connectivity_assembler());
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix =
            CsrMatrix::from_pattern_and_values(Arc::new(pattern), initial_matrix_values);
        self.assemble_into_csr(&mut matrix, element_assembler)?;
        Ok(matrix)
    }

    pub fn assemble_into_csr(
        &self,
        csr: &mut CsrMatrix<T>,
        element_assembler: &dyn ElementMatrixAssembler<T>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Reuse previously allocated buffers
        let ws = &mut *self.workspace.borrow_mut();
        let connectivity_permutation = &mut ws.connectivity_permutation;
        let element_global_nodes = &mut ws.element_global_nodes;
        let element_matrix = &mut ws.element_matrix;

        let sdim = element_assembler.solution_dim();

        for i in 0..element_assembler.num_elements() {
            let element_node_count = element_assembler.element_node_count(i);
            let element_matrix_dim = sdim * element_node_count;

            element_global_nodes.resize(element_node_count, 0);
            element_matrix.resize_mut(element_matrix_dim, element_matrix_dim, T::zero());
            element_matrix.fill(T::zero());

            let matrix_slice = DMatrixSliceMut::from(&mut *element_matrix);
            element_assembler.assemble_element_matrix_into(i, matrix_slice)?;
            element_assembler.populate_element_nodes(element_global_nodes, i);

            connectivity_permutation.clear();
            connectivity_permutation.extend(0..element_node_count);
            connectivity_permutation.sort_unstable_by_key(|i| element_global_nodes[*i]);

            for (local_node_idx, global_node_idx) in element_global_nodes.iter().enumerate() {
                for i in 0..sdim {
                    let local_row_index = sdim * local_node_idx + i;
                    let global_row_index = sdim * *global_node_idx + i;
                    let mut csr_row = csr.row_mut(global_row_index);

                    let a_row = element_matrix.row(local_row_index);
                    add_element_row_to_csr_row(
                        &mut csr_row,
                        &element_global_nodes,
                        &connectivity_permutation,
                        sdim,
                        &a_row,
                    );
                }
            }
        }

        Ok(())
    }
}

/// A parallel assembler for CSR matrices relying on a graph coloring of elements.
///
/// TODO: Consider using type erasure to store buffers without needing the generic type parameter
#[derive(Debug)]
pub struct CsrParAssembler<T: Scalar + Send> {
    workspace: ThreadLocal<RefCell<CsrAssemblerWorkspace<T>>>,
}

impl<T: Scalar + Send> Default for CsrParAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: Default::default(),
        }
    }
}

impl<T: Scalar + Send> CsrParAssembler<T> {
    pub fn assemble_pattern(
        &self,
        element_assembler: &(dyn Sync + ElementConnectivityAssembler),
    ) -> SparsityPattern {
        let sdim = element_assembler.solution_dim();

        // Count number of (including duplicate) triplets
        let num_total_triplets = (0..element_assembler.num_elements())
            .into_par_iter()
            .with_min_len(50)
            .map(|element_idx| {
                let num_entries = sdim * element_assembler.element_node_count(element_idx);
                num_entries * num_entries
            })
            .sum();

        // TODO: Can we do this next stage in parallel somehow?
        // (it is however entirely memory bound, but a single thread
        // probably cannot exhaust that on its own)
        let mut coordinates = Vec::with_capacity(num_total_triplets);
        let mut index_workspace = Vec::new();
        for element_idx in 0..element_assembler.num_elements() {
            let node_count = element_assembler.element_node_count(element_idx);
            index_workspace.resize(node_count, 0);
            element_assembler.populate_element_nodes(&mut index_workspace, element_idx);

            for node_i in &index_workspace {
                for node_j in &index_workspace {
                    for i in 0..sdim {
                        for j in 0..sdim {
                            coordinates.push((sdim * node_i + i, sdim * node_j + j));
                        }
                    }
                }
            }
        }

        coordinates.par_sort_unstable();

        // TODO: Can we parallelize the final part?
        // TODO: move this into something like SparsityPattern::from_coordinates ?
        // But then we'd probably also have to deal with the case in which
        // the coordinates are perhaps not sorted (either error out or
        // deal with it on the fly)
        let num_rows = sdim * element_assembler.num_nodes();
        let mut row_offsets = Vec::with_capacity(num_rows);
        let mut column_indices = Vec::new();
        row_offsets.push(0);

        let mut coord_iter = coordinates.into_iter();
        let mut current_row = 0;
        let mut prev_col = None;

        while let Some((i, j)) = coord_iter.next() {
            assert!(i < num_rows, "Coordinates must be in bounds");

            while i > current_row {
                row_offsets.push(column_indices.len());
                current_row += 1;
                prev_col = None;
            }

            // Only add column if it is not a duplicate
            if Some(j) != prev_col {
                column_indices.push(j);
                prev_col = Some(j);
            }
        }

        // Fill out offsets for remaining empty rows
        for _ in current_row..num_rows {
            row_offsets.push(column_indices.len());
        }

        SparsityPattern::from_offsets_and_indices(num_rows, num_rows, row_offsets, column_indices)
    }
}

impl<T: RealField + Send> CsrParAssembler<T> {
    pub fn assemble_into_csr(
        &self,
        csr: &mut CsrMatrix<T>,
        colors: &[DisjointSubsets],
        element_assembler: &(dyn Sync + ElementMatrixAssembler<T>),
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let sdim = element_assembler.solution_dim();

        for color in colors {
            let mut block_adapter = BlockAdapter::with_block_size(csr, sdim);
            color
                .subsets_par_iter(&mut block_adapter)
                .map(|mut subset| {
                    let ws = &mut *self.workspace.get_or_default().borrow_mut();

                    let element_index = subset.label();
                    let element_node_count = element_assembler.element_node_count(element_index);
                    let element_matrix_dim = sdim * element_node_count;

                    ws.element_global_nodes.resize(element_node_count, 0);
                    ws.element_matrix
                        .resize_mut(element_matrix_dim, element_matrix_dim, T::zero());
                    ws.element_matrix.fill(T::zero());

                    let matrix_slice = DMatrixSliceMut::from(&mut ws.element_matrix);
                    element_assembler.assemble_element_matrix_into(element_index, matrix_slice)?;
                    element_assembler
                        .populate_element_nodes(&mut ws.element_global_nodes, element_index);
                    debug_assert_eq!(subset.global_indices(), ws.element_global_nodes.as_slice());

                    {
                        let element_global_nodes = &ws.element_global_nodes;
                        ws.connectivity_permutation.clear();
                        ws.connectivity_permutation.extend(0..element_node_count);
                        ws.connectivity_permutation
                            .sort_unstable_by_key(|i| element_global_nodes[*i]);
                    }

                    for local_node_idx in 0..element_node_count {
                        let mut csr_block_row = subset.get_mut(local_node_idx);
                        for i in 0..sdim {
                            let local_row_index = sdim * local_node_idx + i;
                            let mut csr_row = csr_block_row.get_mut(i).unwrap();

                            let a_row = ws.element_matrix.row(local_row_index);
                            add_element_row_to_csr_row(
                                &mut csr_row,
                                &ws.element_global_nodes,
                                &ws.connectivity_permutation,
                                sdim,
                                &a_row,
                            );
                        }
                    }

                    Ok(())
                })
                .collect::<Result<(), Box<dyn Error + Send + Sync>>>()?;
        }

        Ok(())
    }
}

pub fn apply_homogeneous_dirichlet_bc_csr<T, SolutionDim>(
    matrix: &mut CsrMatrix<T>,
    nodes: &[usize],
) where
    T: RealField,
    SolutionDim: DimName,
{
    let d = SolutionDim::dim();

    // Determine an appropriately scale element to put on the diagonal
    // (Simply setting 1 would ignore the scaling of the entries of the matrix, leading
    // to potentially poor condition numbers)

    // Here we just take the first non-zero diagonal entry as a representative scale.
    // This is cheap and I think reasonably safe option
    let scale = matrix
        .diag_iter()
        .skip_while(|&x| x == T::zero())
        .map(|x| x.abs())
        .next()
        .unwrap_or(T::one());

    // We need to do the following:
    //  - zero all rows corresponding to Dirichlet nodes
    //  - zero all columns corresponding to Dirichlet nodes
    //  - set diagonal entries corresponding to Dirichlet nodes to a non-zero value
    // In order to zero all columns, a naive approach would need to visit all elements in the matrix,
    // which might be very expensive.
    // Instead, we can exploit symmetry to determine that if we visit column j in row i,
    // where i corresponds to a Dirichlet node, we would also need to visit row j in order
    // to zero out columns.

    let mut dirichlet_membership = vec![false; d * matrix.nrows()];
    let mut rows_to_visit = vec![false; d * matrix.nrows()];

    for &node in nodes {
        for i in 0..d {
            let row_idx = d * node + i;
            dirichlet_membership[row_idx] = true;
            let mut row = matrix.row_mut(row_idx);
            let (cols, values) = row.columns_and_values_mut();

            for (&col_idx, val) in cols.iter().zip(values) {
                if col_idx == row_idx {
                    *val = scale;
                } else {
                    *val = T::zero();
                    // If we need to zero out (r, c), then we also need to zero out (c, r),
                    // so we need to visit column c in r later
                    rows_to_visit[col_idx] = true;
                }
            }
        }
    }

    let row_visit_iter = rows_to_visit
        .iter()
        .enumerate()
        .filter_map(|(index, &should_visit)| if should_visit { Some(index) } else { None });
    for row_index in row_visit_iter {
        let row_is_dirichlet = dirichlet_membership[row_index];
        if !row_is_dirichlet {
            let mut row = matrix.row_mut(row_index);
            let (cols, values) = row.columns_and_values_mut();
            for (local_idx, &global_idx) in cols.iter().enumerate() {
                let col_is_dirichlet = dirichlet_membership[global_idx];
                if col_is_dirichlet {
                    values[local_idx] = T::zero();
                }
            }
        }
    }
}

pub fn apply_homogeneous_dirichlet_bc_matrix<T, SolutionDim>(
    matrix: &mut DMatrix<T>,
    nodes: &[usize],
) where
    T: RealField,
    SolutionDim: DimName,
{
    let d = SolutionDim::dim();

    // Determine an appropriately scale element to put on the diagonal
    // (Simply setting 1 would ignore the scaling of the entries of the matrix, leading
    // to potentially poor condition numbers)
    let scale = matrix
        .diagonal()
        .map(|x| x.abs())
        .fold(T::zero(), |a, b| a + b)
        / T::from_usize(matrix.nrows()).unwrap();

    for node in nodes {
        for i in 0..d {
            let idx = d * node + i;
            matrix.index_mut((.., idx)).fill(T::zero());
            matrix.index_mut((idx, ..)).fill(T::zero());
            *matrix.index_mut((idx, idx)) = scale;
        }
    }
}

pub fn apply_homogeneous_dirichlet_bc_rhs<'a, T>(
    rhs: impl Into<DVectorSliceMut<'a, T>>,
    nodes: &[usize],
    solution_dim: usize,
) where
    T: RealField,
{
    let mut rhs = rhs.into();
    let d = solution_dim;

    for node in nodes {
        for i in 0..d {
            let idx = d * node + i;
            *rhs.index_mut(idx) = T::zero();
        }
    }
}

pub fn assemble_generalized_elliptic_term_into_par<'a, T, SolutionDim, Connectivity>(
    mut f: DVectorSliceMut<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    g: &(impl Sync + GeneralizedEllipticOperator<T, SolutionDim, Connectivity::GeometryDim>),
    u: impl Into<DVectorSlice<'a, T>>,
    quadrature_table: &(impl Sync + QuadratureTable<T, Connectivity::GeometryDim>),
    colors: &[DisjointSubsets],
) where
    T: RealField,
    Connectivity: Sync
        + ElementConnectivity<T, ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim>,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: ElementConnectivityAllocator<T, Connectivity>
        + Allocator<T, Connectivity::GeometryDim, SolutionDim>
        + Allocator<T, SolutionDim, Connectivity::GeometryDim>,
    <DefaultAllocator as Allocator<T, Connectivity::GeometryDim>>::Buffer: Sync,
{
    let u = u.into();
    let f_slice = f.as_mut_slice();

    for color in colors {
        let mut block_adapter = BlockAdapter::with_block_size(f_slice, SolutionDim::dim());
        color
            .subsets_par_iter(&mut block_adapter)
            .for_each(|mut subset| {
                let connectivity_idx = subset.label();
                let connectivity = &connectivity[connectivity_idx];
                debug_assert_eq!(subset.global_indices(), connectivity.vertex_indices());
                let element = connectivity
                    .element(vertices)
                    .expect("All vertices of element are assumed to be in bounds.");
                // TODO: Don't allocate!
                let mut u_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
                connectivity.populate_element_variables(
                    MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element),
                    u,
                );

                // TODO: Don't allocate!
                let mut f_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
                local::assemble_generalized_element_elliptic_term(
                    MatrixSliceMut::from(&mut f_element),
                    &element,
                    g,
                    &MatrixSlice::from(&u_element),
                    &quadrature_table.quadrature_for_element(connectivity_idx),
                );

                let sdim = SolutionDim::dim();
                for local_idx in 0..connectivity.vertex_indices().len() {
                    let mut block = subset.get_mut(local_idx);
                    let f_col = f_element.fixed_slice::<SolutionDim, U1>(0, local_idx);
                    for i in 0..sdim {
                        *block.index_mut(i) += f_col[i];
                    }
                }
            });
    }
}

pub fn assemble_generalized_elliptic_term_into<'a, T, SolutionDim, Connectivity>(
    mut f: DVectorSliceMut<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    g: &impl GeneralizedEllipticOperator<T, SolutionDim, Connectivity::GeometryDim>,
    u: impl Into<DVectorSlice<'a, T>>,
    // TODO: Introduce trait for the "element quadrature map"?
    quadrature_table: &impl QuadratureTable<T, Connectivity::GeometryDim>,
) where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: ElementConnectivityAllocator<T, Connectivity>
        + Allocator<T, Connectivity::GeometryDim, SolutionDim>
        + Allocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    let u = u.into();
    for (i, connectivity) in connectivity.iter().enumerate() {
        let element = connectivity.element(vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );
        // TODO: Don't allocate!
        let mut u_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
        connectivity.populate_element_variables(
            MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element),
            u,
        );

        // TODO: Don't allocate!
        let mut f_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
        local::assemble_generalized_element_elliptic_term(
            MatrixSliceMut::from(&mut f_element),
            &element,
            g,
            &MatrixSlice::from(&u_element),
            &quadrature_table.quadrature_for_element(i),
        );

        distribute_local_to_global_vector(
            DVectorSliceMut::from(&mut f),
            connectivity.vertex_indices(),
            &f_element,
        );
    }
}

/// Add a row of a local element matrix to the provided row of a CSR matrix.
///
/// `node_connectivity`: The global indices of nodes.
/// `sorted_permutation`: The local indices of nodes in the element, ordered such that the
///    corresponding global indices are sorted.
/// `dim`: The solution dimension.
/// `local_row`: The local row of the element matrix that should be added to the CSR matrix.
fn add_element_row_to_csr_row<T, S>(
    row: &mut CsrRowMut<T>,
    node_connectivity: &[usize],
    sorted_permutation: &[usize],
    dim: usize,
    local_row: &Matrix<T, U1, Dynamic, S>,
) where
    T: RealField,
    S: Storage<T, U1, Dynamic>,
{
    assert_eq!(node_connectivity.len(), sorted_permutation.len());
    assert_eq!(node_connectivity.len() * dim, local_row.ncols());
    assert!(dim >= 1);

    let (column_indices, values) = row.columns_and_values_mut();

    let mut csr_col_idx_iter = column_indices.iter().copied().enumerate();

    for &node_local_idx in sorted_permutation {
        let node_global_idx = node_connectivity[node_local_idx];

        for i in 0..dim {
            let local_col_idx = dim * node_local_idx + i;
            let global_col_index = dim * node_global_idx + i;

            // TODO: If the CSR matrix has a large number of entries in each row,
            // an exponential search may be faster than a linear search as we do here
            let (local_csr_col_idx, _) = csr_col_idx_iter
                .find(|(_, csr_col_idx)| *csr_col_idx == global_col_index)
                .expect("Could not find column index associated with node in CSR row");
            values[local_csr_col_idx] += local_row[local_col_idx];
        }
    }
}

pub fn assemble_generalized_stiffness_into_csr<T, SolutionDim, Connectivity>(
    csr: &mut CsrMatrix<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &impl GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>,
    u: &DVector<T>,
    quadrature_table: &impl QuadratureTable<T, Connectivity::GeometryDim>,
) where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    assemble_transformed_generalized_stiffness_into_csr(
        csr,
        vertices,
        connectivity,
        contraction,
        u,
        quadrature_table,
        &NoTransformation,
    )
}

// TODO: Remove this function in favor of using separate global and element assemblers
pub fn assemble_transformed_generalized_stiffness_into_csr<T, SolutionDim, Connectivity>(
    csr: &mut CsrMatrix<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &impl GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>,
    u: &DVector<T>,
    quadrature_table: &impl QuadratureTable<T, Connectivity::GeometryDim>,
    transformation: &(impl ?Sized + ElementMatrixTransformation<T>),
) where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    let element_assembler = GeneralizedStiffnessElementAssembler {
        vertices,
        connectivity,
        contraction,
        u,
        quadrature_table,
        transformation,
        solution_dim_marker: PhantomData,
    };

    let csr_assembler = CsrAssembler::default();
    csr_assembler
        .assemble_into_csr(csr, &element_assembler)
        .expect("TODO: Propagate error")
}

pub fn assemble_generalized_stiffness_into_csr_par<T, SolutionDim, Connectivity>(
    csr: &mut CsrMatrix<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &(impl Sync
          + GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, Connectivity::GeometryDim>),
    colors: &[DisjointSubsets],
) where
    T: RealField,
    Connectivity: Sync
        + ElementConnectivity<T, ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim>,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
    <DefaultAllocator as Allocator<T, Connectivity::GeometryDim>>::Buffer: Sync,
{
    assemble_transformed_generalized_stiffness_into_csr_par(
        csr,
        vertices,
        connectivity,
        contraction,
        u,
        quadrature_table,
        &NoTransformation,
        colors,
    )
}

pub fn assemble_transformed_generalized_stiffness_into_csr_par<T, SolutionDim, Connectivity>(
    csr: &mut CsrMatrix<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &(impl Sync
          + GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, Connectivity::GeometryDim>),
    transformation: &(impl ?Sized + Sync + ElementMatrixTransformation<T>),
    colors: &[DisjointSubsets],
) where
    T: RealField,
    Connectivity: Sync
        + ElementConnectivity<T, ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim>,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
    <DefaultAllocator as Allocator<T, Connectivity::GeometryDim>>::Buffer: Sync,
{
    let element_assembler = GeneralizedStiffnessElementAssembler {
        vertices,
        connectivity,
        contraction,
        u,
        quadrature_table,
        transformation,
        solution_dim_marker: PhantomData,
    };

    let csr_assembler = CsrParAssembler::default();
    csr_assembler
        .assemble_into_csr(csr, colors, &element_assembler)
        .expect("TODO: Propagate error")
}

pub fn assemble_transformed_generalized_stiffness_into<T, SolutionDim, Connectivity>(
    coo: &mut CooMatrix<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &impl GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>,
    u: &DVector<T>,
    quadrature_table: &impl QuadratureTable<T, Connectivity::GeometryDim>,
    transformation: &(impl ?Sized + ElementMatrixTransformation<T>),
) where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    for (i, connectivity) in connectivity.iter().enumerate() {
        let element = connectivity.element(vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );

        // TODO: Don't allocate!
        let mut u_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
        connectivity.populate_element_variables(
            MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element),
            u,
        );

        // TODO: Don't allocate!
        let matrix_size = SolutionDim::dim() * element.num_nodes();
        let mut a_element = DMatrix::zeros(matrix_size, matrix_size);

        local::assemble_generalized_element_stiffness(
            DMatrixSliceMut::from(&mut a_element),
            &element,
            contraction,
            MatrixSlice::from(&u_element),
            &quadrature_table.quadrature_for_element(i),
        );

        let a_rows = a_element.nrows();
        let a_cols = a_element.ncols();
        // TODO: Shouldn't need this coercion anymore
        let mut a_dynamic_slice_mut =
            coerce_col_major_slice_mut(&mut a_element, Dynamic::new(a_rows), Dynamic::new(a_cols));
        transformation.transform_element_matrix(&mut a_dynamic_slice_mut);
        distribute_local_to_global(coo, connectivity.vertex_indices(), &a_element);
    }
}

pub fn assemble_generalized_stiffness_into<T, SolutionDim, Connectivity>(
    coo: &mut CooMatrix<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &impl GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>,
    u: &DVector<T>,
    quadrature_table: &impl QuadratureTable<T, Connectivity::GeometryDim>,
) where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    assemble_transformed_generalized_stiffness_into(
        coo,
        vertices,
        connectivity,
        contraction,
        u,
        quadrature_table,
        &NoTransformation,
    )
}

pub fn assemble_generalized_stiffness<T, SolutionDim, Connectivity>(
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &impl GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>,
    u: &DVector<T>,
    quadrature_table: &impl QuadratureTable<T, Connectivity::GeometryDim>,
) -> CooMatrix<T>
where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    let ndof = vertices.len() * SolutionDim::dim();
    let mut coo = CooMatrix::new(ndof, ndof);
    assemble_generalized_stiffness_into(
        &mut coo,
        vertices,
        connectivity,
        contraction,
        u,
        quadrature_table,
    );
    coo
}

pub fn assemble_generalized_stiffness_par<T, SolutionDim, Connectivity>(
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &(impl Sync
          + GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, Connectivity::GeometryDim>),
) -> CooMatrix<T>
where
    T: RealField,
    Connectivity: Sync
        + ElementConnectivity<T, ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim>,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
    <DefaultAllocator as Allocator<T, Connectivity::GeometryDim>>::Buffer: Sync,
{
    assemble_transformed_generalized_stiffness_par(
        vertices,
        connectivity,
        contraction,
        u,
        quadrature_table,
        &NoTransformation,
    )
}

pub fn assemble_transformed_generalized_stiffness_par<T, SolutionDim, Connectivity>(
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    contraction: &(impl Sync
          + GeneralizedEllipticContraction<T, SolutionDim, Connectivity::GeometryDim>),
    u: &DVector<T>,
    quadrature_table: &(impl Sync + QuadratureTable<T, Connectivity::GeometryDim>),
    transformation: &(impl ?Sized + Sync + ElementMatrixTransformation<T>),
) -> CooMatrix<T>
where
    T: RealField,
    Connectivity: Sync
        + ElementConnectivity<T, ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim>,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
    <DefaultAllocator as Allocator<T, Connectivity::GeometryDim>>::Buffer: Sync,
{
    let ndof = vertices.len() * SolutionDim::dim();
    connectivity
        .par_iter()
        .enumerate()
        .fold(
            || CooMatrix::new(ndof, ndof),
            |mut coo, (i, connectivity)| {
                let element = connectivity.element(vertices).expect(
                    "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
                );
                // TODO: Don't allocate!
                let mut u_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
                connectivity.populate_element_variables(
                    MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element),
                    u,
                );

                // TODO: Don't allocate!
                let matrix_size = SolutionDim::dim() * element.num_nodes();
                let mut a_element = DMatrix::zeros(matrix_size, matrix_size);

                local::assemble_generalized_element_stiffness(
                    DMatrixSliceMut::from(&mut a_element),
                    &element,
                    contraction,
                    MatrixSlice::from(&u_element),
                    &quadrature_table.quadrature_for_element(i),
                );

                let a_rows = a_element.nrows();
                let a_cols = a_element.ncols();
                let mut a_dynamic_slice_mut = coerce_col_major_slice_mut(
                    &mut a_element,
                    Dynamic::new(a_rows),
                    Dynamic::new(a_cols),
                );
                transformation.transform_element_matrix(&mut a_dynamic_slice_mut);
                distribute_local_to_global(&mut coo, connectivity.vertex_indices(), &a_element);
                coo
            },
        )
        .reduce_with(|mut coo1, coo2| {
            // TODO: As a slight optimization, we might consider making sure that the
            // matrix with the largest number of nonzeros is on the left-hand side
            coo1 += &coo2;
            coo1
        })
        .unwrap_or_else(|| CooMatrix::new(ndof, ndof))
}

pub fn assemble_generalized_mass_into<T, SolutionDim, Connectivity, Table>(
    coo: &mut CooMatrix<T>,
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    // TODO: Generalize density somehow? Attach properties to quadrature points?
    density: T,
    quadrature_table: &Table,
) where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    Table: QuadratureTable<T, Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    for (i, connectivity) in connectivity.iter().enumerate() {
        let element = connectivity.element(vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );
        // TODO: Don't allocate!
        let matrix_size = SolutionDim::dim() * element.num_nodes();
        let mut m_element = DMatrix::zeros(matrix_size, matrix_size);
        local::assemble_generalized_element_mass(
            DMatrixSliceMut::from(&mut m_element),
            &element,
            density,
            &quadrature_table.quadrature_for_element(i),
        );

        distribute_local_to_global(coo, connectivity.vertex_indices(), &m_element);
    }
}

pub fn assemble_generalized_mass<T, SolutionDim, Connectivity, Table>(
    vertices: &[Point<T, Connectivity::GeometryDim>],
    connectivity: &[Connectivity],
    // TODO: Generalize density somehow? Attach properties to quadrature points?
    density: T,
    quadrature_table: &Table,
) -> CooMatrix<T>
where
    T: RealField,
    Connectivity: ElementConnectivity<
        T,
        ReferenceDim = <Connectivity as ElementConnectivity<T>>::GeometryDim,
    >,
    Connectivity::GeometryDim:
        DimMin<Connectivity::GeometryDim, Output = Connectivity::GeometryDim>,
    Table: QuadratureTable<T, Connectivity::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Connectivity::GeometryDim>,
{
    let ndof = vertices.len() * SolutionDim::dim();
    let mut coo = CooMatrix::new(ndof, ndof);
    assemble_generalized_mass_into(&mut coo, vertices, connectivity, density, quadrature_table);
    coo
}

pub fn color_nodes<C: Connectivity>(connectivity: &[C]) -> Vec<DisjointSubsets> {
    let mut nested = NestedVec::new();

    for conn in connectivity {
        nested.push(conn.vertex_indices());
    }

    sequential_greedy_coloring(&nested)
}

fn distribute_local_to_global<T, D, S>(
    coo: &mut CooMatrix<T>,
    global_indices: &[usize],
    element_matrix: &SquareMatrix<T, D, S>,
) where
    T: ComplexField,
    D: Dim,
    S: Storage<T, D, D>,
{
    assert_eq!(
        element_matrix.nrows() % global_indices.len(),
        0,
        "Element matrix must have number of rows/cols divisible by \
         number of nodes in element."
    );
    let dim = element_matrix.nrows() / global_indices.len();
    // Distribute values from element matrix to global matrix in the form of triplets
    for (i_local, i_global) in global_indices.iter().enumerate() {
        for (j_local, j_global) in global_indices.iter().enumerate() {
            for i in 0..dim {
                for j in 0..dim {
                    coo.push(
                        dim * i_global + i,
                        dim * j_global + j,
                        *element_matrix.index((dim * i_local + i, dim * j_local + j)),
                    );
                }
            }
        }
    }
}

fn distribute_local_to_global_vector<T, Dimension, Nodes, S>(
    mut vector: DVectorSliceMut<T>,
    global_indices: &[usize],
    element_vectors: &Matrix<T, Dimension, Nodes, S>,
) where
    T: ComplexField,
    Dimension: Dim,
    Nodes: Dim,
    S: Storage<T, Dimension, Nodes>,
{
    assert_eq!(
        element_vectors.ncols(),
        global_indices.len(),
        "Number of elements vectors must be same as number of nodes in element."
    );
    let dim = element_vectors.nrows();
    for (i_local, i_global) in global_indices.iter().enumerate() {
        for i in 0..dim {
            vector[dim * i_global + i] += element_vectors[dim * i_local + i];
        }
    }
}

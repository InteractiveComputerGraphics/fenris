use std::cell::RefCell;
use std::collections::BTreeSet;
use std::error::Error;
use std::ops::AddAssign;

use itertools::izip;
use nalgebra::base::storage::Storage;
use nalgebra::{
    DMatrix, DMatrixSliceMut, DVectorSliceMut, DimName, Dynamic, Matrix, RealField, Scalar, U1,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use thread_local::ThreadLocal;

use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use nested_vec::NestedVec;
use paradis::adapter::BlockAdapter;
use paradis::coloring::sequential_greedy_coloring;
use paradis::DisjointSubsets;

use crate::allocators::{BiDimAllocator, SmallDimAllocator};
use crate::assembly::local::{
    ElementConnectivityAssembler, ElementMatrixAssembler, ElementVectorAssembler, QuadratureTable,
};
use crate::connectivity::Connectivity;
use crate::element::MatrixSliceMut;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{DVector, DVectorSlice, DefaultAllocator, Point};
use crate::space::FiniteElementSpace;
use crate::SmallDim;
use fenris_sparse::ParallelCsrRowCollection;

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

        // TODO: Avoid validation?
        SparsityPattern::try_from_offsets_and_indices(num_rows, num_rows, offsets, column_indices)
            .expect("Pattern data must be valid")
    }
}

impl<T: RealField> CsrAssembler<T> {
    pub fn assemble(
        &self,
        element_assembler: &dyn ElementMatrixAssembler<T>,
    ) -> eyre::Result<CsrMatrix<T>> {
        let pattern = self.assemble_pattern(element_assembler.as_connectivity_assembler());
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        self.assemble_into_csr(&mut matrix, element_assembler)?;
        Ok(matrix)
    }

    pub fn assemble_into_csr(
        &self,
        csr: &mut CsrMatrix<T>,
        element_assembler: &dyn ElementMatrixAssembler<T>,
    ) -> eyre::Result<()> {
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
                    let (cols, values) = csr_row.cols_and_values_mut();

                    let a_row = element_matrix.row(local_row_index);
                    add_element_row_to_csr_row(
                        values,
                        cols,
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

        // TODO: Avoid validation?
        SparsityPattern::try_from_offsets_and_indices(
            num_rows,
            num_rows,
            row_offsets,
            column_indices,
        )
        .expect("Pattern data must be valid by definition")
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
            let mut csr_rows = ParallelCsrRowCollection(csr);
            let mut block_adapter = BlockAdapter::with_block_size(&mut csr_rows, sdim);
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
                            let (cols, values) = csr_row.cols_and_values_mut();

                            let a_row = ws.element_matrix.row(local_row_index);
                            add_element_row_to_csr_row(
                                values,
                                cols,
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

pub fn apply_homogeneous_dirichlet_bc_csr<T>(
    matrix: &mut CsrMatrix<T>,
    nodes: &[usize],
    solution_dim: usize,
) where
    T: RealField,
{
    let d = solution_dim;

    // Determine an appropriately scale element to put on the diagonal
    // (Simply setting 1 would ignore the scaling of the entries of the matrix, leading
    // to potentially poor condition numbers)

    // Here we just take the first non-zero diagonal entry as a representative scale.
    // This is cheap and I think reasonably safe option
    let scale = matrix
        .triplet_iter()
        // Only consider diagonal elements
        .filter(|(i, j, _)| i == j)
        .map(|(_, _, v)| v)
        .skip_while(|&x| x == &T::zero())
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
            let (cols, values) = row.cols_and_values_mut();

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
            let (cols, values) = row.cols_and_values_mut();
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

/// Add a row of a local element matrix to the provided row of a CSR matrix.
///
/// `node_connectivity`: The global indices of nodes.
/// `sorted_permutation`: The local indices of nodes in the element, ordered such that the
///    corresponding global indices are sorted.
/// `dim`: The solution dimension.
/// `local_row`: The local row of the element matrix that should be added to the CSR matrix.
fn add_element_row_to_csr_row<T, S>(
    row_values: &mut [T],
    row_col_indices: &[usize],
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

    let (col_indices, values) = (row_col_indices, row_values);
    let mut csr_col_idx_iter = col_indices.iter().copied().enumerate();

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

pub fn color_nodes<C: Connectivity>(connectivity: &[C]) -> Vec<DisjointSubsets> {
    let mut nested = NestedVec::new();

    for conn in connectivity {
        nested.push(conn.vertex_indices());
    }

    sequential_greedy_coloring(&nested)
}

#[derive(Debug)]
struct SerialVectorAssemblerWorkspace<T: Scalar> {
    vector: DVector<T>,
    nodes: Vec<usize>,
}

impl<T: RealField> Default for SerialVectorAssemblerWorkspace<T> {
    fn default() -> Self {
        Self {
            vector: DVector::zeros(0),
            nodes: vec![],
        }
    }
}

#[derive(Debug)]
pub struct SerialVectorAssembler<T: Scalar> {
    workspace: RefCell<SerialVectorAssemblerWorkspace<T>>,
}

impl<T: RealField> Default for SerialVectorAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: RefCell::new(SerialVectorAssemblerWorkspace::default()),
        }
    }
}

impl<T: RealField> SerialVectorAssembler<T> {
    pub fn assemble_vector_into<'a>(
        &self,
        output: impl Into<DVectorSliceMut<'a, T>>,
        element_assembler: &dyn ElementVectorAssembler<T>,
    ) -> eyre::Result<()> {
        // TODO: Move impl into _ method to remove the impl Into<> compilation overhead
        let mut output = output.into();
        let num_elements = element_assembler.num_elements();
        let n = element_assembler.num_nodes();
        let s = element_assembler.solution_dim();
        assert_eq!(output.len(), s * n, "Output dimensions mismatch");

        let mut workspace = self.workspace.borrow_mut();

        for i in 0..num_elements {
            let element_node_count = element_assembler.element_node_count(i);
            workspace.nodes.resize(element_node_count, usize::MAX);
            workspace
                .vector
                .resize_vertically_mut(s * element_node_count, T::zero());
            element_assembler.populate_element_nodes(&mut workspace.nodes, i);
            element_assembler.assemble_element_vector_into(i, (&mut workspace.vector).into())?;
            add_local_to_global(&workspace.vector, &mut output, &workspace.nodes, s);
        }

        Ok(())
    }

    pub fn assemble_vector(
        &self,
        element_assembler: &dyn ElementVectorAssembler<T>,
    ) -> eyre::Result<DVector<T>> {
        let n = element_assembler.num_nodes();
        let mut result = DVector::zeros(element_assembler.solution_dim() * n);
        self.assemble_vector_into(&mut result, element_assembler)?;
        Ok(result)
    }
}

// TODO: Maybe move to some other module?
pub fn gather_global_to_local<'a, T: Scalar>(
    global: impl Into<DVectorSlice<'a, T>>,
    local: impl Into<DVectorSliceMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    gather_global_to_local_(global.into(), local.into(), indices, solution_dim)
}

fn gather_global_to_local_<T: Scalar>(
    global: DVectorSlice<T>,
    mut local: DVectorSliceMut<T>,
    indices: &[usize],
    solution_dim: usize,
) {
    assert_eq!(
        local.len(),
        indices.len() * solution_dim,
        "Size of local vector must be compatible with solutio mdim and index count"
    );
    let s = solution_dim;
    for (i_local, i_global) in indices.iter().enumerate() {
        local
            .index_mut((s * i_local..s * i_local + s, 0))
            .copy_from(&global.index((s * i_global..s * i_global + s, 0)));
    }
}

pub fn add_local_to_global<'a, T: RealField>(
    local: impl Into<DVectorSlice<'a, T>>,
    global: impl Into<DVectorSliceMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    add_local_to_global_(local.into(), global.into(), indices, solution_dim)
}

fn add_local_to_global_<'a, T: RealField>(
    local: DVectorSlice<'a, T>,
    mut global: DVectorSliceMut<'a, T>,
    indices: &[usize],
    solution_dim: usize,
) {
    assert_eq!(
        local.len(),
        indices.len() * solution_dim,
        "Size of local vector must be compatible with solution dim and index count"
    );
    let s = solution_dim;
    for (i_local, i_global) in indices.iter().enumerate() {
        global
            .index_mut((s * i_global..s * i_global + s, 0))
            .add_assign(&local.index((s * i_local..s * i_local + s, 0)));
    }
}

/// A buffer for storing intermediate quadrature data.
///
/// TODO: Move to local assembly???
#[derive(Debug)]
pub struct QuadratureBuffer<T, D, Data>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    quad_weights: Vec<T>,
    quad_points: Vec<Point<T, D>>,
    quad_data: Vec<Data>,
}

impl<T, D, Data> Default for QuadratureBuffer<T, D, Data>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    fn default() -> Self {
        Self {
            quad_weights: Vec::new(),
            quad_points: Vec::new(),
            quad_data: Vec::new(),
        }
    }
}

impl<T, GeometryDim, Data> QuadratureBuffer<T, GeometryDim, Data>
where
    T: RealField,
    GeometryDim: SmallDim,
    Data: Default + Clone,
    DefaultAllocator: SmallDimAllocator<T, GeometryDim>,
{
    /// Resizes the internal buffer storages to the given size.
    pub fn resize(&mut self, quadrature_size: usize) {
        self.quad_points.resize(quadrature_size, Point::origin());
        self.quad_weights.resize(quadrature_size, T::zero());
        self.quad_data.resize(quadrature_size, Data::default());
    }

    /// Populates the buffer by querying a quadrature table with the given element index.
    pub fn populate_element_quadrature_from_table(
        &mut self,
        element_index: usize,
        table: &dyn QuadratureTable<T, GeometryDim, Data = Data>,
    ) {
        let quadrature_size = table.element_quadrature_size(element_index);
        self.resize(quadrature_size);
        table.populate_element_quadrature_and_data(
            element_index,
            &mut self.quad_points,
            &mut self.quad_weights,
            &mut self.quad_data,
        );
    }

    pub fn weights(&self) -> &[T] {
        &self.quad_weights
    }

    pub fn points(&self) -> &[Point<T, GeometryDim>] {
        &self.quad_points
    }

    pub fn data(&self) -> &[Data] {
        &self.quad_data
    }

    /// Calls a closure for each quadrature point currently in the workspace.
    pub fn for_each_quadrature_point<F>(&self, mut f: F) -> eyre::Result<()>
    where
        F: FnMut(T, &Point<T, GeometryDim>, &Data) -> eyre::Result<()>,
    {
        assert_eq!(self.quad_weights.len(), self.quad_points.len());
        assert_eq!(self.quad_weights.len(), self.quad_data.len());
        let iter = izip!(&self.quad_weights, &self.quad_points, &self.quad_data);
        for (&w, xi, data) in iter {
            f(w, xi, data)?;
        }
        Ok(())
    }
}

/// TODO: Move to local assembly?
#[derive(Debug)]
pub struct BasisFunctionBuffer<T: Scalar> {
    element_nodes: Vec<usize>,
    element_basis_values: Vec<T>,
    element_basis_gradients: DMatrix<T>,
}

impl<T: RealField> Default for BasisFunctionBuffer<T> {
    fn default() -> Self {
        Self {
            element_nodes: Vec::new(),
            element_basis_values: Vec::new(),
            element_basis_gradients: DMatrix::zeros(0, 0),
        }
    }
}

impl<T: RealField> BasisFunctionBuffer<T> {
    pub fn resize(&mut self, node_count: usize, reference_dim: usize) {
        self.element_nodes.resize(node_count, usize::MAX);
        self.element_basis_values.resize(node_count, T::zero());
        self.element_basis_gradients
            .resize_mut(reference_dim, node_count, T::zero());
    }

    pub fn populate_element_nodes_from_space<Space>(&mut self, element_index: usize, space: &Space)
    where
        Space: FiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        space.populate_element_nodes(&mut self.element_nodes, element_index);
    }

    /// TODO: Document that populate_element_nodes should be called first
    pub fn populate_element_basis_values_from_space<Space>(
        &mut self,
        element_index: usize,
        space: &Space,
        reference_coords: &Point<T, Space::ReferenceDim>,
    ) where
        Space: FiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        space.populate_element_basis(
            element_index,
            &mut self.element_basis_values,
            reference_coords,
        );
    }

    pub fn populate_element_basis_gradients_from_space<Space>(
        &mut self,
        element_index: usize,
        space: &Space,
        reference_coords: &Point<T, Space::ReferenceDim>,
    ) where
        Space: FiniteElementSpace<T>,
        DefaultAllocator: BiDimAllocator<T, Space::GeometryDim, Space::ReferenceDim>,
    {
        space.populate_element_gradients(
            element_index,
            MatrixSliceMut::from(&mut self.element_basis_gradients),
            reference_coords,
        );
    }

    pub fn element_nodes(&self) -> &[usize] {
        &self.element_nodes
    }

    pub fn element_basis_values(&self) -> &[T] {
        &self.element_basis_values
    }

    pub fn element_basis_values_mut(&mut self) -> &mut [T] {
        &mut self.element_basis_values
    }

    pub fn element_gradients_mut<D: DimName>(&mut self) -> MatrixSliceMut<T, D, Dynamic> {
        MatrixSliceMut::from(&mut self.element_basis_gradients)
    }
}

use crate::assembly::local::{
    ElementConnectivityAssembler, ElementMatrixAssembler, ElementScalarAssembler, ElementVectorAssembler,
};
use crate::space::FiniteElementConnectivity;
use crate::Real;
use fenris_nested_vec::NestedVec;
use fenris_paradis::adapter::BlockAdapter;
use fenris_paradis::coloring::sequential_greedy_coloring;
use fenris_paradis::{DisjointSubsets, ParallelIndexedCollection};
use fenris_sparse::ParallelCsrRowCollection;
use itertools::{enumerate, izip};
use nalgebra::base::storage::Storage;
use nalgebra::{DMatrix, DMatrixViewMut, DVector, DVectorView, DVectorViewMut, DimName, Dyn, Matrix, Scalar, U1};
use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use num::integer::div_ceil;
use parking_lot::Mutex;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::cmp::min;
use std::ops::{AddAssign, IndexMut};
use thread_local::ThreadLocal;

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
    /// Assembles the sparsity pattern associated with the given element assembler.
    ///
    /// The implementation explicitly avoids storing duplicate entries in order to prevent
    /// excessive memory costs.
    pub fn assemble_pattern(&self, element_assembler: &impl ElementConnectivityAssembler) -> SparsityPattern {
        let sdim = element_assembler.solution_dim();
        let num_nodes = element_assembler.num_nodes();
        let num_rows = sdim * num_nodes;
        let mut node_sets: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); num_nodes];
        let mut element_global_nodes = Vec::new();
        for i in 0..element_assembler.num_elements() {
            let element_node_count = element_assembler.element_node_count(i);
            element_global_nodes.resize(element_node_count, usize::MAX);
            element_assembler.populate_element_nodes(&mut element_global_nodes, i);

            for &node_i in &element_global_nodes {
                for &node_j in &element_global_nodes {
                    node_sets[node_i].insert(node_j);
                }
            }
        }

        let mut offsets = Vec::with_capacity(num_rows);
        offsets.push(0);
        let mut current_offset = 0;
        for node_set in &node_sets {
            for _ in 0..sdim {
                let count = sdim * node_set.len();
                offsets.push(current_offset + count);
                current_offset += count;
            }
        }
        assert_eq!(offsets.len(), num_rows + 1);

        let mut col_indices = Vec::with_capacity(*offsets.last().unwrap());
        let mut node_buffer: Vec<usize> = Vec::new();
        for node_set in &node_sets {
            node_buffer.clear();
            node_buffer.extend(node_set);
            node_buffer.sort_unstable();
            // We have sdim identical rows (in terms of pattern)
            for _ in 0..sdim {
                for node_j in &node_buffer {
                    for j in 0..sdim {
                        let col_idx = sdim * node_j + j;
                        col_indices.push(col_idx);
                    }
                }
            }
        }

        assert_eq!(*offsets.last().unwrap(), col_indices.len());

        debug_assert!(
            SparsityPattern::try_from_offsets_and_indices(num_rows, num_rows, offsets.clone(), col_indices.clone())
                .is_ok(),
            "Internal error: constructed sparsity pattern is not valid. This is a bug!"
        );
        unsafe { SparsityPattern::from_offset_and_indices_unchecked(num_rows, num_rows, offsets, col_indices) }
    }
}

impl<T: Real> CsrAssembler<T> {
    pub fn assemble(&self, element_assembler: &impl ElementMatrixAssembler<T>) -> eyre::Result<CsrMatrix<T>> {
        let pattern = self.assemble_pattern(element_assembler);
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        self.assemble_into_csr(&mut matrix, element_assembler)?;
        Ok(matrix)
    }

    pub fn assemble_into_csr(
        &self,
        csr: &mut CsrMatrix<T>,
        element_assembler: &impl ElementMatrixAssembler<T>,
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

            let matrix_slice = DMatrixViewMut::from(&mut *element_matrix);
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
    /// Assembles the sparsity pattern associated with the given element assembler.
    ///
    /// The implementation explicitly avoids storing duplicate entries in order to prevent
    /// excessive memory costs.
    pub fn assemble_pattern(&self, element_assembler: &(impl Sync + ElementConnectivityAssembler)) -> SparsityPattern {
        let sdim = element_assembler.solution_dim();
        let num_nodes = element_assembler.num_nodes();
        let num_elements = element_assembler.num_elements();
        let num_rows = sdim * num_nodes;
        // We store a HashSet (with a fast hash) for each node,
        // eventually containing the set of (unique) neighbors for that node
        let mut node_sets: Vec<Mutex<FxHashSet<usize>>> = (0..num_nodes)
            .map(|_| Mutex::new(FxHashSet::default()))
            .collect();
        let node_buffer: ThreadLocal<RefCell<Vec<usize>>> = ThreadLocal::new();

        // Batch computation in order to make each Rayon unit of work larger
        let batch_size = 10;
        let num_batches = div_ceil(num_elements, batch_size);
        (0..num_batches).into_par_iter().for_each(|batch_index| {
            let batch_start = batch_size * batch_index;
            let batch_end = min(num_elements, batch_start + batch_size);
            assert!(batch_end >= batch_start);
            let mut node_buffer = node_buffer.get_or_default().borrow_mut();
            for i in batch_start..batch_end {
                let element_node_count = element_assembler.element_node_count(i);
                node_buffer.resize(element_node_count, usize::MAX);
                element_assembler.populate_element_nodes(&mut node_buffer, i);

                for &node_i in &*node_buffer {
                    let mut node_set = node_sets[node_i].lock();
                    for &node_j in &*node_buffer {
                        node_set.insert(node_j);
                    }
                }
            }
        });

        // TODO: Parallelize offset computation
        // (only takes up relatively small proportion of time though, not worth spending much effort atm)
        let mut offsets = Vec::with_capacity(num_rows);
        offsets.push(0);
        let mut current_offset = 0;
        for node_set in &node_sets {
            for _ in 0..sdim {
                let count = sdim * node_set.lock().len();
                offsets.push(current_offset + count);
                current_offset += count;
            }
        }

        let nnz = current_offset;
        assert_eq!(offsets.len(), num_rows + 1);

        let mut col_indices = vec![0; nnz];
        let col_indices_access = unsafe { col_indices.create_access() };

        // Note: We use the same batch size, but before we were batching over *elements*,
        // now we're batching over *rows*
        node_sets
            .par_chunks_mut(batch_size)
            .enumerate()
            .for_each(|(batch_index, locked_node_sets)| {
                let batch_start = batch_size * batch_index;
                let batch_end = min(num_nodes, batch_start + batch_size);
                assert!(batch_end >= batch_start);
                let mut node_buffer = node_buffer.get_or_default().borrow_mut();

                for (i, locked_node_set) in izip!(batch_start..batch_end, locked_node_sets) {
                    let node_set = locked_node_set.get_mut();
                    node_buffer.clear();
                    node_buffer.extend(node_set.iter());
                    node_buffer.sort_unstable();

                    for s_i in 0..sdim {
                        let begin = offsets[sdim * i + s_i];
                        let end = offsets[sdim * i + s_i + 1];
                        let subslice = unsafe { col_indices_access.subslice_mut(begin..end) };
                        for (i, node_j) in enumerate(node_buffer.iter()) {
                            let block = subslice.index_mut(sdim * i..(sdim * (i + 1)));
                            for (j, col_idx) in enumerate(block) {
                                *col_idx = sdim * node_j + j;
                            }
                        }
                    }
                }
            });

        assert_eq!(*offsets.last().unwrap(), col_indices.len());
        debug_assert!(
            SparsityPattern::try_from_offsets_and_indices(num_rows, num_rows, offsets.clone(), col_indices.clone())
                .is_ok(),
            "Internal error: constructed sparsity pattern is not valid. This is a bug!"
        );
        unsafe { SparsityPattern::from_offset_and_indices_unchecked(num_rows, num_rows, offsets, col_indices) }
    }
}

impl<T: Real + Send> CsrParAssembler<T> {
    pub fn assemble(
        &self,
        colors: &[DisjointSubsets],
        element_assembler: &(impl ElementMatrixAssembler<T> + Sync),
    ) -> eyre::Result<CsrMatrix<T>> {
        let pattern = self.assemble_pattern(element_assembler);
        let initial_matrix_values = vec![T::zero(); pattern.nnz()];
        let mut matrix = CsrMatrix::try_from_pattern_and_values(pattern, initial_matrix_values)
            .expect("CSR data must be valid by definition");
        self.assemble_into_csr(&mut matrix, colors, element_assembler)?;
        Ok(matrix)
    }

    pub fn assemble_into_csr(
        &self,
        csr: &mut CsrMatrix<T>,
        colors: &[DisjointSubsets],
        element_assembler: &(dyn Sync + ElementMatrixAssembler<T>),
    ) -> eyre::Result<()> {
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

                    let matrix_slice = DMatrixViewMut::from(&mut ws.element_matrix);
                    element_assembler.assemble_element_matrix_into(element_index, matrix_slice)?;
                    element_assembler.populate_element_nodes(&mut ws.element_global_nodes, element_index);
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
                .collect::<eyre::Result<()>>()?;
        }

        Ok(())
    }
}

pub fn apply_homogeneous_dirichlet_bc_csr<T>(matrix: &mut CsrMatrix<T>, nodes: &[usize], solution_dim: usize)
where
    T: Real,
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

pub fn apply_homogeneous_dirichlet_bc_matrix<T, SolutionDim>(matrix: &mut DMatrix<T>, nodes: &[usize])
where
    T: Real,
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
    rhs: impl Into<DVectorViewMut<'a, T>>,
    nodes: &[usize],
    solution_dim: usize,
) where
    T: Real,
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
    local_row: &Matrix<T, U1, Dyn, S>,
) where
    T: Real,
    S: Storage<T, U1, Dyn>,
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

/// Computes a coloring for the nodes of the given element connectivity.
pub fn color_nodes<C: FiniteElementConnectivity + ?Sized>(connectivity: &C) -> Vec<DisjointSubsets> {
    let mut nested = NestedVec::new();

    let mut node_buffer = Vec::new();
    for element_index in 0..connectivity.num_elements() {
        node_buffer.resize(connectivity.element_node_count(element_index), 0);
        connectivity.populate_element_nodes(&mut node_buffer, element_index);
        nested.push(&node_buffer);
    }

    sequential_greedy_coloring(&nested)
}

#[derive(Debug)]
struct VectorAssemblerWorkspace<T: Scalar> {
    vector: DVector<T>,
    nodes: Vec<usize>,
}

impl<T: Real> Default for VectorAssemblerWorkspace<T> {
    fn default() -> Self {
        Self {
            vector: DVector::zeros(0),
            nodes: vec![],
        }
    }
}

#[derive(Debug)]
pub struct VectorAssembler<T: Scalar> {
    workspace: RefCell<VectorAssemblerWorkspace<T>>,
}

impl<T: Real> Default for VectorAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: RefCell::new(VectorAssemblerWorkspace::default()),
        }
    }
}

impl<T: Real> VectorAssembler<T> {
    pub fn assemble_vector_into<'a>(
        &self,
        output: impl Into<DVectorViewMut<'a, T>>,
        element_assembler: &impl ElementVectorAssembler<T>,
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

    pub fn assemble_vector(&self, element_assembler: &impl ElementVectorAssembler<T>) -> eyre::Result<DVector<T>> {
        let n = element_assembler.num_nodes();
        let mut result = DVector::zeros(element_assembler.solution_dim() * n);
        self.assemble_vector_into(&mut result, element_assembler)?;
        Ok(result)
    }
}

#[derive(Debug)]
pub struct VectorParAssembler<T: Scalar + Send> {
    workspace: ThreadLocal<RefCell<VectorAssemblerWorkspace<T>>>,
}

impl<T: Real> Default for VectorParAssembler<T> {
    fn default() -> Self {
        Self {
            workspace: Default::default(),
        }
    }
}

impl<T: Real> VectorParAssembler<T> {
    pub fn assemble_vector(
        &self,
        colors: &[DisjointSubsets],
        element_assembler: &(impl ElementVectorAssembler<T> + Sync),
    ) -> eyre::Result<DVector<T>> {
        let n = element_assembler.num_nodes();
        let mut result = DVector::zeros(element_assembler.solution_dim() * n);
        self.assemble_vector_into(&mut result, colors, element_assembler)?;
        Ok(result)
    }

    pub fn assemble_vector_into<'a>(
        &self,
        output: impl Into<DVectorViewMut<'a, T>>,
        colors: &[DisjointSubsets],
        element_assembler: &(impl ElementVectorAssembler<T> + ?Sized + Sync),
    ) -> eyre::Result<()> {
        let mut output = output.into();
        let n = element_assembler.num_nodes();
        let s = element_assembler.solution_dim();
        assert_eq!(output.len(), s * n, "Output dimensions mismatch");

        for color in colors {
            let mut block_adapter = BlockAdapter::with_block_size(output.as_mut_slice(), s);

            color
                .subsets_par_iter(&mut block_adapter)
                .map(|mut subset| {
                    let ws = &mut *self.workspace.get_or_default().borrow_mut();

                    let element_index = subset.label();
                    let element_node_count = element_assembler.element_node_count(element_index);

                    ws.nodes.resize(element_node_count, usize::MAX);
                    ws.vector
                        .resize_vertically_mut(s * element_node_count, T::zero());
                    element_assembler.populate_element_nodes(&mut ws.nodes, element_index);
                    element_assembler.assemble_element_vector_into(element_index, (&mut ws.vector).into())?;

                    for local_node_idx in 0..element_node_count {
                        let mut block = subset.get_mut(local_node_idx);
                        let v_rows = ws.vector.rows(s * local_node_idx, s);
                        for i in 0..s {
                            *block.index_mut(i) += v_rows[i];
                        }
                    }

                    Ok(())
                })
                .collect::<eyre::Result<()>>()?;
        }

        Ok(())
    }
}

#[deprecated = "Use assemble_scalar instead"]
pub fn compute_global_potential<T>(element_assembler: &(impl ElementScalarAssembler<T> + ?Sized)) -> eyre::Result<T>
where
    T: Real,
{
    assemble_scalar(element_assembler)
}

/// Computes the value of a global scalar potential as a sum of element-wise scalars.
pub fn assemble_scalar<T>(element_assembler: &(impl ElementScalarAssembler<T> + ?Sized)) -> eyre::Result<T>
where
    T: Real,
{
    let num_elements = element_assembler.num_elements();
    let mut global_potential = T::zero();
    for i in 0..num_elements {
        let element_contrib = element_assembler
            .assemble_element_scalar(i)
            .map_err(|error| error.wrap_err(format!("Assembling scalar failed for element {}", i)))?;
        global_potential += element_contrib;
    }
    Ok(global_potential)
}

/// Computes the value of a global scalar potential as a sum of element-wise scalars in parallel.
#[deprecated = "Use par_assemble_scalar instead"]
pub fn par_compute_global_potential<T>(
    element_assembler: &(impl ElementScalarAssembler<T> + ?Sized + Sync),
) -> eyre::Result<T>
where
    T: Real,
{
    par_assemble_scalar(element_assembler)
}

/// Computes the value of a global scalar potential as a sum of element-wise scalars in parallel.
pub fn par_assemble_scalar<T>(element_assembler: &(impl ElementScalarAssembler<T> + ?Sized + Sync)) -> eyre::Result<T>
where
    T: Real,
{
    let num_elements = element_assembler.num_elements();
    let global_potential = (0..num_elements)
        .into_par_iter()
        .map(|i| {
            element_assembler
                .assemble_element_scalar(i)
                .map_err(|error| error.wrap_err(format!("Assembling scalar failed for element {}", i)))
        })
        .try_reduce(|| T::zero(), |a, b| Ok(a + b));

    global_potential
}

// TODO: Maybe move to some other module?
pub fn gather_global_to_local<'a, T: Scalar>(
    global: impl Into<DVectorView<'a, T>>,
    local: impl Into<DVectorViewMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    gather_global_to_local_(global.into(), local.into(), indices, solution_dim)
}

fn gather_global_to_local_<T: Scalar>(
    global: DVectorView<T>,
    mut local: DVectorViewMut<T>,
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

pub fn add_local_to_global<'a, T: Real>(
    local: impl Into<DVectorView<'a, T>>,
    global: impl Into<DVectorViewMut<'a, T>>,
    indices: &[usize],
    solution_dim: usize,
) {
    add_local_to_global_(local.into(), global.into(), indices, solution_dim)
}

fn add_local_to_global_<'a, T: Real>(
    local: DVectorView<'a, T>,
    mut global: DVectorViewMut<'a, T>,
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

#[deprecated = "Use fenris::assembly::buffers::QuadratureBuffer instead"]
pub type QuadratureBuffer<T, D, Data = ()> = crate::assembly::buffers::QuadratureBuffer<T, D, Data>;

#[deprecated = "Use fenris::assembly::buffers::QuadratureBuffer instead"]
pub type BasisFunctionBuffer<T> = crate::assembly::buffers::BasisFunctionBuffer<T>;

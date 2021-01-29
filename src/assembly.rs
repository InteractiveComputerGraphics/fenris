use crate::element::{ElementConnectivity, VolumetricFiniteElement, ReferenceFiniteElement, MatrixSlice, MatrixSliceMut};
use crate::quadrature::Quadrature;
use nalgebra::dimension::{DimNameMul, DimNameProd};
use nalgebra::{
    ComplexField, DMatrix, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, Dim, DimMin,
    DimName, Matrix, MatrixMN, MatrixN, Point, RealField, Scalar, SquareMatrix, VectorN, U1,
};
use nalgebra::{DMatrixSliceMut, Dynamic, MatrixSliceMN};
use std::ops::AddAssign;

use nalgebra::SymmetricEigen;

use crate::allocators::{
    ElementConnectivityAllocator, FiniteElementMatrixAllocator, VolumeFiniteElementAllocator,
};
use crate::connectivity::Connectivity;
use crate::mesh::Mesh;
use crate::util::{coerce_col_major_slice, coerce_col_major_slice_mut};
use alga::general::{ClosedAdd, ClosedMul};
use nalgebra::allocator::Allocator;
use nalgebra::storage::Storage;
use nested_vec::NestedVec;
use num::{One, Zero};
use paradis::adapter::BlockAdapter;
use paradis::coloring::sequential_greedy_coloring;
use paradis::DisjointSubsets;
use rayon::prelude::*;
use sparse::{CooMatrix, CsrMatrix};
use sparse::{CsrRowMut, SparsityPattern};
use std::cell::RefCell;
use std::error::Error;
use std::marker::PhantomData;
use thread_local::ThreadLocal;

pub trait ElementConnectivityAssembler {
    fn solution_dim(&self) -> usize;

    fn num_elements(&self) -> usize;

    fn num_nodes(&self) -> usize;

    fn element_node_count(&self, element_index: usize) -> usize;

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize);
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

pub trait ElementAssembler<T: Scalar>: ElementConnectivityAssembler {
    fn assemble_element_matrix_into(
        &self,
        output: DMatrixSliceMut<T>,
        element_index: usize,
    ) -> Result<(), Box<dyn Send + Error>>;
}

/// An assembler for CSR matrices.
#[derive(Debug, Clone)]
pub struct CsrAssembler<T: Scalar> {
    // All members are buffers that help prevent unnecessary allocations
    // when assembling multiple matrices with the same assembler
    connectivity_permutation: Vec<usize>,
    element_global_nodes: Vec<usize>,
    element_matrix: DMatrix<T>,
}

impl<T: RealField> Default for CsrAssembler<T> {
    fn default() -> Self {
        Self {
            connectivity_permutation: Vec::new(),
            element_global_nodes: Vec::new(),
            element_matrix: DMatrix::zeros(0, 0),
        }
    }
}

impl<T: RealField> CsrAssembler<T> {
    pub fn assemble_into_csr(
        &mut self,
        csr: &mut CsrMatrix<T>,
        element_assembler: &dyn ElementAssembler<T>,
    ) -> Result<(), Box<dyn Send + Error>> {
        // Reuse previously allocated buffers
        let connectivity_permutation = &mut self.connectivity_permutation;
        let element_global_nodes = &mut self.element_global_nodes;
        let element_matrix = &mut self.element_matrix;

        let sdim = element_assembler.solution_dim();

        for i in 0..element_assembler.num_elements() {
            let element_node_count = element_assembler.element_node_count(i);
            let element_matrix_dim = sdim * element_node_count;

            element_global_nodes.resize(element_node_count, 0);
            element_matrix.resize_mut(element_matrix_dim, element_matrix_dim, T::zero());
            element_matrix.fill(T::zero());

            let matrix_slice = DMatrixSliceMut::from(&mut *element_matrix);
            element_assembler.assemble_element_matrix_into(matrix_slice, i)?;
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
    // All members are buffers that help prevent unnecessary allocations
    // when assembling multiple matrices with the same assembler
    connectivity_permutation: ThreadLocal<RefCell<Vec<usize>>>,
    element_global_nodes: ThreadLocal<RefCell<Vec<usize>>>,
    element_matrix: ThreadLocal<RefCell<DMatrix<T>>>,
}

impl<T: Scalar + Send> Default for CsrParAssembler<T> {
    fn default() -> Self {
        Self {
            connectivity_permutation: ThreadLocal::new(),
            element_global_nodes: ThreadLocal::new(),
            element_matrix: ThreadLocal::new(),
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
        &mut self,
        csr: &mut CsrMatrix<T>,
        colors: &[DisjointSubsets],
        element_assembler: &(dyn Sync + ElementAssembler<T>),
    ) -> Result<(), Box<dyn Send + Error>> {
        let sdim = element_assembler.solution_dim();

        for color in colors {
            let mut block_adapter = BlockAdapter::with_block_size(csr, sdim);
            color
                .subsets_par_iter(&mut block_adapter)
                .map(|mut subset| {
                    let mut connectivity_permutation =
                        self.connectivity_permutation.get_or_default().borrow_mut();
                    let mut element_global_nodes =
                        self.element_global_nodes.get_or_default().borrow_mut();
                    let mut element_matrix = self
                        .element_matrix
                        .get_or(|| RefCell::new(DMatrix::zeros(0, 0)))
                        .borrow_mut();

                    let element_index = subset.label();
                    let element_node_count = element_assembler.element_node_count(element_index);
                    let element_matrix_dim = sdim * element_node_count;

                    element_global_nodes.resize(element_node_count, 0);
                    element_matrix.resize_mut(element_matrix_dim, element_matrix_dim, T::zero());
                    element_matrix.fill(T::zero());

                    let matrix_slice = DMatrixSliceMut::from(&mut *element_matrix);
                    element_assembler.assemble_element_matrix_into(matrix_slice, element_index)?;
                    element_assembler
                        .populate_element_nodes(&mut element_global_nodes, element_index);
                    debug_assert_eq!(subset.global_indices(), element_global_nodes.as_slice());

                    connectivity_permutation.clear();
                    connectivity_permutation.extend(0..element_node_count);
                    connectivity_permutation.sort_unstable_by_key(|i| element_global_nodes[*i]);

                    for local_node_idx in 0..element_node_count {
                        let mut csr_block_row = subset.get_mut(local_node_idx);
                        for i in 0..sdim {
                            let local_row_index = sdim * local_node_idx + i;
                            let mut csr_row = csr_block_row.get_mut(i).unwrap();

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

                    Ok(())
                })
                .collect::<Result<(), Box<dyn Send + Error>>>()?;
        }

        Ok(())
    }
}

struct GeneralizedStiffnessElementAssembler<'a, T, SolutionDim, C, Contraction, Q, Transformation>
where
    T: Scalar,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    Transformation: ?Sized,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
    vertices: &'a [Point<T, C::GeometryDim>],
    connectivity: &'a [C],
    contraction: &'a Contraction,
    u: &'a DVector<T>,
    quadrature_table: &'a Q,
    transformation: &'a Transformation,
    solution_dim_marker: PhantomData<SolutionDim>,
}

impl<'a, T, SolutionDim, C, Contraction, Q, Transformation> ElementConnectivityAssembler
    for GeneralizedStiffnessElementAssembler<'a, T, SolutionDim, C, Contraction, Q, Transformation>
where
    T: Scalar,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    SolutionDim: DimName,
    Transformation: ?Sized,
    DefaultAllocator: ElementConnectivityAllocator<T, C>,
{
    fn num_nodes(&self) -> usize {
        self.vertices.len()
    }

    fn solution_dim(&self) -> usize {
        SolutionDim::dim()
    }

    fn num_elements(&self) -> usize {
        self.connectivity.len()
    }

    fn element_node_count(&self, element_index: usize) -> usize {
        self.connectivity[element_index].vertex_indices().len()
    }

    fn populate_element_nodes(&self, output: &mut [usize], element_index: usize) {
        output.copy_from_slice(self.connectivity[element_index].vertex_indices());
    }
}

impl<'a, T, SolutionDim, C, Contraction, Q, Transformation> ElementAssembler<T>
    for GeneralizedStiffnessElementAssembler<'a, T, SolutionDim, C, Contraction, Q, Transformation>
where
    T: RealField,
    C: ElementConnectivity<T, ReferenceDim = <C as ElementConnectivity<T>>::GeometryDim>,
    C::GeometryDim: DimMin<C::GeometryDim, Output = C::GeometryDim>,
    SolutionDim: DimName,
    Contraction: GeneralizedEllipticContraction<T, SolutionDim, C::GeometryDim>,
    Q: QuadratureTable<T, C::GeometryDim>,
    Transformation: ?Sized + ElementMatrixTransformation<T>,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, C::GeometryDim>,
{
    fn assemble_element_matrix_into(
        &self,
        mut output: DMatrixSliceMut<T>,
        element_index: usize,
    ) -> Result<(), Box<dyn Send + Error>> {
        let connectivity = &self.connectivity[element_index];
        let element = connectivity.element(self.vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );

        // TODO: Don't allocate!
        let mut u_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
        connectivity.populate_element_variables(MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element), self.u);
        assemble_generalized_element_stiffness(
            DMatrixSliceMut::from(&mut output),
            &element,
            self.contraction,
            MatrixSlice::from(&u_element),
            &self.quadrature_table.quadrature_for_element(element_index),
        );

        self.transformation.transform_element_matrix(&mut output);

        Ok(())
    }
}

pub trait GeneralizedEllipticOperator<T, SolutionDim, GeometryDim>
where
    T: Scalar,
    SolutionDim: DimName,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, SolutionDim>,
{
    fn compute_elliptic_term(
        &self,
        gradient: &MatrixMN<T, GeometryDim, SolutionDim>,
    ) -> MatrixMN<T, GeometryDim, SolutionDim>;
}

pub trait GeneralizedEllipticContraction<T, SolutionDim, GeometryDim>
where
    T: Scalar + Zero + One + ClosedAdd + ClosedMul,
    SolutionDim: DimName,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, SolutionDim>
        + Allocator<T, GeometryDim, GeometryDim>
        + Allocator<T, SolutionDim, SolutionDim>
        + Allocator<T, GeometryDim>
        + Allocator<T, U1, GeometryDim>,
{
    fn contract(
        &self,
        gradient: &MatrixMN<T, GeometryDim, SolutionDim>,
        a: &VectorN<T, GeometryDim>,
        b: &VectorN<T, GeometryDim>,
    ) -> MatrixMN<T, SolutionDim, SolutionDim>;

    /// Compute multiple contractions and store the result in the provided matrix.
    ///
    /// The matrix `a` is a `GeometryDim x NodalDim` sized matrix, in which each column
    /// corresponds to a vector of dimension `GeometryDim`. The output matrix is a square matrix
    /// with row and col dimensions `SolutionDim * NodalDim`, consisting of `NodalDim x NodalDim`
    /// block matrices, each with dimension `SolutionDim x SolutionDim`.
    ///
    /// Let c(gradient, a, b) denote the contraction of vectors a and b.
    /// Then the result of c(gradient, a_I, a_J) for each I, J in the range `(0 .. NodalDim)`
    /// must be *added* to `output_IJ`, where `output_IJ` is the `SolutionDim x SolutionDim`
    /// block matrix corresponding to nodes `I` and `J`.
    ///
    /// TODO: Consider using a unit-stride matrix slice for performance reasons.
    fn contract_multiple_into(
        &self,
        output: &mut DMatrixSliceMut<T>,
        gradient: &MatrixMN<T, GeometryDim, SolutionDim>,
        a: &MatrixSliceMN<T, GeometryDim, Dynamic>,
    ) {
        let num_nodes = a.ncols();
        let output_dim = num_nodes * SolutionDim::dim();
        assert_eq!(output_dim, output.nrows());
        assert_eq!(output_dim, output.ncols());

        let sdim = SolutionDim::dim();
        for i in 0..num_nodes {
            for j in i..num_nodes {
                let a_i = a.fixed_slice::<GeometryDim, U1>(0, i).clone_owned();
                let a_j = a.fixed_slice::<GeometryDim, U1>(0, j).clone_owned();
                let contraction = self.contract(gradient, &a_i, &a_j);
                output
                    .fixed_slice_mut::<SolutionDim, SolutionDim>(i * sdim, j * sdim)
                    .add_assign(&contraction);

                // TODO: We currently assume symmetry. Should maybe have a method that
                // says whether it is symmetric or not?
                if i != j {
                    output
                        .fixed_slice_mut::<SolutionDim, SolutionDim>(j * sdim, i * sdim)
                        .add_assign(&contraction.transpose());
                }
            }
        }
    }
}

/// Lookup table mapping elements to quadrature rules.
pub trait QuadratureTable<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
{
    type QuadratureRule: Quadrature<T, GeometryDim>;

    fn quadrature_for_element(&self, element_index: usize) -> Self::QuadratureRule;
}

impl<'a, T, GeometryDim, F, Q> QuadratureTable<T, GeometryDim> for F
where
    F: 'a + Fn(usize) -> Q,
    Q: 'a + Quadrature<T, GeometryDim>,
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
{
    type QuadratureRule = Q;

    fn quadrature_for_element(&self, element_index: usize) -> Self::QuadratureRule {
        self(element_index)
    }
}

/// Convenience wrapper to turn a single quadrature into a quadrature table.
///
/// More precisely, this implies that the same quadrature rule will be used for every
/// element.
///
/// Note that the given quadrature will be cloned, so it's often more useful to wrap
/// a reference to a quadrature than letting the quadrature be cloned for each element.
#[derive(Copy, Clone, Debug)]
pub struct UniformQuadratureTable<Q>(pub Q);

impl<T, GeometryDim, Q> QuadratureTable<T, GeometryDim> for UniformQuadratureTable<Q>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim, U1>,
    Q: Clone + Quadrature<T, GeometryDim>,
{
    type QuadratureRule = Q;

    fn quadrature_for_element(&self, _element_index: usize) -> Self::QuadratureRule {
        self.0.clone()
    }
}

/// A transformation for element matrices.
///
/// This is most often used to adapt the spectrum of an element matrix so that it
/// becomes semi-definite.
pub trait ElementMatrixTransformation<T: Scalar> {
    fn transform_element_matrix(&self, element_matrix: &mut DMatrixSliceMut<T>);
}

/// Leaves the given element matrix unaltered.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NoTransformation;

impl<T: Scalar> ElementMatrixTransformation<T> for NoTransformation {
    fn transform_element_matrix(&self, _element_matrix: &mut DMatrixSliceMut<T>) {
        // Do nothing
    }
}

/// Projects the given matrix onto semidefiniteness by using `nalgebra`'s symmetric
/// eigendecomposition.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DefaultSemidefiniteProjection;

impl<T: RealField> ElementMatrixTransformation<T> for DefaultSemidefiniteProjection {
    fn transform_element_matrix(&self, element_matrix: &mut DMatrixSliceMut<T>) {
        let mut eigendecomp = SymmetricEigen::new(element_matrix.clone_owned());
        for eigenval in &mut eigendecomp.eigenvalues {
            *eigenval = T::max(T::zero(), *eigenval);
        }
        // TODO: Don't recompose if we didn't change anything
        let recomposed = eigendecomp.recompose();
        element_matrix.copy_from(&recomposed);
    }
}

/// Computes
fn mat_mul_mat_transpose<'a, T, R, C>(
    a: impl Into<MatrixSlice<'a, T, R, Dynamic>>,
    b: impl Into<MatrixSlice<'a, T, C, Dynamic>>)
    -> MatrixMN<T, R, C>
where
    T: RealField,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>
{
    mat_mul_mat_transpose_(a.into(), b.into())
}

/// Computes the matrix product `A * B^T` where `A` and `B` have fixed row counts, but
/// dynamic column counts.
fn mat_mul_mat_transpose_<'a, T, R, C>(
    a: MatrixSlice<'a, T, R, Dynamic>,
    b: MatrixSlice<'a, T, C, Dynamic>)
    -> MatrixMN<T, R, C>
where
    T: RealField,
    R: DimName,
    C: DimName,
    DefaultAllocator: Allocator<T, R, C>
{
    assert_eq!(a.nrows(), b.nrows());
    let mut result = MatrixMN::<T, R, C>::zeros();

    // Compute A B^T = sum_k a_k * b_k^T   (outer product)
    // where a_k and b_k represent column k in A and B, respectively
    for (a_k, b_k) in a.column_iter().zip(b.column_iter()) {
        result.ger(T::one(), &a_k, &b_k, T::one());
    }
    result
}

/// Computes the integral of a scalar function f(X, u, grad u) over an element.
#[allow(non_snake_case)]
pub fn compute_element_integral<T, SolutionDim, Element, F>(
    element: &Element,
    u_element: &MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: &impl Quadrature<T, Element::GeometryDim>,
    function: F,
) -> T
where
    T: RealField,
    Element: VolumetricFiniteElement<T>,
    Element::GeometryDim: DimName + DimMin<Element::GeometryDim, Output = Element::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: VolumeFiniteElementAllocator<T, Element::GeometryDim>
        + Allocator<T, Element::GeometryDim, SolutionDim>
        + Allocator<T, SolutionDim, Element::GeometryDim>
        + Allocator<T, SolutionDim, U1>,
    F: Fn(
        &VectorN<T, Element::GeometryDim>,
        &VectorN<T, SolutionDim>,
        &MatrixMN<T, Element::GeometryDim, SolutionDim>,
    ) -> T,
{
    let mut f_e = T::zero();

    let weights = quadrature.weights();
    let points = quadrature.points();

    // TODO: Avoid allocation!
    let mut basis_values = MatrixMN::<_, U1, Dynamic>::zeros(element.num_nodes());
    let mut gradients = MatrixMN::<_, Element::GeometryDim, Dynamic>::zeros(element.num_nodes());

    for (&w, xi) in weights.iter().zip(points) {
        element.populate_basis(MatrixSliceMut::from(&mut basis_values), xi);
        element.populate_basis_gradients(MatrixSliceMut::from(&mut gradients), xi);

        // Jacobian
        let J = element.reference_jacobian(xi);

        let J_det = J.determinant();
        let J_inv = J.try_inverse().expect("Jacobian must be invertible");

        let X = element.map_reference_coords(xi);
        let u = mat_mul_mat_transpose(u_element, &basis_values);
        let u_grad = compute_volume_u_grad(&J_inv.transpose(),
                                           MatrixSlice::from(&gradients),
                                           MatrixSlice::from(u_element));
        let f = function(&X, &u, &u_grad);
        f_e += f * w * J_det.abs();
    }

    f_e
}

/// Assemble the generalized element matrix for the given element.
///
/// TODO: Allow non-uniform density. Possible API: Take callback that gets both index of quadrature
/// point and position of quadrature point...? That way one can i.e. associate density with
/// a particular quadrature point (or return an element-wise constant) or use an analytic function.
#[allow(non_snake_case)]
pub fn assemble_generalized_element_mass<T, SolutionDim, Element, Q>(
    mut element_matrix: DMatrixSliceMut<T>,
    element: &Element,
    density: T,
    quadrature: &Q,
)
where
    T: RealField,
    Element: VolumetricFiniteElement<T>,
    Element::GeometryDim: DimMin<Element::GeometryDim, Output = Element::GeometryDim>,
    SolutionDim: DimName,
    Q: Quadrature<T, Element::GeometryDim>,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Element::GeometryDim>,
{
    // TODO: Avoid allocation!!!
    let mut m_element_scalar = DMatrix::zeros(element.num_nodes(), element.num_nodes());
    let mut basis_values = DMatrix::zeros(1, element.num_nodes());

    let weights = quadrature.weights();
    let points = quadrature.points();

    let num_nodes = element.num_nodes();
    let sol_dim = SolutionDim::dim();

    for (&w, xi) in weights.iter().zip(points) {
        let J = element.reference_jacobian(xi);
        let J_det = J.determinant();

        element.populate_basis(MatrixSliceMut::from(&mut basis_values), xi);
        let phi = MatrixSlice::<_, U1, Dynamic>::from(&basis_values);

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                // Product of shape functions
                m_element_scalar[(i, j)] += density * J_det.abs() * w * phi[i] * phi[j];
            }
        }
    }

    let skip_shape = (sol_dim.saturating_sub(1), sol_dim.saturating_sub(1));
    for i in 0..sol_dim {
        element_matrix
            .slice_with_steps_mut((i, i), (num_nodes, num_nodes), skip_shape)
            .copy_from(&m_element_scalar);
    }
}

#[allow(non_snake_case)]
fn compute_volume_u_grad<T, GeometryDim, SolutionDim>(
    jacobian_inv_t: &MatrixMN<T, GeometryDim, GeometryDim>,
    phi_grad_ref: MatrixSlice<T, GeometryDim, Dynamic>,
    u: MatrixSlice<T, SolutionDim, Dynamic>)
-> MatrixMN<T, GeometryDim, SolutionDim>
where
    T: RealField,
    SolutionDim: DimName,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>
        + Allocator<T, GeometryDim, SolutionDim>
        + Allocator<T, SolutionDim, GeometryDim>
        + Allocator<T, GeometryDim, GeometryDim>
{
    // We have that grad u = sum_I grad phi_I u_I^T,
    // which can alternatively be written
    //  grad u = G U^T,
    // where G = [ grad phi_1, grad phi_2, ... ] and U = [ u_1, u_2, ... ]
    // are matrices containing the (physical domain) gradients and per-node solution variables u_I.
    // We note that `grad phi_I = inv(J)^T * grad phi_ref_I`, where `grad phi_ref_I` is the
    // gradient of basis function `phi_I` with respect to the reference coordinates.
    // Unfortunately, we cannot easily directly compute G U^T since there is only
    // A^T B through gemm_tr available in `nalgebra`. Therefore we instead compute
    // the sum of outer products
    let G_ref = phi_grad_ref;
    let mut u_grad = MatrixMN::<_, GeometryDim, SolutionDim>::zeros();
    for (phi_I_grad_ref, u_I) in G_ref.column_iter().zip(u.column_iter()) {
        // The gradients here are with respect to reference coordinates, so we need to
        // transform them to gradients in physical coordinates
        let phi_grad = jacobian_inv_t * phi_I_grad_ref;
        // u_grad += phi_I u_I^T
        u_grad.ger(T::one(), &phi_grad, &u_I, T::one());
    }
    u_grad
}

#[allow(non_snake_case)]
pub fn assemble_generalized_element_elliptic_term<T, SolutionDim, Element>(
    result: MatrixSliceMut<T, SolutionDim, Dynamic>,
    element: &Element,
    g: &impl GeneralizedEllipticOperator<T, SolutionDim, Element::GeometryDim>,
    u: &MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: &impl Quadrature<T, Element::GeometryDim>,
)
where
    T: RealField,
    Element: VolumetricFiniteElement<T>,
    Element::GeometryDim: DimName + DimMin<Element::GeometryDim, Output = Element::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: VolumeFiniteElementAllocator<T, Element::GeometryDim>
        + Allocator<T, Element::GeometryDim, SolutionDim>
        + Allocator<T, SolutionDim, Element::GeometryDim>
{
    let mut f_e = result;

    // TODO: Avoid allocation
    let mut phi_ref = DMatrix::zeros(Element::ReferenceDim::dim(), element.num_nodes());

    let weights = quadrature.weights();
    let points = quadrature.points();

    for (&w, xi) in weights.iter().zip(points) {
        // Compute gradients with respect to reference coords
        element.populate_basis_gradients(MatrixSliceMut::from(&mut phi_ref), xi);

        // Jacobian
        let J = element.reference_jacobian(xi);
        let J_det = J.determinant();

        // TODO: Make error instead of panic?
        let J_inv = J.try_inverse().expect("Jacobian must be invertible");
        let J_inv_t = J_inv.transpose();
        let u_grad = compute_volume_u_grad(&J_inv_t,
                                           MatrixSlice::from(&phi_ref),
                                           MatrixSlice::from(u));

        let g = g.compute_elliptic_term(&u_grad);
        let g_J_inv_t = g.transpose() * &J_inv_t;
        f_e.gemm(w * J_det.abs(), &g_J_inv_t, &phi_ref, T::one());

        // TODO: Remove old comments
        // let u_grad = (u * (&G.transpose() * &J_inv)).transpose();
        // f_e += (g.transpose() * J_inv.transpose() * G) * w * J_det.abs();
    }
}

#[allow(non_snake_case)]
pub fn assemble_generalized_element_stiffness<T, SolutionDim, Element>(
    mut element_matrix: DMatrixSliceMut<T>,
    element: &Element,
    contraction: &impl GeneralizedEllipticContraction<T, SolutionDim, Element::GeometryDim>,
    u: MatrixSlice<T, SolutionDim, Dynamic>,
    quadrature: &impl Quadrature<T, Element::GeometryDim>,
)
where
    T: RealField,
    Element: VolumetricFiniteElement<T>,
    Element::GeometryDim: DimMin<Element::GeometryDim, Output = Element::GeometryDim>,
    SolutionDim: DimName,
    DefaultAllocator: FiniteElementMatrixAllocator<T, SolutionDim, Element::GeometryDim>,
{
    let weights = quadrature.weights();
    let points = quadrature.points();

    // TODO: The structure here is very similar to that of internal force term assembly
    // Can we refactor these to re-use more of the same functionality?

    // Basis function gradients with respect to reference coords
    // TODO: Avoid allocation
    let mut phi_grad_ref = DMatrix::zeros(Element::ReferenceDim::dim(), element.num_nodes());

    for (&w, xi) in weights.iter().zip(points) {
        // Jacobian
        let J = element.reference_jacobian(xi);
        let J_det = J.determinant();

        // TODO: Make error instead of panic?
        let J_inv = J.try_inverse().expect("Jacobian must be invertible");
        let J_inv_t = J_inv.transpose();

        // TODO: Rename gradients to populate_gradients and similarly for basis function values
        element.populate_basis_gradients(MatrixSliceMut::from(&mut phi_grad_ref), xi);

        let u_grad = compute_volume_u_grad(&J_inv_t,
                                           MatrixSlice::from(&phi_grad_ref),
                                           MatrixSlice::from(&u));

        // Compute gradients with respect to physical coords instead of reference coords
        let mut phi_grad = MatrixSliceMut::<_, Element::GeometryDim, Dynamic>::from(&mut phi_grad_ref);
        for mut phi_grad in phi_grad.column_iter_mut() {
            let new_phi_grad = &J_inv_t * &phi_grad;
            phi_grad.copy_from(&new_phi_grad);
        }

        let scale = w * J_det.abs();
        // We need to multiply the contraction result by the scale factor.
        // We do this implicitly by multiplying the basis gradients by its square root.
        // This way we don't have to allocate an additional matrix or complicate
        // the trait.
        let mut G = phi_grad;
        G *= scale.sqrt();

        let (G_rows, _) = G.data.shape();
        let G_slice = coerce_col_major_slice(&G, G_rows, Dynamic::new(G.ncols()));

        contraction.contract_multiple_into(&mut element_matrix, &u_grad, &G_slice);
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
                connectivity.populate_element_variables(MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element), u);

                // TODO: Don't allocate!
                let mut f_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
                assemble_generalized_element_elliptic_term(
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
        + Allocator<T, SolutionDim, Connectivity::GeometryDim>
{
    let u = u.into();
    for (i, connectivity) in connectivity.iter().enumerate() {
        let element = connectivity.element(vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );
        // TODO: Don't allocate!
        let mut u_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
        connectivity.populate_element_variables(MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element), u);

        // TODO: Don't allocate!
        let mut f_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
        assemble_generalized_element_elliptic_term(
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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

    let mut csr_assembler = CsrAssembler::default();
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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

    let mut csr_assembler = CsrParAssembler::default();
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
{
    for (i, connectivity) in connectivity.iter().enumerate() {
        let element = connectivity.element(vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );

        // TODO: Don't allocate!
        let mut u_element = DMatrix::zeros(SolutionDim::dim(), element.num_nodes());
        connectivity.populate_element_variables(MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element), u);

        // TODO: Don't allocate!
        let matrix_size = SolutionDim::dim() * element.num_nodes();
        let mut a_element = DMatrix::zeros(matrix_size, matrix_size);

        assemble_generalized_element_stiffness(
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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
                connectivity.populate_element_variables(MatrixSliceMut::<T, SolutionDim, Dynamic>::from(&mut u_element), u);

                // TODO: Don't allocate!
                let matrix_size = SolutionDim::dim() * element.num_nodes();
                let mut a_element = DMatrix::zeros(matrix_size, matrix_size);

                assemble_generalized_element_stiffness(
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
{
    for (i, connectivity) in connectivity.iter().enumerate() {
        let element = connectivity.element(vertices).expect(
            "All vertices of element are assumed to be in bounds.\
             TODO: Ensure this upon construction of basis?",
        );
        // TODO: Don't allocate!
        let matrix_size = SolutionDim::dim() * element.num_nodes();
        let mut m_element = DMatrix::zeros(matrix_size, matrix_size);
        assemble_generalized_element_mass(
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
    DefaultAllocator: FiniteElementMatrixAllocator<
        T,
        SolutionDim,
        Connectivity::GeometryDim,
    >,
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

// TODO: Write tests for distribute_local_to_global

use crate::allocators::{BiDimAllocator, DimAllocator, TriDimAllocator};
use crate::assembly::global::{gather_global_to_local};
use crate::assembly::buffers::{BasisFunctionBuffer, QuadratureBuffer};
use crate::assembly::local::{
    ElementConnectivityAssembler, ElementMatrixAssembler, ElementScalarAssembler, ElementVectorAssembler,
    QuadratureTable,
};
use crate::assembly::operators::{EllipticContraction, EllipticEnergy, EllipticOperator, Operator};
use crate::define_thread_local_workspace;
use crate::element::VolumetricFiniteElement;
use crate::nalgebra::allocator::Allocator;
use crate::nalgebra::{
    DMatrixSliceMut, DVector, DVectorSlice, DVectorSliceMut, DefaultAllocator, Dim, DimName, Dynamic, MatrixSlice,
    MatrixSliceMut, MatrixSliceMutMN, OMatrix, OPoint, RealField, Scalar, U1,
};
use crate::space::{ElementInSpace, VolumetricFiniteElementSpace};
use crate::util::{clone_upper_to_lower, reshape_to_slice};
use crate::workspace::with_thread_local_workspace;
use crate::Symmetry;
use eyre::eyre;
use itertools::izip;

// TODO: Move this to the right spot and don't make it pub(crate)
#[allow(non_snake_case)]
pub(crate) fn compute_volume_u_grad<'a, T, GeometryDim, SolutionDim>(
    jacobian_inv_t: &OMatrix<T, GeometryDim, GeometryDim>,
    phi_grad_ref: impl Into<MatrixSlice<'a, T, GeometryDim, Dynamic>>,
    u: impl Into<MatrixSlice<'a, T, SolutionDim, Dynamic>>,
) -> OMatrix<T, GeometryDim, SolutionDim>
where
    T: RealField,
    SolutionDim: DimName,
    GeometryDim: DimName,
    DefaultAllocator: BiDimAllocator<T, SolutionDim, GeometryDim>,
{
    let phi_grad_ref = phi_grad_ref.into();
    let u = u.into();
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
    let mut u_grad = OMatrix::<_, GeometryDim, SolutionDim>::zeros();
    for (phi_I_grad_ref, u_I) in G_ref.column_iter().zip(u.column_iter()) {
        // Instead of computing each column of G (gradients in physical space),
        // we instead use the reference coordinates and only multiply by J^{-T} for the end result,
        // which involves substantially less work
        // TODO: Rewrite the earlier explanation to clarify this
        // (i.e. that we compute J^{-T} * G_ref * U^T insetad of G U^T)
        u_grad.ger(T::one(), &phi_I_grad_ref, &u_I, T::one());
    }
    jacobian_inv_t * u_grad
}

/// TODO: The builder here is pretty complex. Is it possible to simplify without losing too
/// much type safety?
pub struct ElementEllipticAssemblerBuilder<Space, Op, QTable, U> {
    space: Space,
    op: Op,
    qtable: QTable,
    u: U,
}

impl ElementEllipticAssemblerBuilder<(), (), (), ()> {
    pub fn new() -> Self {
        Self {
            space: (),
            op: (),
            qtable: (),
            u: (),
        }
    }
}

impl<Op, QTable, U> ElementEllipticAssemblerBuilder<(), Op, QTable, U> {
    pub fn with_finite_element_space<Space>(
        self,
        space: &Space,
    ) -> ElementEllipticAssemblerBuilder<&Space, Op, QTable, U> {
        ElementEllipticAssemblerBuilder {
            space,
            op: self.op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

impl<Space, QTable, U> ElementEllipticAssemblerBuilder<Space, (), QTable, U> {
    pub fn with_operator<Op>(self, op: &Op) -> ElementEllipticAssemblerBuilder<Space, &Op, QTable, U> {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

impl<Space, Op, U> ElementEllipticAssemblerBuilder<Space, Op, (), U> {
    pub fn with_quadrature_table<QTable>(
        self,
        qtable: QTable,
    ) -> ElementEllipticAssemblerBuilder<Space, Op, QTable, U> {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op: self.op,
            qtable,
            u: self.u,
        }
    }
}

impl<Space, Op, QTable> ElementEllipticAssemblerBuilder<Space, Op, QTable, ()> {
    pub fn with_u<'a, T>(
        self,
        u: impl Into<DVectorSlice<'a, T>>,
    ) -> ElementEllipticAssemblerBuilder<Space, Op, QTable, DVectorSlice<'a, T>>
    where
        T: Scalar,
    {
        ElementEllipticAssemblerBuilder {
            space: self.space,
            op: self.op,
            qtable: self.qtable,
            u: u.into(),
        }
    }
}

impl<'a, T, Space, Op, QTable> ElementEllipticAssemblerBuilder<&'a Space, &'a Op, &'a QTable, DVectorSlice<'a, T>>
where
    T: Scalar,
    QTable: ?Sized,
{
    pub fn build(self) -> ElementEllipticAssembler<'a, T, Space, Op, QTable> {
        ElementEllipticAssembler {
            space: self.space,
            op: self.op,
            qtable: self.qtable,
            u: self.u,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ElementEllipticAssembler<'a, T: Scalar, Space, Op, QTable: ?Sized> {
    space: &'a Space,
    op: &'a Op,
    qtable: &'a QTable,
    u: DVectorSlice<'a, T>,
}

impl<'a, T, Space, Op, QTable> ElementConnectivityAssembler for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: Scalar,
    Space: VolumetricFiniteElementSpace<T>,
    Op: Operator<T, Space::GeometryDim>,
    QTable: ?Sized,
    DefaultAllocator: DimAllocator<T, Space::GeometryDim>,
{
    fn solution_dim(&self) -> usize {
        Op::SolutionDim::dim()
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
struct EllipticAssemblerWorkspace<T, GeometryDim, Data>
where
    T: Scalar,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    u_element: DVector<T>,
    quadrature_buffer: QuadratureBuffer<T, GeometryDim, Data>,
    basis_buffer: BasisFunctionBuffer<T>,
}

impl<T, GeometryDim, Data> Default for EllipticAssemblerWorkspace<T, GeometryDim, Data>
where
    T: RealField,
    GeometryDim: DimName,
    DefaultAllocator: Allocator<T, GeometryDim>,
{
    fn default() -> Self {
        Self {
            u_element: DVector::zeros(0),
            quadrature_buffer: Default::default(),
            basis_buffer: Default::default(),
        }
    }
}

define_thread_local_workspace!(WORKSPACE);

impl<'a, T, Space, Op, QTable> ElementScalarAssembler<T> for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Op: EllipticEnergy<T, Space::ReferenceDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Parameters> + ?Sized,
    DefaultAllocator: TriDimAllocator<T, Op::SolutionDim, Space::GeometryDim, Space::ReferenceDim>,
{
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T> {
        let s = self.solution_dim();
        let n = self.element_node_count(element_index);

        with_thread_local_workspace(
            &WORKSPACE,
            |ws: &mut EllipticAssemblerWorkspace<T, Space::ReferenceDim, Op::Parameters>| {
                ws.basis_buffer.resize(n, Space::ReferenceDim::dim());
                ws.basis_buffer
                    .populate_element_nodes_from_space(element_index, self.space);
                ws.u_element.resize_vertically_mut(s * n, T::zero());
                gather_global_to_local(&self.u, &mut ws.u_element, ws.basis_buffer.element_nodes(), s);

                ws.quadrature_buffer
                    .populate_element_quadrature_from_table(element_index, self.qtable);

                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                compute_element_elliptic_energy(
                    &element,
                    self.op,
                    DVectorSlice::from(&ws.u_element),
                    ws.quadrature_buffer.weights(),
                    ws.quadrature_buffer.points(),
                    ws.quadrature_buffer.data(),
                    ws.basis_buffer.element_gradients_mut(),
                )
            },
        )
    }
}

impl<'a, T, Space, Op, QTable> ElementVectorAssembler<T> for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Op: EllipticOperator<T, Space::ReferenceDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Parameters> + ?Sized,
    DefaultAllocator: TriDimAllocator<T, Op::SolutionDim, Space::GeometryDim, Space::ReferenceDim>,
{
    #[allow(non_snake_case)]
    fn assemble_element_vector_into(&self, element_index: usize, output: DVectorSliceMut<T>) -> eyre::Result<()> {
        let s = self.solution_dim();
        let n = self.element_node_count(element_index);
        assert_eq!(output.len(), s * n, "Output vector dimension mismatch");

        with_thread_local_workspace(
            &WORKSPACE,
            |ws: &mut EllipticAssemblerWorkspace<T, Space::ReferenceDim, Op::Parameters>| {
                ws.basis_buffer.resize(n, Space::ReferenceDim::dim());
                ws.basis_buffer
                    .populate_element_nodes_from_space(element_index, self.space);
                ws.u_element.resize_vertically_mut(s * n, T::zero());
                gather_global_to_local(&self.u, &mut ws.u_element, ws.basis_buffer.element_nodes(), s);

                ws.quadrature_buffer
                    .populate_element_quadrature_from_table(element_index, self.qtable);

                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                assemble_element_elliptic_vector(
                    output,
                    &element,
                    self.op,
                    DVectorSlice::from(&ws.u_element),
                    ws.quadrature_buffer.weights(),
                    ws.quadrature_buffer.points(),
                    ws.quadrature_buffer.data(),
                    ws.basis_buffer.element_gradients_mut(),
                )
            },
        )
    }
}

impl<'a, T, Space, Op, QTable> ElementMatrixAssembler<T> for ElementEllipticAssembler<'a, T, Space, Op, QTable>
where
    T: RealField,
    Space: VolumetricFiniteElementSpace<T>,
    Op: EllipticContraction<T, Space::ReferenceDim>,
    QTable: QuadratureTable<T, Space::ReferenceDim, Data = Op::Parameters> + ?Sized,
    DefaultAllocator: TriDimAllocator<T, Op::SolutionDim, Space::GeometryDim, Space::ReferenceDim>,
{
    #[allow(non_snake_case)]
    fn assemble_element_matrix_into(&self, element_index: usize, output: DMatrixSliceMut<T>) -> eyre::Result<()> {
        let s = self.solution_dim();
        let n = self.element_node_count(element_index);
        assert_eq!(output.nrows(), s * n, "Output matrix dimension mismatch");
        assert_eq!(output.ncols(), s * n, "Output matrix dimension mismatch");

        with_thread_local_workspace(
            &WORKSPACE,
            |ws: &mut EllipticAssemblerWorkspace<T, Space::ReferenceDim, Op::Parameters>| {
                ws.basis_buffer.resize(n, Space::ReferenceDim::dim());
                ws.basis_buffer
                    .populate_element_nodes_from_space(element_index, self.space);
                ws.u_element.resize_vertically_mut(s * n, T::zero());
                gather_global_to_local(&self.u, &mut ws.u_element, ws.basis_buffer.element_nodes(), s);

                ws.quadrature_buffer
                    .populate_element_quadrature_from_table(element_index, self.qtable);

                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                assemble_element_elliptic_matrix(
                    output,
                    &element,
                    self.op,
                    DVectorSlice::from(&ws.u_element),
                    ws.quadrature_buffer.weights(),
                    ws.quadrature_buffer.points(),
                    ws.quadrature_buffer.data(),
                    ws.basis_buffer.element_gradients_mut(),
                )
            },
        )
    }
}

/// Assembles the element (derivative) matrix associated with the given elliptic operator.
///
/// Given a finite element, an elliptic operator and a quadrature rule and associated operator
/// parameters, stores the resulting element matrix in the provided output vector.
/// This is effectively the **element stiffness matrix** for the given element and elliptic
/// operator.
///
/// See the documentation for [`EllipticContraction`] for more information about
/// contraction of elliptic operators.
///
/// The computation requires a buffer for evaluating gradients. The buffer must be able to
/// store gradients for each node in the element.
///
/// # Panics
///
/// Panics if the quadrature data arrays do not have the same lengths.
///
/// Panics if the number of columns in the gradient buffer is not equal to the number of nodes
/// in the element.
pub fn assemble_element_elliptic_matrix<T, Element, Contraction>(
    mut output: DMatrixSliceMut<T>,
    element: &Element,
    operator: &Contraction,
    u_element: DVectorSlice<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    quadrature_data: &[Contraction::Parameters],
    basis_gradients_buffer: MatrixSliceMutMN<T, Element::ReferenceDim, Dynamic>,
) -> eyre::Result<()>
where
    T: RealField,
    // We only support volumetric elements atm
    Element: VolumetricFiniteElement<T>,
    Contraction: EllipticContraction<T, Element::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Contraction::SolutionDim, Element::GeometryDim>,
{
    assert_eq!(quadrature_weights.len(), quadrature_points.len());
    assert_eq!(quadrature_points.len(), quadrature_data.len());
    assert_eq!(basis_gradients_buffer.ncols(), element.num_nodes());

    let d = Element::GeometryDim::dim();
    let s = Contraction::SolutionDim::name();
    let n = element.num_nodes();
    assert_eq!(
        u_element.len(),
        s.value() * n,
        "Local element dofs (u_element) dimension mismatch"
    );
    assert_eq!(output.nrows(), s.value() * n, "Output matrix dimension mismatch");
    assert_eq!(output.ncols(), s.value() * n, "Output matrix dimension mismatch");

    output.fill(T::zero());

    let mut phi_grad = basis_gradients_buffer;

    let quadrature_iter = izip!(quadrature_weights, quadrature_points, quadrature_data);
    for (&weight, point, data) in quadrature_iter {
        let j = element.reference_jacobian(point);
        let j_det = j.determinant();
        let j_inv = j
            .try_inverse()
            // TODO: Return a "proper" error instead of using eyre
            .ok_or_else(|| eyre!("Singular element Jacobian encountered"))?;
        let j_inv_t = j_inv.transpose();

        // First populate gradients with respect to reference coords
        element.populate_basis_gradients(MatrixSliceMut::from(&mut phi_grad), &point);

        // We currently have to compute u_grad by providing reference gradients
        let u_element = reshape_to_slice(&u_element, (s, Dynamic::new(n)));
        let u_grad = compute_volume_u_grad(&j_inv_t, &phi_grad, u_element);

        // Transform reference gradients to gradients with respect to physical coords
        for mut phi_grad in phi_grad.column_iter_mut() {
            let new_phi_grad = &j_inv_t * &phi_grad;
            phi_grad.copy_from(&new_phi_grad);
        }

        // Note: We need to multiply the contraction result by a scale factor to account for the
        // quadrature weight and jacobian determinant
        let scale = weight * j_det.abs();
        let phi_grad = reshape_to_slice(&phi_grad, (Dynamic::new(d * n), U1::name()));
        operator.accumulate_contractions_into(
            DMatrixSliceMut::from(&mut output),
            scale,
            &u_grad,
            phi_grad.clone(),
            phi_grad,
            data,
        );
    }

    if matches!(operator.symmetry(), Symmetry::Symmetric) {
        clone_upper_to_lower(&mut output);
    }

    Ok(())
}

/// Assemble the element vector associated with the elliptic operator.
///
/// Given a finite element, an elliptic operator and a quadrature rule and associated operator
/// parameters, stores the resulting element vector in the provided output vector.
///
/// See the documentation for [`EllipticOperator`] for more information about elliptic operators.
///
/// The computation requires a buffer for evaluating gradients. The buffer must be able to
/// store gradients for each node in the element.
///
/// # Panics
///
/// Panics if the quadrature data arrays do not have the same lengths.
///
/// Panics if the number of columns in the gradient buffer is not equal to the number of nodes
/// in the element.
pub fn assemble_element_elliptic_vector<T, Element, Operator>(
    mut output: DVectorSliceMut<T>,
    element: &Element,
    operator: &Operator,
    u_element: DVectorSlice<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    quadrature_data: &[Operator::Parameters],
    basis_gradients_buffer: MatrixSliceMutMN<T, Element::ReferenceDim, Dynamic>,
) -> eyre::Result<()>
where
    T: RealField,
    // We only support volumetric elements atm
    Element: VolumetricFiniteElement<T>,
    Operator: EllipticOperator<T, Element::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Operator::SolutionDim, Element::GeometryDim>,
{
    assert_eq!(quadrature_weights.len(), quadrature_points.len());
    assert_eq!(quadrature_points.len(), quadrature_data.len());
    assert_eq!(basis_gradients_buffer.ncols(), element.num_nodes());

    let s = Operator::SolutionDim::dim();
    let n = element.num_nodes();
    assert_eq!(
        u_element.len(),
        s * n,
        "Local element dofs (u_element) dimension mismatch"
    );
    assert_eq!(output.nrows(), s * n, "Output vector dimension mismatch");

    output.fill(T::zero());

    let mut phi_grad_ref = basis_gradients_buffer;

    let quadrature_iter = izip!(quadrature_weights, quadrature_points, quadrature_data);
    for (&weight, point, data) in quadrature_iter {
        let j = element.reference_jacobian(point);
        let j_det = j.determinant();
        let j_inv = j
            .try_inverse()
            // TODO: Return a "proper" error instead of using eyre
            .ok_or_else(|| eyre!("Singular element Jacobian encountered"))?;
        let j_inv_t = j_inv.transpose();

        // First populate gradients with respect to reference coords
        element.populate_basis_gradients(MatrixSliceMut::from(&mut phi_grad_ref), &point);

        let u_element =
            MatrixSlice::from_slice_generic(u_element.as_slice(), Operator::SolutionDim::name(), Dynamic::new(n));
        let u_grad = compute_volume_u_grad(&j_inv_t, &phi_grad_ref, u_element);

        // We want to compute the vector
        //
        // [ g^T phi_1 ]
        // [ g^T phi_2 ]
        // [   ...     ]
        // [ g^T phi_n ]
        //
        // We can reorganize this expression into the alternative expression
        //
        // [ g^T phi_1     g^T phi_2     ...    g^T phi_n ] = g^T P
        // where
        // P = [ phi_1 phi_2 ... phi_n ] = J^{-T} * [ phi_1^ref phi_2^ref ... phi_n^ref ]
        //   = J^{-T} P_0
        // and phi_i^ref represents the gradient with respect to reference coordinates.
        // Hence we may compute (g^T J^{-T}) P_0

        let mut output =
            MatrixSliceMutMN::from_slice_generic(output.as_mut_slice(), Operator::SolutionDim::name(), Dynamic::new(n));
        let g = operator.compute_elliptic_operator(&u_grad, data);
        let g_t_j_inv_t = g.transpose() * j_inv_t;
        output.gemm(weight * j_det.abs(), &g_t_j_inv_t, &phi_grad_ref, T::one());
    }

    Ok(())
}

/// Numerically integrate the elliptic energy over the given element.
///
/// Using the provided weights of `u` associated with the finite element, the provided quadrature
/// and provided finite element $K$, this function approximates the integral
///
/// $$ \int_K \psi(\nabla u_h) \dx. $$
///
/// See the documentation for [`EllipticEnergy`] for more information about elliptic energies.
///
/// The computation requires a buffer for evaluating gradients. The buffer must be able to
/// store gradients for each node in the element.
///
/// # Panics
///
/// Panics if the quadrature data arrays do not have the same lengths.
///
/// Panics if the number of columns in the gradient buffer is not equal to the number of nodes
/// in the element.
pub fn compute_element_elliptic_energy<T, Element, Operator>(
    element: &Element,
    operator: &Operator,
    u_element: DVectorSlice<T>,
    quadrature_weights: &[T],
    quadrature_points: &[OPoint<T, Element::ReferenceDim>],
    quadrature_data: &[Operator::Parameters],
    basis_gradients_buffer: MatrixSliceMutMN<T, Element::ReferenceDim, Dynamic>,
) -> eyre::Result<T>
where
    T: RealField,
    // We only support volumetric elements atm
    Element: VolumetricFiniteElement<T>,
    Operator: EllipticEnergy<T, Element::GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, Operator::SolutionDim, Element::GeometryDim>,
{
    let s = Operator::SolutionDim::dim();
    let n = element.num_nodes();
    assert_eq!(quadrature_weights.len(), quadrature_points.len());
    assert_eq!(quadrature_points.len(), quadrature_data.len());
    assert_eq!(basis_gradients_buffer.ncols(), n);
    assert_eq!(
        u_element.len(),
        s * n,
        "Local element dofs (u_element) dimension mismatch"
    );

    let mut phi_grad_ref = basis_gradients_buffer;

    let mut integral = T::zero();
    let quadrature_iter = izip!(quadrature_weights, quadrature_points, quadrature_data);
    for (&weight, point, data) in quadrature_iter {
        // All this stuff is basically the same for energy, vector and matrix. TODO: Consolidate?

        let j = element.reference_jacobian(point);
        let j_det = j.determinant();
        let j_inv = j
            .try_inverse()
            // TODO: Return a "proper" error instead of using eyre
            .ok_or_else(|| eyre!("Singular element Jacobian encountered"))?;
        let j_inv_t = j_inv.transpose();

        // First populate gradients with respect to reference coords
        element.populate_basis_gradients(MatrixSliceMut::from(&mut phi_grad_ref), &point);

        let u_element =
            MatrixSlice::from_slice_generic(u_element.as_slice(), Operator::SolutionDim::name(), Dynamic::new(n));
        let u_grad = compute_volume_u_grad(&j_inv_t, &phi_grad_ref, u_element);

        let psi = operator.compute_energy(&u_grad, data);

        integral += weight * j_det.abs() * psi;
    }

    Ok(integral)
}

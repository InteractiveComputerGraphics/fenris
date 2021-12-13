//! Tools for integrating functions on finite element spaces.
use crate::allocators::{BiDimAllocator, DimAllocator, TriDimAllocator};
use crate::assembly::buffers::{BasisFunctionBuffer, QuadratureBuffer};
use crate::assembly::global::gather_global_to_local;
use crate::assembly::local::{ElementConnectivityAssembler, ElementScalarAssembler, QuadratureTable};
use crate::define_thread_local_workspace;
use crate::element::{FiniteElement, VolumetricFiniteElement};
use crate::nalgebra::{DVector, DefaultAllocator, DimName, OMatrix, OPoint, Scalar, U1};
use crate::quadrature::Quadrature;
use crate::space::{ElementInSpace, FiniteElementSpace, VolumetricFiniteElementSpace};
use crate::util::{reshape_to_slice, try_transmute_ref};
use crate::workspace::with_thread_local_workspace;
use crate::SmallDim;
use eyre::eyre;
use nalgebra::{DVectorSlice, Dynamic, OVector, RealField};
use std::marker::PhantomData;

/// Computes the Riemannian volume form for the given dimensions.
///
/// TODO: This is not actively tested at the moment, need to do this.
pub fn volume_form<T, GeometryDim, ReferenceDim>(jacobian: &OMatrix<T, GeometryDim, ReferenceDim>) -> T
where
    T: RealField,
    GeometryDim: SmallDim,
    ReferenceDim: SmallDim,
    DefaultAllocator: BiDimAllocator<T, GeometryDim, ReferenceDim>,
{
    if GeometryDim::is::<ReferenceDim>() {
        let jacobian: &OMatrix<T, GeometryDim, GeometryDim> =
            try_transmute_ref(jacobian).expect("This cannot fail since we know that GeometryDim == ReferenceDim");
        jacobian.determinant().abs()
    } else {
        // TODO: Specialize other dimension combinations
        (jacobian.transpose() * jacobian).determinant().sqrt()
    }
}

/// A vector-valued function $f(x, u)$.
///
/// Here $u = u(x)$ is a function $u: \Omega \rightarrow \mathbb{R}^s$.
///
/// Functions of this type can be integrated over both volumes and surfaces.
pub trait Function<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator:
        DimAllocator<T, Self::SolutionDim> + DimAllocator<T, Self::OutputDim> + DimAllocator<T, GeometryDim>,
{
    type OutputDim: SmallDim;
    type SolutionDim: SmallDim;

    fn evaluate(&self, x: &OPoint<T, GeometryDim>, u: &OVector<T, Self::SolutionDim>) -> OVector<T, Self::OutputDim>;
}

impl<'a, T, GeometryDim, F> Function<T, GeometryDim> for &'a F
where
    T: Scalar,
    GeometryDim: SmallDim,
    F: Function<T, GeometryDim>,
    DefaultAllocator: DimAllocator<T, F::SolutionDim> + DimAllocator<T, F::OutputDim> + DimAllocator<T, GeometryDim>,
{
    type OutputDim = F::OutputDim;
    type SolutionDim = F::SolutionDim;

    fn evaluate(&self, x: &OPoint<T, GeometryDim>, u: &OVector<T, Self::SolutionDim>) -> OVector<T, Self::OutputDim> {
        F::evaluate(self, x, u)
    }
}

#[derive(Debug)]
pub struct Integrand<SolutionDim = (), F = ()> {
    marker: PhantomData<SolutionDim>,
    function: F,
}

impl Integrand {
    pub fn new_with_solution_dim<SolutionDim>() -> Integrand<SolutionDim> {
        Integrand {
            marker: Default::default(),
            function: (),
        }
    }
}

impl<SolutionDim> Integrand<SolutionDim, ()> {
    pub fn with_function<F>(self, f: F) -> Integrand<SolutionDim, F> {
        Integrand {
            marker: Default::default(),
            function: f,
        }
    }

    pub fn with_volume_function<F>(self, f: F) -> Integrand<SolutionDim, VolumeIntegrand<F>> {
        Integrand {
            marker: Default::default(),
            function: VolumeIntegrand(f),
        }
    }
}

impl<F, T, OutputDim, SolutionDim, GeometryDim> Function<T, GeometryDim> for Integrand<SolutionDim, F>
where
    F: Fn(&OPoint<T, GeometryDim>, &OVector<T, SolutionDim>) -> OVector<T, OutputDim>,
    T: Scalar,
    OutputDim: SmallDim,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + DimAllocator<T, SolutionDim> + DimAllocator<T, GeometryDim>,
{
    type OutputDim = OutputDim;
    type SolutionDim = SolutionDim;

    fn evaluate(&self, x: &OPoint<T, GeometryDim>, u: &OVector<T, SolutionDim>) -> OVector<T, OutputDim> {
        (self.function)(x, u)
    }
}

/// A vector-valued function $f(x, u, \nabla u)$.
///
/// Here $u = u(x)$ is a function $u: \Omega \rightarrow \mathbb{R}^s$.
///
/// Functions of this type can be integrated over both volumes and surfaces.
pub trait VolumeFunction<T, GeometryDim>
where
    T: Scalar,
    GeometryDim: SmallDim,
    DefaultAllocator: DimAllocator<T, Self::OutputDim> + BiDimAllocator<T, Self::SolutionDim, GeometryDim>,
{
    type OutputDim: SmallDim;
    type SolutionDim: SmallDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        u: &OVector<T, Self::SolutionDim>,
        u_grad: &OMatrix<T, GeometryDim, Self::SolutionDim>,
    ) -> OVector<T, Self::OutputDim>;
}

impl<'a, T, GeometryDim, F> VolumeFunction<T, GeometryDim> for &'a F
where
    T: Scalar,
    GeometryDim: SmallDim,
    F: VolumeFunction<T, GeometryDim>,
    DefaultAllocator: BiDimAllocator<T, F::SolutionDim, GeometryDim> + DimAllocator<T, F::OutputDim>,
{
    type OutputDim = F::OutputDim;
    type SolutionDim = F::SolutionDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        u: &OVector<T, Self::SolutionDim>,
        u_grad: &OMatrix<T, GeometryDim, Self::SolutionDim>,
    ) -> OVector<T, Self::OutputDim> {
        F::evaluate(self, x, u, u_grad)
    }
}

impl<F, T, OutputDim, SolutionDim, GeometryDim> VolumeFunction<T, GeometryDim>
    for Integrand<SolutionDim, VolumeIntegrand<F>>
where
    F: Fn(
        &OPoint<T, GeometryDim>,
        &OVector<T, SolutionDim>,
        &OMatrix<T, GeometryDim, SolutionDim>,
    ) -> OVector<T, OutputDim>,
    T: Scalar,
    OutputDim: SmallDim,
    SolutionDim: SmallDim,
    GeometryDim: SmallDim,
    DefaultAllocator: DimAllocator<T, OutputDim> + BiDimAllocator<T, SolutionDim, GeometryDim>,
{
    type OutputDim = OutputDim;
    type SolutionDim = SolutionDim;

    fn evaluate(
        &self,
        x: &OPoint<T, GeometryDim>,
        u: &OVector<T, SolutionDim>,
        u_grad: &OMatrix<T, GeometryDim, SolutionDim>,
    ) -> OVector<T, OutputDim> {
        (self.function.0)(x, u, u_grad)
    }
}

pub struct IntegrationWorkspace<T: Scalar> {
    basis_buffer: BasisFunctionBuffer<T>,
}

impl<T: RealField> Default for IntegrationWorkspace<T> {
    fn default() -> Self {
        Self {
            basis_buffer: BasisFunctionBuffer::default(),
        }
    }
}

/// Integrates the given function on the given element with the provided quadrature and interpolation weights.
pub fn integrate_over_element<'a, T, F, Element, QuadratureRule, IntoDVectorSlice>(
    integrand: F,
    element: &Element,
    quadrature: QuadratureRule,
    interpolation_weights: IntoDVectorSlice,
    workspace: &mut IntegrationWorkspace<T>,
) -> OVector<T, F::OutputDim>
where
    T: RealField,
    F: Function<T, Element::GeometryDim>,
    Element: FiniteElement<T>,
    QuadratureRule: Quadrature<T, Element::ReferenceDim>,
    IntoDVectorSlice: Into<DVectorSlice<'a, T>>,
    DefaultAllocator: TriDimAllocator<T, F::SolutionDim, Element::GeometryDim, Element::ReferenceDim>
        // This is a separate bound because we generally don't need to mix the output dimension
        // with the other dimensions, so this way the bounds necessary for downstream consumers
        // are somewhat relaxed (the output dimension is often *fixed*, so maybe no bounds at all are necessary)
        + DimAllocator<T, F::OutputDim>,
{
    let interpolation_weights = interpolation_weights.into();

    let n = element.num_nodes();
    let (weights, points) = (quadrature.weights(), quadrature.points());
    let basis_buffer = &mut workspace.basis_buffer;
    basis_buffer.resize(element.num_nodes(), Element::ReferenceDim::dim());

    let mut result = OVector::<T, F::OutputDim>::zeros();
    for (w, p_ref) in weights.iter().zip(points) {
        element.populate_basis(basis_buffer.element_basis_values_mut(), p_ref);
        let u_h = crate::util::compute_interpolation(
            interpolation_weights,
            DVectorSlice::from_slice(basis_buffer.element_basis_values(), n),
        );
        let x = element.map_reference_coords(p_ref);
        let jacobian = element.reference_jacobian(p_ref);
        let f = integrand.evaluate(&x, &u_h);
        let volume_form = volume_form(&jacobian);

        result += f * (w.clone() * volume_form);
    }

    result
}

#[derive(Debug)]
pub enum IntegrationFailure {
    SingularJacobian,
}

/// Integrates the given volume function on the given element with the provided quadrature and interpolation weights.
pub fn integrate_over_volume_element<'a, T, Element, F>(
    function: F,
    element: &Element,
    quadrature: impl Quadrature<T, Element::ReferenceDim>,
    interpolation_weights: impl Into<DVectorSlice<'a, T>>,
    workspace: &mut IntegrationWorkspace<T>,
) -> Result<OVector<T, F::OutputDim>, IntegrationFailure>
where
    T: RealField,
    F: VolumeFunction<T, Element::GeometryDim>,
    Element: VolumetricFiniteElement<T>,
    DefaultAllocator:
        TriDimAllocator<T, F::SolutionDim, Element::GeometryDim, Element::ReferenceDim> + DimAllocator<T, F::OutputDim>,
{
    let interpolation_weights = interpolation_weights.into();
    let n = element.num_nodes();
    let r = Element::ReferenceDim::dim();
    let basis_buffer = &mut workspace.basis_buffer;
    basis_buffer.resize(element.num_nodes(), Element::ReferenceDim::dim());

    let mut result = OVector::<T, F::OutputDim>::zeros();
    for (w, p_ref) in quadrature.weights().iter().zip(quadrature.points()) {
        let x = element.map_reference_coords(p_ref);
        let jacobian = element.reference_jacobian(p_ref);
        let jacobian_inv_t = jacobian
            .transpose()
            .try_inverse()
            .ok_or_else(|| IntegrationFailure::SingularJacobian)?;

        // First we compute u_h
        element.populate_basis(basis_buffer.element_basis_values_mut(), p_ref);
        let u_h = crate::util::compute_interpolation(
            interpolation_weights,
            DVectorSlice::from_slice(basis_buffer.element_basis_values(), n),
        );

        // Then we compute u_h_grad. To do so we first compute the gradient with respect to *reference element coords*,
        // then we transform this to physical coordinates by the inverse transposed Jacobian
        element.populate_basis_gradients(basis_buffer.element_gradients_mut(), p_ref);
        let reference_gradients = basis_buffer.element_gradients::<Element::ReferenceDim>();
        let reference_gradients = reshape_to_slice(&reference_gradients, (Dynamic::new(r * n), U1::name()));
        let u_h_ref_grad: OMatrix<T, Element::ReferenceDim, F::SolutionDim> =
            crate::util::compute_interpolation_gradient(interpolation_weights, &reference_gradients);
        let u_h_grad = jacobian_inv_t * u_h_ref_grad;
        let f = function.evaluate(&x, &u_h, &u_h_grad);
        let volume_form = volume_form(&jacobian);

        result += f * (w.clone() * volume_form);
    }

    Ok(result)
}

pub struct ElementIntegralAssembler<'a, T, F, Space, QTable>
where
    T: Scalar,
{
    space: &'a Space,
    u: DVectorSlice<'a, T>,
    integrand: F,
    qtable: &'a QTable,
}

pub struct ElementIntegralVolumeAssembler<'a, T, F, Space, QTable>
where
    T: Scalar,
{
    space: &'a Space,
    u: DVectorSlice<'a, T>,
    integrand: F,
    qtable: &'a QTable,
}

pub struct ElementIntegralAssemblerBuilder<'a, T, F, Space, QTable>
where
    T: Scalar,
{
    space: Option<&'a Space>,
    u: Option<DVectorSlice<'a, T>>,
    integrand: Option<F>,
    qtable: Option<&'a QTable>,
}

pub struct VolumeIntegrand<T>(pub T);

impl<'a, T, F, Space, QTable> ElementIntegralAssemblerBuilder<'a, T, F, Space, QTable>
where
    T: Scalar,
{
    pub fn new() -> Self {
        Self {
            space: None,
            u: None,
            integrand: None,
            qtable: None,
        }
    }

    pub fn with_space(self, space: &'a Space) -> Self {
        Self {
            space: Some(space),
            ..self
        }
    }

    pub fn with_quadrature_table(self, qtable: &'a QTable) -> Self {
        Self {
            qtable: Some(qtable),
            ..self
        }
    }

    pub fn with_interpolation_weights(self, u: impl Into<DVectorSlice<'a, T>>) -> Self {
        Self {
            u: Some(u.into()),
            ..self
        }
    }

    pub fn with_integrand(self, integrand: F) -> Self {
        Self {
            integrand: Some(integrand),
            ..self
        }
    }

    pub fn build_integrator(self) -> ElementIntegralAssembler<'a, T, F, Space, QTable>
    where
        Space: FiniteElementSpace<T>,
        F: Function<T, Space::GeometryDim>,
        DefaultAllocator:
            TriDimAllocator<T, F::SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
    {
        // We take all the trait bounds here so that we can do some sanity checking.
        // This makes it much easier for the user to debug where something went wrong,
        // such as mismatch between interpolation length vector size and number of nodes in space etc.
        let assembler = ElementIntegralAssembler {
            space: self.space.expect("Must provide space"),
            u: self.u.expect("Must provide interpolation weights"),
            integrand: self.integrand.expect("Must provide integrand"),
            qtable: self.qtable.expect("Must provide quadrature table"),
        };

        let ndof = assembler.space.num_nodes() * F::SolutionDim::dim();
        assert_eq!(
            assembler.u.len(),
            ndof,
            "Size of interpolation weight vector does not match expected number of DOFs ( {} x {} )",
            F::SolutionDim::dim(),
            assembler.space.num_nodes()
        );

        assembler
    }

    pub fn build_volume_integrator(self) -> ElementIntegralVolumeAssembler<'a, T, F, Space, QTable>
    where
        Space: VolumetricFiniteElementSpace<T>,
        F: VolumeFunction<T, Space::ReferenceDim>,
        DefaultAllocator:
            TriDimAllocator<T, F::SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
    {
        let assembler = ElementIntegralVolumeAssembler {
            space: self.space.expect("Must provide space"),
            u: self.u.expect("Must provide interpolation weights"),
            integrand: self.integrand.expect("Must provide integrand"),
            qtable: self.qtable.expect("Must provide quadrature table"),
        };

        let ndof = assembler.space.num_nodes() * F::SolutionDim::dim();
        assert_eq!(
            assembler.u.len(),
            ndof,
            "Size of interpolation weight vector does not match expected number of DOFs ( {} x {} )",
            F::SolutionDim::dim(),
            assembler.space.num_nodes()
        );

        assembler
    }
}

impl<'a, T, F, Space, QTable> ElementConnectivityAssembler for ElementIntegralAssembler<'a, T, F, Space, QTable>
where
    T: Scalar,
    Space: FiniteElementSpace<T>,
    F: Function<T, Space::GeometryDim>,
    DefaultAllocator:
        TriDimAllocator<T, F::SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
{
    fn solution_dim(&self) -> usize {
        F::SolutionDim::dim()
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

define_thread_local_workspace!(WORKSPACE);

struct ElementIntegralAssemblerWorkspace<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: DimAllocator<T, D>,
{
    integration_workspace: IntegrationWorkspace<T>,
    quadrature_buffer: QuadratureBuffer<T, D>,
    local_interpolation_weights: DVector<T>,
    nodes: Vec<usize>,
}

impl<T, D> Default for ElementIntegralAssemblerWorkspace<T, D>
where
    T: RealField,
    D: DimName,
    DefaultAllocator: DimAllocator<T, D>,
{
    fn default() -> Self {
        Self {
            integration_workspace: Default::default(),
            quadrature_buffer: Default::default(),
            local_interpolation_weights: DVector::zeros(0),
            nodes: Default::default(),
        }
    }
}

impl<'a, T, F, Space, QTable> ElementScalarAssembler<T> for ElementIntegralAssembler<'a, T, F, Space, QTable>
where
    T: RealField,
    F: Function<T, Space::GeometryDim>,
    Space: FiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator:
        TriDimAllocator<T, F::SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
{
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T> {
        let n = self.element_node_count(element_index);
        let s = self.solution_dim();
        let element_ndof = n * s;
        let integral = with_thread_local_workspace(
            &WORKSPACE,
            |workspace: &mut ElementIntegralAssemblerWorkspace<T, Space::ReferenceDim>| {
                workspace
                    .quadrature_buffer
                    .populate_element_weights_and_points_from_table(element_index, self.qtable);
                workspace
                    .local_interpolation_weights
                    .resize_vertically_mut(element_ndof, T::zero());
                workspace.nodes.resize(n, usize::MAX);
                self.populate_element_nodes(&mut workspace.nodes, element_index);
                let u_local = &mut workspace.local_interpolation_weights;
                let quadrature = workspace.quadrature_buffer.weights_and_points();
                gather_global_to_local(&self.u, &mut *u_local, &workspace.nodes, s);
                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                integrate_over_element(
                    &self.integrand,
                    &element,
                    quadrature,
                    u_local,
                    &mut workspace.integration_workspace,
                )
            },
        );
        Ok(integral[0])
    }
}

impl<'a, T, F, Space, QTable> ElementConnectivityAssembler for ElementIntegralVolumeAssembler<'a, T, F, Space, QTable>
where
    T: RealField,
    // TODO: For some reason this only works if we specify Space::ReferenceDim. However, Space::GeometryDim would be
    // more appropriate, and we anyway have Space::GeometryDim == Space::ReferenceDim by definition of
    // a volumetric finite element space... But unsure if it may cause downstream issues
    F: VolumeFunction<T, Space::ReferenceDim>,
    Space: VolumetricFiniteElementSpace<T>,
    DefaultAllocator:
        TriDimAllocator<T, F::SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
{
    fn solution_dim(&self) -> usize {
        F::SolutionDim::dim()
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

impl<'a, T, F, Space, QTable> ElementScalarAssembler<T> for ElementIntegralVolumeAssembler<'a, T, F, Space, QTable>
where
    T: RealField,
    // TODO: See comment in impl for ElementConnectivityAssembler. Here we should ideally have Space::GeometryDim
    F: VolumeFunction<T, Space::ReferenceDim>,
    Space: VolumetricFiniteElementSpace<T>,
    QTable: QuadratureTable<T, Space::ReferenceDim>,
    DefaultAllocator:
        TriDimAllocator<T, F::SolutionDim, Space::GeometryDim, Space::ReferenceDim> + DimAllocator<T, F::OutputDim>,
{
    fn assemble_element_scalar(&self, element_index: usize) -> eyre::Result<T> {
        let n = self.element_node_count(element_index);
        let s = self.solution_dim();
        let element_ndof = n * s;
        let integral = with_thread_local_workspace(
            &WORKSPACE,
            |workspace: &mut ElementIntegralAssemblerWorkspace<T, Space::ReferenceDim>| {
                workspace
                    .quadrature_buffer
                    .populate_element_weights_and_points_from_table(element_index, self.qtable);
                workspace
                    .local_interpolation_weights
                    .resize_vertically_mut(element_ndof, T::zero());
                workspace.nodes.resize(n, usize::MAX);
                self.populate_element_nodes(&mut workspace.nodes, element_index);
                let u_local = &mut workspace.local_interpolation_weights;
                let quadrature = workspace.quadrature_buffer.weights_and_points();
                gather_global_to_local(&self.u, &mut *u_local, &workspace.nodes, s);
                let element = ElementInSpace::from_space_and_element_index(self.space, element_index);
                integrate_over_volume_element(
                    &self.integrand,
                    &element,
                    quadrature,
                    u_local,
                    &mut workspace.integration_workspace,
                )
            },
        )
        .map_err(|err| match err {
            // TODO: Handle this better? Alternatively we could make the integral "work"
            // since a singular Jacobian also means that the volume form is 0,
            // so the integral vanishes in some sense
            IntegrationFailure::SingularJacobian => {
                eyre!("Failed to compute integral due to singular Jacobian")
            }
        })?;
        Ok(integral[0])
    }
}

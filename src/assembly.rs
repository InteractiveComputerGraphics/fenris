//! Finite element assembly functionality.
//!
//! Assembly in the context of the FEM generally means turning equations into vectors and matrices.
//!
//! `fenris` distinguishes between *local* and *global* assembly. Local assembly refers to the
//! act of assembling (small) vectors and matrices associated with a single finite element,
//! whereas global assembly refers to the machinery that is responsible for adding the local
//! contributions from each element to the global vector or matrix.
//!
//! TODO: Document more
//!
//! # Operators
//!
//! TODO: Explain what "operator" means in this context
//!
//! TODO: Explain notation, and how we need something that is consistent across all dimensions
//! (thereby ruling out using the typical notation used in continuum mechanics)
//!
//! ## Elliptic operators
//!
//! The prototypical elliptic PDE is Laplace's equation
//! $$ - \Delta u = 0, $$
//! whose weak form (assuming homogeneous Dirichlet boundary conditions) reads
//! $$ \int_{\Omega} \nabla u \cdot \nabla v \enspace \mathrm{d} x = 0 \qquad \forall v \in V. $$
//!
//! It turns out that a broad range of PDEs incorporate a term which has some conceptual
//! similarities to the left-hand side of the above weak form. In particular, we introduce the
//! operator $g: \mathbb{R}^{d \times s} \rightarrow \mathbb{R}^{d \times s}$
//! and consider the generalization
//! $$ - \nabla \cdot g (\nabla u) = 0. $$
//! The above equation describes a *vector-valued* PDE in $d$ dimensions. That is, at every point
//! $x \in \Omega \subset \mathbb{R}^d$, we associate the $s$-dimensional vector
//! $u(x) \in \mathbb{R}^s$.
//! With some effort, we can show that the corresponding weak form is
//! $$ \int_{\Omega} g : \nabla v \enspace \mathrm{d} x
//!     - \int_{\partial \Omega} (g^T n) \cdot v \enspace \mathrm{d} x
//!     = 0
//!     \qquad \forall v \in V,$$
//! where $\partial \Omega$ is the boundary of the domain $\Omega \subset \mathbb{R}^d$ and
//! $n : \partial \Omega \rightarrow \mathbb{R}^d$ denotes the outwards-facing normal vector.
//! The utility of this formulation is that we can
//! write code for assembling the vectors and matrices associated with these terms for a very
//! general operator $g$ and re-use it across a wide range of problems simply by replacing
//! $g$ with an appropriate application-specific definition. For example, for $s = 1$ we see
//! that we can recover Laplace's equation by defining $g (\nabla u) = \nabla u$.
//!
//! ## Discrete quantities
//!
//! Discrete matrices and vectors are obtained by substituting $u$ with the finite element
//! interpolation $u_h$ and $v$ with concrete (vector-valued) basis functions.
//! In this context, the FEM interpolation is given by
//! $$ u_h = \sum_I u_I \phi_I, $$
//! where $\phi_I: \Omega \rightarrow \mathbb{R}$ denotes the basis function associated with node
//! $I$ and $u_I \in \mathbb{R}^s$ is its accompanying weight. For $s > 1$, we have here implicitly
//! assumed that we may use the same basis function in each dimension, which while not
//! applicable to all problems, is a reasonable assumption for the kind of problems `fenris`
//! was designed for. The gradient $\nabla u_h$ becomes
//! $$ \nabla u_h = \sum_I \nabla \phi_I \otimes u_I, $$
//! where $\otimes$ denotes the outer product. For $i = 1, \dots, s$ we obtain for node $I$ the
//! $i$th component associated with the vector
//! quantity associated with node $I$ of the weak form. For example, the expression
//! $$ \int_{\Omega} g : \nabla v \enspace \mathrm{d} x $$
//! becomes the global vector $\hat{g}$, whose entry $\hat{g}_{Ii}$ associated with the
//! $i$th component of the $I$th node is given by inserting $u = u_h$ and $v = \phi_I e_i$ where
//! $e_i$ is the unit basis vector $(0, \dots, 1, \dots, 0)$ consisting of zeroes except for the
//! $i$th entry which is one. We obtain:
//!
// For some reason the rendering of the below equation totally fails if I don't use two backslashes
//! $$ \\hat g_{Ii} := \int_{\Omega} g(\nabla u_h) : \nabla (\phi_I e_i) \enspace \mathrm{d} x
//!    = \int_{\Omega} (g_h^T \nabla \phi_I)_i \enspace \mathrm{d} x, $$
//!
//! where we have defined $g_h := g(\nabla u_h)$. More conveniently, we associate with each node
//! $I$ a (small) $s$-dimensional vector $\\hat g_I \in \mathbb{R}^s$:
//!
//! $$ \\hat g_I := \int_{\Omega} g_h^T \\; \nabla \phi_I \enspace \mathrm{d} x. $$
//!
//! Since the above quantity depends on $u_h$, it depends on the nodal values $u_I$. In general,
//! we have that $\\hat g_I = \\hat g_I(u_1, \dots, u_N)$ where $N$ is the number of nodes.
//! We let $\\hat u := (u_1, \dots, u_N)$ denote the concatenation of all nodal solution variables,
//! so that the vector $\\hat g = \\hat g(\\hat u) = (\\hat g_1, \dots, \\hat g_N)$ is dependent on
//! the solution variables associated with each individual node.
//!
//! In order to assemble the global vector $\\hat g$, we see that each vector $\\hat g_I$ is a
//! sum of contributions for each element $K$ on which the basis function $\phi_I$ is supported:
//!
//! $$ \\hat g_I = \sum_K \int_{K} g_h^T \\; \nabla \phi_I \enspace \mathrm{d} x.$$
//!
//! As a result, the global vector $\\hat g$ can also be decomposed into a sum of
//! element-wise contributions. Since only a small number of basis functions are supported on each
//! element, most entries in these element-wise contributions are zero. The non-zero contributions
//! form the \emph{local} vector associated with the element. Specifically, given an element $K$
//! consisting of nodes $[3, 1, 5, 2]$, we define
//! $\\hat g^K = (\\hat g_3, \\hat g_1, \\hat g_5, \\hat g_2)$ to be the local contribution
//! associated with element $K$. The relationship between $\\hat g$ and the local contributions
//! $\\hat g^K$ is then given by the abstract expression
//!
//! $$ \\hat g = \sum_K R_K^T \\hat g^K. $$
//!
//! The operator $R_K: \mathbb{R}^{s N} \rightarrow \mathbb{R}^{s N_K}$ simply "gathers" the
//! local entries associated with an element from a vector of global entries. For example,
//! $R_K \\hat u$ gathers the entries of $\\hat u$ that are associated with the element $K$
//! into a small vector with $s N_K$ entries, where $N_K$ is the number of nodes associated
//! with the element. As a result, the transpose $R_K^T$ *scatters* the local vector into
//! the global vector. Hence, $R_K^T$ scatters the local contributions $\\hat g^K$ into the global
//! vector $\\hat g$.
//!
//! The distinction between local and assembly for vectors can now be made precise:
//!
//! - **Local assembly** is concerned with the construction of $\\hat g^K$ for an element $K$.
//! - **Global assembly** is concerned with scattering the local vector $\\hat g^K$ into the
//!   global vector $\\hat g$.
//!
////!
////!     = \int_{\Omega} (g^T \nabla \phi_I)_i \enspace \mathrm{d} x $$
//!
////! $$ \hat{g}_{Ii} := \int_{\Omega} g(\nabla u_h) : \nabla (\phi_I e_i) \enspace \mathrm{d} x
////!     = \int_{\Omega} (g^T \nabla \phi_I)_i \enspace \mathrm{d} x $$
//!
//!
//! TODO: Explain relationship between general weak form and entries in FE vectors. Then go from
//! there to introduce "elliptic contractions" as a necessary ingredient for assembling
//! the corresponding matrices
//!
//!
//!

pub mod global;
pub mod local;
pub mod operators;

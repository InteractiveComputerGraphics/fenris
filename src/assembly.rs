//! Finite element assembly functionality.
//!
//! Assembly in the context of the FEM generally means turning equations into vectors and matrices.
//!
//! `fenris` distinguishes between *local* and *global* assembly. Local assembly refers to the
//! act of assembling (small) vectors and matrices associated with a single finite element,
//! whereas global assembly refers to the machinery that is responsible for adding the local
//! contributions from each element to the global vector or matrix.
//!
//! TODO: The documentation here should probably be split up and placed in different locations
//!
//! TODO: Here we should probably just write a more concise set of formulas for the
//! local/global assembly distinction (see below) and leave the detailed elliptic stuff for
//! a different module.
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
//! <div>$$ u_h = \sum_I u_I \phi_I, $$</div>
//! where $\phi_I: \Omega \rightarrow \mathbb{R}$ denotes the basis function associated with node
//! $I$ and $u_I \in \mathbb{R}^s$ is its accompanying weight. For $s > 1$, we have here implicitly
//! assumed that we may use the same basis function in each dimension, which while not
//! applicable to all problems, is a reasonable assumption for the kind of problems `fenris`
//! was designed for. The gradient $\nabla u_h$ becomes
//! <div>$$ \nabla u_h = \sum_I \nabla \phi_I \otimes u_I, $$</div>
//! where $\otimes$ denotes the outer product. For $i = 1, \dots, s$ we obtain for node $I$ the
//! $i$th component associated with the vector
//! quantity associated with node $I$ of the weak form. For example, the expression
//! <div>$$ \int_{\Omega} g : \nabla v \enspace \mathrm{d} x $$</div>
//! becomes the global vector $\hat{g}$, whose entry $\hat{g}_{Ii}$ associated with the
//! $i$th component of the $I$th node is given by inserting $u = u_h$ and $v = \phi_I e_i$ where
//! $e_i$ is the unit basis vector $(0, \dots, 1, \dots, 0)$ consisting of zeroes except for the
//! $i$th entry which is one. We obtain:
//!
//! <div>$$ \hat g_{Ii} := \int_{\Omega} g(\nabla u_h) : \nabla (\phi_I e_i) \enspace \mathrm{d} x
//!    = \int_{\Omega} (g_h^T \nabla \phi_I)_i \enspace \mathrm{d} x, $$</div>
//!
//! where we have defined $g_h := g(\nabla u_h)$. More conveniently, we associate with each node
//! $I$ a (small) $s$-dimensional vector $\hat g_I \in \mathbb{R}^s$:
//!
//! <div>$$ \hat g_I := \int_{\Omega} g_h^T \; \nabla \phi_I \enspace \mathrm{d} x. $$</div>
//!
//! Since the above quantity depends on $u_h$, it depends on the nodal values $u_I$. In general,
//! we have that $\hat g_I = \hat g_I(u_1, \dots, u_N)$ where $N$ is the number of nodes.
//! We let $\hat u := (u_1, \dots, u_N)$ denote the concatenation of all nodal solution variables,
//! so that the vector $\hat g = \hat g(\hat u) = (\hat g_1, \dots, \hat g_N)$ is dependent on
//! the solution variables associated with each individual node.
//!
//! In order to assemble the global vector $\hat g$, we see that each vector $\hat g_I$ is a
//! sum of contributions for each element $K$ on which the basis function $\phi_I$ is supported:
//!
//! <div>$$ \hat g_I = \sum_K \int_{K} g_h^T \; \nabla \phi_I \enspace \mathrm{d} x = \sum_K \hat g_I^K.$$</div>
//!
//! As a result, the global vector $\hat g$ can also be decomposed into a sum of
//! element-wise contributions. Since only a small number of basis functions are supported on each
//! element, most entries in these element-wise contributions are zero. The non-zero contributions
//! form the *local* vector associated with the element. Specifically, given an element $K$
//! consisting of nodes $[3, 1, 5, 2]$, we define
//! $\hat g^K = (\hat g_3^K, \hat g_1^K, \hat g_5^K, \hat g_2^K)$ to be the local contribution
//! associated with element $K$. The relationship between $\hat g$ and the local contributions
//! $\hat g^K$ is then given by the abstract expression
//!
//! <div>$$ \hat g = \sum_K R_K^T \hat g^K. $$</div>
//!
//! The operator $R_K: \mathbb{R}^{s N} \rightarrow \mathbb{R}^{s N_K}$ simply "gathers" the
//! local entries associated with an element from a vector of global entries. For example,
//! $R_K \hat u$ gathers the entries of $\hat u$ that are associated with the element $K$
//! into a small vector with $s N_K$ entries, where $N_K$ is the number of nodes associated
//! with the element. As a result, the transpose $R_K^T$ *scatters* the local vector into
//! the global vector. Hence, $R_K^T$ scatters the local contributions $\hat g^K$ into the global
//! vector $\hat g$.
//!
//! The distinction between local and assembly for vectors can now be made precise:
//!
//! - **Local assembly** is concerned with the construction of $\hat g^K$ for an element $K$.
//! - **Global assembly** is concerned with scattering the local vector $\hat g^K$ into the
//!   global vector $\hat g$.
//!
//! ### Discrete matrix quantities
//!
//! Usually we will also need derivatives of (non-linear) vector quantities. For example, when
//! solving a non-linear problem involving $\hat g$ with Newton's method, we need to be able
//! to compute the Jacobian of $\hat g$. We obtain
//!
//! <div>$$ \frac{\partial \hat g}{\partial \hat u}
//!     = \sum_K R_K^T \frac{\partial \hat g^K}{\partial \hat u^K} R_K. $$</div>
//!
//! The quantity $\frac{\partial \hat g^K}{\partial \hat u^K}$ is a $sN_K \times sN_K$ matrix,
//! and in the context of our $g$ operator, it represents the **element stiffness matrix** for
//! element $K$. As before, assembling these local matrix contributions constitute the
//! *local assembly*, whereas storing the result in the global matrix constitutes the
//! *global assembly*.
//!
//! For our general elliptic operator $g$, we next determine an expression for the associated
//! element stiffness matrix. To ease notation, we write $G = \nabla u$ and $G_h = \nabla u_h$.
//! We note that
//!
//! <div>$$ \frac{\partial G_h}{\partial u_J} = \sum_j \nabla \phi_J \otimes e_j \otimes e_j.$$</div>
//!
//! The derivative $\pd{\hat g^K}{\hat u^K}$ can be expressed in terms of
//! the individual $s \times s$ blocks that correspond to individual pairs of basis functions
//! $I$ and $J$. With some labor, we find (using Einstein summation notation)
//!
//! <div>$$ \pd{\hat g^K_I}{u_J^K}
//!     = \int_K \pd{\phi_I}{x_k} \pd{g_{ki}}{G_{mj}} \pd{\phi_J}{x_m} e_i \otimes e_j \dx
//!     = \int_K \mathcal{C}_g(\nabla \phi_I, \nabla \phi_J) \dx, $$</div>
//!
//! where the *contraction operator*
//! $\mathcal{C}_g: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^{s \times s}$
//! defined by
//!
//! <div>$$ \mathcal{C}_{g} (a, b) := a_k \pd{g_{ki}}{G_{mj}} b_m \enspace e_i \otimes e_j $$</div>
//!
//! encodes the derivative information of $g$ independent of FE basis functions. This is a
//! convenient interface, because the full tensor derivative of $g$ may be sparse, so computing
//! all entries might lead to redundant computation, whereas allowing an implementation of an
//! operator $g$ to directly compute the contraction allows it to use a more efficient
//! internal expression for the contraction. By further providing the operator with several vectors
//! to be evaluated at once, shared computations can be amortized.
//!
//! TODO: Document symmetry assumptions (actually we should encode this in our type system)
//!
//! TODO: Extend elliptic operators to have a dependency on the domain, e.g. $g = g(x, \nabla u)$.
//!
//!

pub mod buffers;
pub mod global;
pub mod local;
pub mod operators;

# fenris

![Fenris logo](assets/logo/fenris_logo.svg)

A Rust library for building advanced applications with the Finite Element Method (FEM).

Although developed with a special emphasis on solid mechanics in computer graphics,
Fenris is a highly generic and versatile library applicable to many other domains.

## Status

As of October 2021, Fenris is heavily developed alongside several of our ongoing academic
projects, with the overarching goal of supporting our research efforts.

Our goal is to rework, document and polish the library for external users during 2022.
In its current state, Fenris is **not recommended for general usage**. We currently
offer **no API stability** whatsoever, the documentation is severely lacking,
and only parts of the library has been extensively tested. Moreover, some parts of the
library will likely be removed altogether in future versions.

## Goals

With Fenris, we aim to provide an **open-source**, **composable**, **flexible** and **performant** library
for advanced finite element computations in Rust.

Fenris is intended primarily as an alternative to C++ FEM libraries. With Fenris, users can
take advantage of the significant productivity boost afforded by working with Rust, a modern programming
language with a best-in-class dependency management system and a revolutionary model for
memory-safe, high-performance system programming.

This statement is motivated by our own experiences writing FEM code in Rust: gone are the hours upon hours
of wrestling with CMake to integrate an external library. Furthermore, the expressive type system and borrow checker model
employed by Rust encourages good designs that are free of the myriads of footguns associated with (even modern) C++.
And perhaps even more important, the excellent generic trait system found in Rust - which importantly is type-checked
at compile time - allows us to write highly generic code with explicit invariants encoded in the type system
that are automatically and correctly represented in the generated documentation.

In short, we have found that Rust allows us to spend more time on solving interesting and complex problems (the fun part),
and less time on dealing with auxiliary issues largely caused by language deficiencies (the annoying part).

### Summary of technical goals

- Provide a number of low-order and high-order standard Lagrange elements of a variety of geometric shapes in 2D and 3D
  (at the very least triangles, quadrilaterals, tetrahedra and hexahedra).
- High-performance shared-memory parallelism for assembly.
- A composable architecture: Higher-level functionality is built by the composition of lower-level functionality.
  Users choose the level at which they need to work at. 
- Facilitates generic programming: write code once and have it work across a number of different elements,
  dimensions and operators.
- Convent I/O, currently in the form of export to VTK/VTU for visualization in ParaView.

### Non-goals

- Fenris is not intended to compete with the likes of [FEniCS](https://fenicsproject.org/) or similar libraries that
  let users provide high-level weak formulations of PDEs. In contrast, Fenris targets users who need lower-level
  functionality, and is perhaps more comparable to the likes of [deal.II](https://www.dealii.org/).
- Fenris by itself provides no functionality for solving (non-)linear systems, only the functionality
  for assembling (as scalars/vectors/matrices) and applying discrete operators.
- We have no plans for supporting distributed computing or GPU acceleration at this time.

## Publications

An older incarnation of Fenris was used for the code associated with the following academic papers:

- Longva, A., Löschner, F., Kugelstadt, T., Fernández-Fernández, J. A., & Bender, J. (2020).
  *Higher-order finite elements for embedded simulation*. 
  ACM Transactions on Graphics (TOG), 39(6), 1-14.
- Löschner, F., Longva, A., Jeske, S., Kugelstadt, T., & Bender, J. (2020).  
  *Higher‐Order Time Integration for Deformable Solids*.
  In Computer Graphics Forum (Vol. 39, No. 8, pp. 157-169).

## Contribution

Apart from minor bug fixes or changes, we do not accept source code contributions at this time.
We would however be happy for daring users who want to try out our library to report issues
on our issue tracker.

We have a number of unrealized plans that will require significant reworking of
parts of the library. Once the library is in a more stable state we will be grateful
for contributions from the community.

## License

Fenris is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See LICENSE-APACHE and LICENSE-MIT for details.
Opening a pull requests is assumed to signal agreement with these licensing terms.



use fenris::allocators::DimAllocator;
use fenris::assembly::local::{Density, SourceFunction};
use fenris::assembly::operators::Operator;
use fenris::nalgebra::{DefaultAllocator, OPoint, OVector, RealField, Scalar};
use fenris::SmallDim;

/// A source for the gravitational force.
///
/// This source implements the force density
/// <div>$$
/// \rho \vec g,
/// $$</div>
/// where $\rho: \Omega \rightarrow \mathbb{R}_{\geq 0}$ is a density function and $\vec g \in \mathbb{R}^d$
/// is the $d$-dimensional graviational acceleration vector.
///
/// In conjunction with [`ElementSourceAssembler`](fenris::assembly::local::ElementSourceAssembler),
/// the source corresponds to the weak form term
/// <div>$$
///  \int_\Omega \rho \vec g \, : \, \vec w \, \d{\vec X},
/// $$</div>
/// where $\vec w: \Omega \rightarrow \mathbb{R}^d$ is a test function.
#[derive(Debug, Clone)]
pub struct GravitySource<T, D>
where
    T: Scalar,
    D: SmallDim,
    DefaultAllocator: DimAllocator<T, D>,
{
    gravitational_acceleration: OVector<T, D>,
}

impl<T, D> GravitySource<T, D>
where
    T: Scalar,
    D: SmallDim,
    DefaultAllocator: DimAllocator<T, D>,
{
    pub fn from_acceleration(gravitational_acceleration: OVector<T, D>) -> Self {
        Self {
            gravitational_acceleration,
        }
    }
}

impl<T, D> Operator<T, D> for GravitySource<T, D>
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: DimAllocator<T, D>,
{
    /// Solution dimension is the same as the spatial dimension.
    type SolutionDim = D;
    /// Density at the quadrature point.
    type Parameters = Density<T>;
}

impl<T, D> SourceFunction<T, D> for GravitySource<T, D>
where
    T: RealField,
    D: SmallDim,
    DefaultAllocator: DimAllocator<T, D>,
{
    fn evaluate(&self, _coords: &OPoint<T, D>, Density(density): &Self::Parameters) -> OVector<T, Self::SolutionDim> {
        &self.gravitational_acceleration * density.clone()
    }
}

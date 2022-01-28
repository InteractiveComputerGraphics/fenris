use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, Scalar, U2, U3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize, OPoint<T, D>: Serialize",
    deserialize = "T: Deserialize<'de>, OPoint<T, D>: Deserialize<'de>"
))]
pub struct Hyperball<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    center: OPoint<T, D>,
    radius: T,
}

impl<T, D> Hyperball<T, D>
where
    T: Scalar,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn from_center_and_radius(center: OPoint<T, D>, radius: T) -> Self {
        Self { center, radius }
    }

    pub fn center(&self) -> &OPoint<T, D> {
        &self.center
    }

    pub fn radius(&self) -> T {
        self.radius.clone()
    }
}

pub type Disk<T> = Hyperball<T, U2>;
pub type Ball<T> = Hyperball<T, U3>;

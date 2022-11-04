use fenris_geometry::AxisAlignedBoundingBox;
use nalgebra::{DefaultAllocator, DimName, OPoint, point, U2};
use nalgebra::allocator::Allocator;
use fenris::allocators::DimAllocator;

#[test]
fn aabb_intersects_2d() {
    type Aabb = AxisAlignedBoundingBox<f64, U2>;

    let aabb1 = Aabb::new(point![1.0, 1.0], point![4.0, 3.0]);

    macro_rules! assert_no_intersection {
        ($aabb2:expr) => {
            assert!(!aabb1.intersects(&$aabb2));
            // Check that we get the same result when reversing the order
            assert!(!$aabb2.intersects(&aabb1));
        };
    }

    macro_rules! assert_intersection {
        ($aabb2:expr) => {
            assert!(aabb1.intersects(&$aabb2));
            // Check that we get the same result when reversing the order
            assert!($aabb2.intersects(&aabb1));
        };
    }

    assert_no_intersection!(Aabb::new(point![6.0, 4.0], point![9.0, 6.0]));
    assert_no_intersection!(Aabb::new(point![5.0, 2.0], point![8.0, 4.0]));
    assert_no_intersection!(Aabb::new(point![5.0, 1.5], point![8.0, 2.5]));
    assert_no_intersection!(Aabb::new(point![5.0, -1.0], point![7.0, 0.0]));
    assert_no_intersection!(Aabb::new(point![1.5, -1.0], point![3.5, 0.5]));
    assert_no_intersection!(Aabb::new(point![-3.0, -2.0], point![0.0, 0.0]));
    assert_no_intersection!(Aabb::new(point![-3.0, 1.5], point![0.0, 2.5]));
    assert_no_intersection!(Aabb::new(point![-3.0, 2.5], point![0.0, 3.5]));
    assert_no_intersection!(Aabb::new(point![-3.0, 3.5], point![0.0, 4.5]));
    assert_no_intersection!(Aabb::new(point![1.5, 3.5], point![3.5, 4.5]));

    assert_intersection!(Aabb::new(point![1.5, 1.5], point![3.5, 2.5]));
    assert_intersection!(Aabb::new(point![1.5, 1.5], point![3.5, 3.5]));
    assert_intersection!(Aabb::new(point![1.5, 1.5], point![4.5, 3.5]));
    assert_intersection!(Aabb::new(point![0.0, 0.0], point![2.0, 2.0]));
    assert_intersection!(Aabb::new(point![0.0, 0.0], point![5.0, 4.0]));
}

/// Helper function for collecting corners from corner iterator.
fn corners<D: DimName>(aabb:& AxisAlignedBoundingBox<f64, D>) -> Vec<OPoint<f64, D>>
where
    DefaultAllocator: DimAllocator<f64, D>
{
    aabb.corners_iter().collect()
}

fn compare_unordered_points<D: DimName>(points1: &[OPoint<f64, D>], points2: &[OPoint<f64, D>]) -> bool
where
    DefaultAllocator: Allocator<f64, D>
{
    assert_eq!(points1.len(), points2.len());
    // This is O(n^2) but is acceptable for use in tests here
    points1.iter()
        .all(|p1| points2.contains(p1))
}

macro_rules! assert_unordered_eq {
    ($p1:expr, $p2:expr) => {
        {
            let eq = compare_unordered_points((&$p1).as_ref(), (&$p2).as_ref());
            if !eq {
                dbg!(&$p1, &$p2);
                assert!(eq, "Point lists do not contain the same points");
            }
        }
    }
}


#[test]
fn test_aabb_corners_iter() {

    // 1D
    {
        let aabb = AxisAlignedBoundingBox::new(point![3.0], point![4.0]);
        assert_eq!(corners(&aabb), vec![ point![3.0 ], point![4.0] ]);
    }

    // 2D
    {
        let aabb = AxisAlignedBoundingBox::new(point![3.0, 4.0], point![5.0, 6.0]);
        let expected = [
            [3.0, 4.0],
            [3.0, 6.0],
            [5.0, 4.0],
            [5.0, 6.0]
        ].map(OPoint::from);
        assert_unordered_eq!(corners(&aabb), &expected);
    }

    // 3D
    {
        let aabb = AxisAlignedBoundingBox::new(point![1.0, 2.0, 3.0], point![4.0, 5.0, 6.0]);
        let expected = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 6.0],
            [1.0, 5.0, 3.0],
            [1.0, 5.0, 6.0],
            [4.0, 2.0, 3.0],
            [4.0, 2.0, 6.0],
            [4.0, 5.0, 3.0],
            [4.0, 5.0, 6.0]
        ].map(OPoint::from);
        assert_unordered_eq!(corners(&aabb), expected);
    }


}
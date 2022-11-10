use fenris::allocators::DimAllocator;
use fenris_geometry::proptest::{aabb2, aabb3, point2, point3};
use fenris_geometry::{AxisAlignedBoundingBox, AxisAlignedBoundingBox2d, AxisAlignedBoundingBox3d};
use matrixcompare::assert_scalar_eq;
use nalgebra::allocator::Allocator;
use nalgebra::proptest::vector;
use nalgebra::{distance, distance_squared, Const, Point};
use nalgebra::{point, DefaultAllocator, DimName, OPoint, U2};
use proptest::collection::vec;
use proptest::prelude::*;

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
fn corners<D: DimName>(aabb: &AxisAlignedBoundingBox<f64, D>) -> Vec<OPoint<f64, D>>
where
    DefaultAllocator: DimAllocator<f64, D>,
{
    aabb.corners_iter().collect()
}

fn compare_unordered_points<D: DimName>(points1: &[OPoint<f64, D>], points2: &[OPoint<f64, D>]) -> bool
where
    DefaultAllocator: Allocator<f64, D>,
{
    assert_eq!(points1.len(), points2.len());
    // This is O(n^2) but is acceptable for use in tests here
    points1.iter().all(|p1| points2.contains(p1))
}

macro_rules! assert_unordered_eq {
    ($p1:expr, $p2:expr) => {{
        let eq = compare_unordered_points((&$p1).as_ref(), (&$p2).as_ref());
        if !eq {
            dbg!(&$p1, &$p2);
            assert!(eq, "Point lists do not contain the same points");
        }
    }};
}

fn point_in_aabb<const D: usize>(aabb: AxisAlignedBoundingBox<f64, Const<D>>) -> impl Strategy<Value = Point<f64, D>> {
    // Bias generation 0.0 and 1.0 values to ensure that we generate values also on the boundary
    // of the box
    let values = prop_oneof![
        1 => Just(0.0),
        1 => Just(1.0),
        5 => 0.0 ..= 1.0];
    vector(values, Const::<D>)
        .prop_map(move |v| {
            // Coordinates are in [0, 1] interval, transform to [a_i, b_i]
            v.zip_zip_map(&aabb.min().coords, &aabb.max().coords, |p, a, b| (1.0 - p) * a + p * b)
        })
        .prop_map(Point::from)
}

#[test]
fn test_aabb_corners_iter() {
    // 1D
    {
        let aabb = AxisAlignedBoundingBox::new(point![3.0], point![4.0]);
        assert_eq!(corners(&aabb), vec![point![3.0], point![4.0]]);
    }

    // 2D
    {
        let aabb = AxisAlignedBoundingBox::new(point![3.0, 4.0], point![5.0, 6.0]);
        let expected = [[3.0, 4.0], [3.0, 6.0], [5.0, 4.0], [5.0, 6.0]].map(OPoint::from);
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
            [4.0, 5.0, 6.0],
        ]
        .map(OPoint::from);
        assert_unordered_eq!(corners(&aabb), expected);
    }
}

#[test]
fn test_furthest_point_2d() {
    let aabb = AxisAlignedBoundingBox::new(point![1.0, 1.0], point![2.0, 3.0]);

    {
        let p = point![0.0, 0.0];
        let q = aabb.furthest_point_to(&p);
        assert_eq!(q, point![2.0, 3.0]);
        // We check all the convenience method results here just to have a unit test that does
        // this, but the consistency is separately tested by proptests, which is why
        // we don't do this for every test
        let dist2: f64 = distance_squared(&q, &p);
        assert_scalar_eq!(dist2, 13.0);
        assert_scalar_eq!(dist2, aabb.max_dist2_to(&p));
        assert_scalar_eq!(aabb.max_dist_to(&p), f64::sqrt(13.0));
    }

    {
        let p = point![1.5, 2.0];
        let q = aabb.furthest_point_to(&p);
        // The exact point is not unique: any corner will be applicable
        assert_scalar_eq!(distance_squared(&q, &p), 1.25);
    }
}

#[test]
fn test_closest_point() {
    // Helper macro for succinct checks
    macro_rules! assert_closest_point {
        ($aabb:expr, $p:expr => $expected:expr) => {{
            let aabb = $aabb;
            let p = $p;
            let q = aabb.closest_point_to(&p);
            assert_eq!(&q, &$expected);
            assert_eq!(distance(&q, &p), aabb.dist_to(&p));
            assert_eq!(distance_squared(&q, &p), aabb.dist2_to(&p));
        }};
    }

    // 2D
    {
        let a = point![2.0, 3.0];
        let b = point![3.0, 5.0];
        let aabb = AxisAlignedBoundingBox2d::new(a, b);
        // Outside points
        assert_closest_point!(aabb, point![1.0, 1.0] => point![2.0, 3.0]);
        assert_closest_point!(aabb, point![2.0, 2.0] => point![2.0, 3.0]);
        assert_closest_point!(aabb, point![1.0, 4.0] => point![2.0, 4.0]);
        assert_closest_point!(aabb, point![1.0, 5.0] => point![2.0, 5.0]);
        assert_closest_point!(aabb, point![-1.0, 6.0] => point![2.0, 5.0]);
        assert_closest_point!(aabb, point![2.5, 7.0] => point![2.5, 5.0]);
        assert_closest_point!(aabb, point![4.0, 6.0] => point![3.0, 5.0]);
        assert_closest_point!(aabb, point![6.0, 4.0] => point![3.0, 4.0]);
        assert_closest_point!(aabb, point![5.0, 2.0] => point![3.0, 3.0]);

        // Inside points
        assert_closest_point!(aabb, point![2.5, 4.0] => point![2.5, 4.0]);
        assert_closest_point!(aabb, point![2.3, 4.6] => point![2.3, 4.6]);
    }

    // 3D. We only test a few points since the impl is the same as in 2D and
    // we have proptests that should cover things quite extensively
    {
        let a = point![2.0, 3.0, 1.0];
        let b = point![3.0, 5.0, 6.0];
        let aabb = AxisAlignedBoundingBox3d::new(a, b);
        // Outside points
        assert_closest_point!(aabb, point![1.0, 1.0, 1.0] => point![2.0, 3.0, 1.0]);
        assert_closest_point!(aabb, point![4.0, 6.0, 8.0] => point![3.0, 5.0, 6.0]);
        assert_closest_point!(aabb, point![1.0, 4.0, 5.0] => point![2.0, 4.0, 5.0]);

        // Inside points
        assert_closest_point!(aabb, point![2.5, 4.0, 3.0] => point![2.5, 4.0, 3.0]);
    }
}

proptest! {

    #[test]
    fn aabb_max_dists_agree_with_furthest_point_2d(point in point2(), aabb in aabb2()) {
        let q = aabb.furthest_point_to(&point);
        let dist2 = distance_squared(&q, &point);
        prop_assert_eq!(aabb.max_dist2_to(&point), dist2);
        prop_assert_eq!(aabb.max_dist_to(&point), dist2.sqrt());
    }

    #[test]
    fn aabb_max_dists_agree_with_furthest_point_3d(point in point3(), aabb in aabb3()) {
        let q = aabb.furthest_point_to(&point);
        let dist2 = distance_squared(&q, &point);
        prop_assert_eq!(aabb.max_dist2_to(&point), dist2);
        prop_assert_eq!(aabb.max_dist_to(&point), dist2.sqrt());
    }

    #[test]
    fn aabb_furthest_point_2d(p in point2(), aabb in aabb2()) {
        // The furthest point in the AABB is *always* a corner, so we must satisfy
        //  dist(p, q) <= dist(p, c)
        // for all corners c and furthest point q
        let q = aabb.furthest_point_to(&p);
        let further_away_than_all_corners = aabb.corners_iter()
            .all(|corner| distance_squared(&p, &q) >= distance_squared(&p, &corner));
        prop_assert!(further_away_than_all_corners);

        // The result should be exactly one of the corners, and since there are no floating
        // point operations applied to the result (all numbers are just copied),
        // there should also be no round-off error in the result, so we should
        // be safe to check if the point is contained in the AABB, despite the fact that it
        // resides exactly on the boundary!
        prop_assert!(aabb.contains_point(&q));
    }

    #[test]
    fn aabb_furthest_point_3d(p in point3(), aabb in aabb3()) {
        // The furthest point in the AABB is *always* a corner, so we must satisfy
        //  dist(p, q) <= dist(p, c)
        // for all corners c and furthest point q
        let q = aabb.furthest_point_to(&p);
        let further_away_than_all_corners = aabb.corners_iter()
            .all(|corner| distance_squared(&p, &q) >= distance_squared(&p, &corner));
        prop_assert!(further_away_than_all_corners);

        // The result should be exactly one of the corners, and since there are no floating
        // point operations applied to the result (all numbers are just copied),
        // there should also be no round-off error in the result, so we should
        // be safe to check if the point is contained in the AABB, despite the fact that it
        // resides exactly on the boundary!
        prop_assert!(aabb.contains_point(&q));
    }

    #[test]
    fn aabb_dists_agree_with_closest_point_2d(point in point2(), aabb in aabb2()) {
        let q = aabb.closest_point_to(&point);
        let dist2 = distance_squared(&q, &point);
        prop_assert_eq!(aabb.dist2_to(&point), dist2);
        prop_assert_eq!(aabb.dist_to(&point), dist2.sqrt());
    }

    #[test]
    fn aabb_dists_agree_with_closest_point_3d(point in point3(), aabb in aabb3()) {
        let q = aabb.closest_point_to(&point);
        let dist2 = distance_squared(&q, &point);
        prop_assert_eq!(aabb.dist2_to(&point), dist2);
        prop_assert_eq!(aabb.dist_to(&point), dist2.sqrt());
    }

    #[test]
    fn aabb_closest_point_2d_closer_than_other_points(
        p in point2(),
        (aabb, test_points) in aabb2()
            .prop_flat_map(|aabb| (Just(aabb), vec(point_in_aabb(aabb), 0 .. 50)))
    ) {
        let q = aabb.closest_point_to(&p);
        prop_assert!(aabb.contains_point(&q));
        for test_point in test_points {
            assert!(aabb.contains_point(&test_point));
            prop_assert!(distance(&q, &p) <= distance(&p, &test_point));
        }
    }

    #[test]
    fn aabb_closest_point_3d_closer_than_other_points(
        p in point3(),
        (aabb, test_points) in aabb3()
            .prop_flat_map(|aabb| (Just(aabb), vec(point_in_aabb(aabb), 0 .. 50)))
    ) {
        let q = aabb.closest_point_to(&p);
        prop_assert!(aabb.contains_point(&q));
        for test_point in test_points {
            assert!(aabb.contains_point(&test_point));
            prop_assert!(distance(&q, &p) <= distance(&p, &test_point));
        }
    }

    #[test]
    fn aabb_closest_point_of_internal_point_2d(
        (aabb, p) in aabb2().prop_flat_map(|aabb| (Just(aabb), point_in_aabb(aabb)))
    ) {
        prop_assert_eq!(aabb.closest_point_to(&p), p);
    }

    #[test]
    fn aabb_closest_point_of_internal_point_3d(
        (aabb, p) in aabb3().prop_flat_map(|aabb| (Just(aabb), point_in_aabb(aabb)))
    ) {
        prop_assert_eq!(aabb.closest_point_to(&p), p);
    }
}

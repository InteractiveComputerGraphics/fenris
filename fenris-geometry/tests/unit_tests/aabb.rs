use fenris_geometry::AxisAlignedBoundingBox;
use nalgebra::{vector, U2};

#[test]
fn aabb_intersects_2d() {
    type Aabb = AxisAlignedBoundingBox<f64, U2>;

    let aabb1 = Aabb::new(vector![1.0, 1.0], vector![4.0, 3.0]);

    macro_rules! assert_no_intersection {
        ($aabb2:expr) => {
            assert!(!aabb1.intersects(&$aabb2));
            // Check that we get the same result when reversing the order
            assert!(!$aabb2.intersects(&aabb1));
        }
    }

    macro_rules! assert_intersection {
        ($aabb2:expr) => {
            assert!(aabb1.intersects(&$aabb2));
            // Check that we get the same result when reversing the order
            assert!($aabb2.intersects(&aabb1));
        }
    }

    assert_no_intersection!(Aabb::new(vector![6.0, 4.0], vector![9.0, 6.0]));
    assert_no_intersection!(Aabb::new(vector![5.0, 2.0], vector![8.0, 4.0]));
    assert_no_intersection!(Aabb::new(vector![5.0, 1.5], vector![8.0, 2.5]));
    assert_no_intersection!(Aabb::new(vector![5.0, -1.0], vector![7.0, 0.0]));
    assert_no_intersection!(Aabb::new(vector![1.5, -1.0], vector![3.5, 0.5]));
    assert_no_intersection!(Aabb::new(vector![-3.0, -2.0], vector![0.0, 0.0]));
    assert_no_intersection!(Aabb::new(vector![-3.0, 1.5], vector![0.0, 2.5]));
    assert_no_intersection!(Aabb::new(vector![-3.0, 2.5], vector![0.0, 3.5]));
    assert_no_intersection!(Aabb::new(vector![-3.0, 3.5], vector![0.0, 4.5]));
    assert_no_intersection!(Aabb::new(vector![1.5, 3.5], vector![3.5, 4.5]));

    assert_intersection!(Aabb::new(vector![1.5, 1.5], vector![3.5, 2.5]));
    assert_intersection!(Aabb::new(vector![1.5, 1.5], vector![3.5, 3.5]));
    assert_intersection!(Aabb::new(vector![1.5, 1.5], vector![4.5, 3.5]));
    assert_intersection!(Aabb::new(vector![0.0, 0.0], vector![2.0, 2.0]));
    assert_intersection!(Aabb::new(vector![0.0, 0.0], vector![5.0, 4.0]));
}
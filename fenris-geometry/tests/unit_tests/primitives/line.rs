use fenris_geometry::assert_line_segments_approx_equal;
use fenris_geometry::{Disk, LineSegment2d};
use nalgebra::point;

#[test]
fn test_line_segment_disk_intersection() {
    let a = point![1.0, 2.0];
    let b = point![4.0, 1.0];
    let c = point![2.0, 3.0];
    let segment = LineSegment2d::new(a, b);

    // Radius r = 1, empty intersection
    {
        let disk = Disk::from_center_and_radius(c, 1.0);
        assert!(segment.intersect_disk(&disk).is_none());
    }

    // Radius r = 2, cuts the segment in half
    {
        let disk = Disk::from_center_and_radius(c, 2.0);
        let intersection_point = point![3.069693845669907, 1.310102051443365];
        let expected_intersection = LineSegment2d::new(a, intersection_point);
        let intersection = segment.intersect_disk(&disk).unwrap();
        assert_line_segments_approx_equal!(intersection, expected_intersection, abstol = 1e-14);
    }

    // Radius r = 3, completely contains the line segment
    {
        let disk = Disk::from_center_and_radius(c, 3.0);
        let intersection = segment.intersect_disk(&disk).unwrap();
        assert_line_segments_approx_equal!(intersection, segment, abstol = 1e-14);
    }

    // Test when the disk is on the other side of the line segment
    {
        let c2 = point![2.0, 1.0];
        let disk = Disk::from_center_and_radius(c2, 1.0);
        let expected_intersection = LineSegment2d::new(
            point![1.465153077165047, 1.844948974278318],
            point![2.934846922834954, 1.355051025721682],
        );
        let intersection = segment.intersect_disk(&disk).unwrap();
        assert_line_segments_approx_equal!(intersection, expected_intersection, abstol = 1e-14);
    }
}

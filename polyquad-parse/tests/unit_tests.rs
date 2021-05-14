use polyquad_parse::{parse2d, parse3d};

#[test]
fn basic_test_2d() {
    let rule = r"
        0.3 0.4 0.6
        0.5 0.1 0.2
    ";

    let rule = parse2d(&rule).unwrap();
    assert_eq!(rule.weights, vec![0.6, 0.2]);
    assert_eq!(rule.points, vec![[0.3, 0.4], [0.5, 0.1]]);
}

#[test]
fn basic_test_3d() {
    let rule = r"
        0.3  0.4 -0.3  0.6
        0.5 -0.1  0.2 -0.8
    ";

    let rule = parse3d(&rule).unwrap();
    assert_eq!(rule.weights, vec![0.6, -0.8]);
    assert_eq!(rule.points, vec![[0.3, 0.4, -0.3], [0.5, -0.1, 0.2]]);
}

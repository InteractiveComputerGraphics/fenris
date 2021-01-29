use nested_vec::NestedVec;

#[test]
fn begin_array() {
    let mut storage = NestedVec::<isize>::new();

    {
        // Add empty array
        storage.begin_array();
    }

    assert_eq!(storage.len(), 1);
    assert_eq!(storage.get(0).unwrap(), []);
    assert!(storage.get(1).is_none());
    assert_eq!(
        Vec::<isize>::new(),
        storage.iter().flatten().cloned().collect::<Vec<_>>()
    );

    {
        storage.begin_array().push_single(5).push_single(9);
    }

    assert_eq!(storage.len(), 2);
    assert_eq!(storage.get(0).unwrap(), []);
    assert_eq!(storage.get(1).unwrap(), [5, 9]);
    assert!(storage.get(2).is_none());

    {
        // Add empty array
        storage.begin_array();
    }

    assert_eq!(storage.len(), 3);
    assert_eq!(storage.get(0).unwrap(), []);
    assert_eq!(storage.get(1).unwrap(), [5, 9]);
    assert_eq!(storage.get(2).unwrap(), []);
    assert!(storage.get(3).is_none());

    {
        storage.begin_array().push_single(3);
    }

    assert_eq!(storage.len(), 4);
    assert_eq!(storage.get(0).unwrap(), []);
    assert_eq!(storage.get(1).unwrap(), [5, 9]);
    assert_eq!(storage.get(2).unwrap(), []);
    assert_eq!(storage.get(3).unwrap(), [3]);
    assert!(storage.get(4).is_none());
}

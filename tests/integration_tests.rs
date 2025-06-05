use morphnet_gtl::prelude::*;
use morphnet_gtl::VERSION;

#[test]
fn test_version() {
    let _net = MorphNet::new();
    assert!(!VERSION.is_empty());
}

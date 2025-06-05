use morphnet_gtl::prelude::*;
use morphnet_gtl::VERSION;

#[test]
fn test_version() {
    assert!(!VERSION.is_empty());
}

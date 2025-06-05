use morphnet_gtl::prelude::*;
use morphnet_gtl::VERSION;
use morphnet_gtl::morphnet::point_distance;
use nalgebra::Point2;

fn main() {
    let net = MorphNet::new();
    // Demonstrate use of some helper utilities
    let d = point_distance(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0));
    println!("Distance between points: {:.2}", d);
    println!("Net has {} templates", net.body_plan.templates.len());
    println!("MorphNet version: {}", VERSION);
}

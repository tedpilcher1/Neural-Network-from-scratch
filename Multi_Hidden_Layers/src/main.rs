use crate::init_model::build_model;

mod init_model;

fn main() {

    let model = build_model(2, 2, vec![10, 10, 5, 2]);

    println!("{:?}", model.weights);
}

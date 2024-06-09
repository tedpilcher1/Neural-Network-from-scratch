use crate::init_model::build_model;

mod init_model;
mod forward_propagation;
mod matrix_functions;
mod backpropagation;

fn main() {

    let model = build_model(2, 2, vec![5]);
}

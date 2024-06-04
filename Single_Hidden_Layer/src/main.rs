mod init_model;
mod forward_propagation;
mod dot_product;

use crate::init_model::build_model;
use crate::forward_propagation::apply_forward_propagation; 

fn main(){


    // need to define dataset to train on

    let mut model = build_model(5);

}
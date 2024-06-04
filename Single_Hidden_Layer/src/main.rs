mod init_model;
mod forward_propagation;
mod matrix_functions;
mod backward_propagation;

use crate::init_model::build_model;
use crate::forward_propagation::apply_forward_propagation; 

fn test_run(){

    // for testing
    let mut x: Vec<Vec<f64>> = vec![
        vec![0.67326617, -0.17303301], 
        vec![-0.21095606, 0.35734355], 
        vec![2.17465138, 0.16337821]
    ];

    let mut y = vec![0, 1, 1];

    let mut model = build_model(2, 5, 2);
    let forward_output = apply_forward_propagation(&mut model, &mut x);
}

fn main(){
    
    test_run();
}
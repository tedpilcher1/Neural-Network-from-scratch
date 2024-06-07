mod init_model;
mod forward_propagation;
mod matrix_functions;
mod backward_propagation;
mod gradient_descent;

use crate::init_model::build_model;
use crate::forward_propagation::predict;
use crate::gradient_descent::apply_gradient_descent;

fn test_run(){

    let NUM_PASSES = 20000;

    // for testing
    let x: Vec<Vec<f64>> = vec![
        vec![0.67326617, -0.17303301], 
        vec![-0.21095606, 0.35734355], 
        vec![2.17465138, 0.16337821]
    ];

    let y:Vec<i32> = vec![0, 0, 0];

    let mut model = build_model(2, 5, 2);
    apply_gradient_descent(&mut model, &x, &y, NUM_PASSES, false);

    let x2: Vec<Vec<f64>> = vec![
        vec![0.67326617, -0.17303301],
    ];

    let prediction: i32 = predict(&mut model, &x2);

    println!("{}", prediction);
}

fn main(){
    
    test_run();
}
use crate::backward_propagation::{apply_backward_propagation, GradientWeightsBiases};
use crate::forward_propagation::apply_forward_propagation;
use crate::init_model::NN;

fn update_params(model : &mut NN, gradients: &mut GradientWeightsBiases){

    // TODO for each value in each weight and bias, update by adding: -epsilon * gradient, use f64::EPSILON
}

fn apply_regularisation_terms(model : &mut NN, gradients: &mut GradientWeightsBiases){

    // TODO add regularisation term to both weight gradients using: reg_lambda * weight
}

fn apply_gradient_descent (model : &mut NN, x : &mut Vec<Vec<f64>>, y : &mut Vec<f64>, num_passes: i32, print_loss: bool) {

    let mut gradients = GradientWeightsBiases {

        dW1: Vec::new(),
        dW2: Vec::new(),
        db1: Vec::new(),
        db2: Vec::new(),
    };

    for i in 0..num_passes{

        // forward propagation
        let forward_output = apply_forward_propagation(model, x);

        // backpropagation
        apply_backward_propagation(model, &mut gradients, &forward_output, x, y);

        // add regularisation terms
        apply_regularisation_terms(model, &mut gradients);

        // update params
        update_params(model, &mut gradients);

        // optionally print the loss
        if (print_loss) {

            // TODO
            println!("Loss after iteration {}: {}", i, i);
        }
    }
}
use crate::backward_propagation::{apply_backward_propagation, GradientWeightsBiases};
use crate::forward_propagation::apply_forward_propagation;
use crate::init_model::NN;

fn update_param(param : &mut Vec<Vec<f64>>, gradient : &Vec<Vec<f64>>) {

    for i in 0..param.len(){

        for j in 0..param[0].len(){

            param[i][j] -= f64::EPSILON * gradient[i][j];
        }
    }
}

fn update_params(model : &mut NN, gradients: &mut GradientWeightsBiases){

    // for each value in each weight and bias, update by adding: -epsilon * gradient
    update_param(&mut model.weights_1, &gradients.dW1);
    update_param(&mut model.weights_2, &gradients.dW2);
    update_param(&mut model.bias_1, &gradients.db1);
    update_param(&mut model.bias_2, &gradients.db2);
}

fn apply_regularisation_term(param : &Vec<Vec<f64>>, gradient: &mut Vec<Vec<f64>>) {

    let reg_lambda:f64 = 0.01;

    for i in 0..param.len(){

        for j in 0..param[0].len(){

            gradient[i][j] += reg_lambda * param[i][j];
        }
    }
}

fn apply_regularisation_terms(model : &mut NN, gradients: &mut GradientWeightsBiases){

    // add regularisation term to both weight gradients using: reg_lambda * weight
    apply_regularisation_term(&model.weights_1, &mut gradients.dW1);
    apply_regularisation_term(&model.weights_2, &mut gradients.dW2);

}

pub fn apply_gradient_descent (model : &mut NN, x : &Vec<Vec<f64>>, y : &Vec<i32>, num_passes: i32, print_loss: bool) {

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
        if print_loss {

            // TODO
            println!("Loss after iteration {}: {}", i, i);
        }
    }
}
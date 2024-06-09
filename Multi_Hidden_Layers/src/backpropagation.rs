use crate::forward_propagation::ForwardOutput;
use crate::init_model::NeuralNetwork;
use crate::matrix_functions::{apply_dot_product, transpose};

pub struct GradientsWeightsBiases {

    pub d_W: Vec<Vec<Vec<f64>>>,
    pub d_b: Vec<Vec<Vec<f64>>>,
}

fn calc_output_error(probs : &Vec<Vec<f64>>, y : &Vec<i32>) -> Vec<Vec<f64>> {

    let mut error: Vec<Vec<f64>> = probs.clone();

    // subtract 1 from prob of true class (1) for each example
    for i in 0..error.len() {

        let j: usize = y[i].to_usize().unwrap();
        error[i][j] -= 1.0;
    }

    return error;
}

fn sum(delta : &Vec<Vec<f64>>) -> Vec<Vec<f64>>{

    let mut z: Vec<Vec<f64>> = vec![vec![0.0; delta[0].len()]; 1];

    for column in 0..delta[0].len(){

        for row in 0..delta.len(){

            z[0][column] += delta[row][column];
        }
    }

    return z;
}

pub fn apply_backward_propagation(model : &mut NeuralNetwork, gradients : &mut GradientsWeightsBiases, forward_output: &ForwardOutput, learning_rate : f64, x : &Vec<Vec<f64>>, y : &Vec<i32>){

    // δ_n = y^ - y
    // error = softmax_diff(probs, y)
    // where probs: probability of each output neuron
    let mut error = calc_output_error(&forward_output.probs, y);

    // for each from last to first, i: from N to 0
    for i in (0..model.num_hidden_layers - 1).rev() {

        // ∂L / ∂W_i = a^T * δ_i
        // dW_i = dot_product(a^T, error)
        gradients.d_W[i] = apply_dot_product(&transpose(forward_output.a[i]), &error);

        // ∂L / ∂b_i = δ_i
        // db_i = sum(error)
        gradients.d_b[i] = sum(&error);

        // δ_i-1 = δ_i * W_i^T * (1 - a_i^2)
        // error = dot_product(error, W_i^T)
        error = apply_dot_product(&error, &transpose(model.weights[i]));
    }
}




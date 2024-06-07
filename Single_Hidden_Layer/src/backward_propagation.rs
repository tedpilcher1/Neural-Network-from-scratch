use rand_distr::num_traits::ToPrimitive;
use crate::forward_propagation::ForwardOutput;
use crate::init_model::NN;
use crate::matrix_functions::apply_dot_product;
use crate::matrix_functions::transpose;


pub struct GradientWeightsBiases {

    pub dW1: Vec<Vec<f64>>,
    pub dW2: Vec<Vec<f64>>,
    pub db1: Vec<Vec<f64>>,
    pub db2: Vec<Vec<f64>>,
}


fn sum(delta : &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>{

    let mut z: Vec<Vec<f64>> = vec![vec![0.0; delta[0].len()]; 1];

    for column in 0..delta[0].len(){
        
        for row in 0..delta.len(){
            
            z[0][column] += delta[row][column];
        }
    }

    return z;
}

fn calc_output_error(probs : &Vec<Vec<f64>>, y : &Vec<i32>) -> Vec<Vec<f64>> {

    let mut delta3: Vec<Vec<f64>> = probs.clone();

    // subtract 1 from prob of true class (1) for each example
    for i in 0..delta3.len() {

        let j: usize = y[i].to_usize().unwrap();
        delta3[i][j] -= 1.0;
    }

    return delta3;
}

pub fn apply_backward_propagation (model : &mut NN, gradients: &mut GradientWeightsBiases, forward_output: &ForwardOutput, x : &Vec<Vec<f64>>, y : &Vec<i32>) {

    // calculate delta3, essentially the output layer error
    let mut delta3 = calc_output_error(&forward_output.probs, y);

    // calculate gradients of weights and biases for second layer (output)
    // dW2
    let mut a1T = transpose(forward_output.a1.clone());
    gradients.dW2 = apply_dot_product(&mut a1T, &mut delta3);

    // db2
    gradients.db2 = sum(&mut delta3);

    // delta2
    let mut w2T = transpose(model.weights_2.clone());
    let mut delta2 = apply_dot_product(&mut delta3, &mut w2T);

    // dW1
    let mut xT = transpose(x.clone());
    gradients.dW1 = apply_dot_product(&mut xT, &mut delta2);

    // calculate db1
    gradients.db1 = sum(&mut delta2);
}
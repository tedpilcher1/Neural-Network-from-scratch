use rand_distr::num_traits::ToPrimitive;
use crate::init_model::NeuralNetwork;
use crate::matrix_functions::apply_dot_product;

pub struct ForwardOutput {
    pub a: Vec<Vec<Vec<f64>>>, // output of each layer after applying activation function
    pub probs: Vec<Vec<f64>>, // probabilities of output layer
}

fn add_bias (z : &mut Vec<Vec<f64>>, b : & Vec<Vec<f64>>) {

    for i in 0..z.len(){

        for j in 0..z[0].len(){

            z[i][j] += b[0][j];
        }
    }
}

fn apply_tanh(z: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{

    let mut a = z.clone();

    for i in 0..a.len(){

        for j in 0..a[0].len(){

            a[i][j] = a[i][j].tanh();
        }
    }

    return a;
}

fn calc_a_i(i : usize, model : &NeuralNetwork, a_prev : &Vec<Vec<f64>>) -> Vec<Vec<f64>>{

    let mut z_i: Vec<Vec<f64>> = Vec::new();

    // calculate z_i, weighted sum of previous layer output + bias
    z_i = apply_dot_product(a_prev, &model.weights[i]);
    add_bias(&mut z_i, &model.biases[i]);

    // calculate a_i and return
    let a_i = apply_tanh(&z_i);

    return a_i;
}

fn calc_exp(z : &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    let mut exp = z.clone();

    for i in 0..exp.len() {
        for j in 0..exp[0].len() {
            exp[i][j] = exp[i][j].exp();
        }
    }

    return exp;
}

fn calc_probs(z : &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    // calc exp_scores
    let mut exp_scores = calc_exp(z);

    // sum scores
    let mut sums = vec![0.0; exp_scores.len()];

    for i in 0..exp_scores.len(){

        let mut curr_sum:f64 = 0.0;
        for j in 0..exp_scores[0].len(){

            curr_sum += exp_scores[i][j];
        }

        sums[i] = curr_sum;
    }

    // prob = scores / sum
    for i in 0..exp_scores.len(){

        for j in 0..exp_scores[0].len(){

            exp_scores[i][j] /= sums[i]
        }
    }

    return exp_scores;
}

pub fn apply_forward_propagation(model : &mut NeuralNetwork, forward_output: &mut ForwardOutput, x : &Vec<Vec<f64>>) {

    // assumes forward_output a is already initialised with correct length

    // calculate input of each layer
    // e.g. z1 is weighted sum of x + b1
    // e.g. z2 is weighted sum of a1 + b2

    // calculate the output of each layer after activation function
    // e.g. a1 = activation_function(z1)

    // a1
    forward_output.a[0] = calc_a_i(0, model, x);

    for i in 1..model.hidden_layer_sizes.len(){

        forward_output.a[i] = calc_a_i(i, model, &forward_output.a[i - 1])
    }

    // calculate probabilities
    forward_output.probs = calc_probs(&forward_output.a[forward_output.a.len() - 1]);
}
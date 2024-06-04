use crate::init_model::NN;
use crate::dot_product::apply_dot_product;

use std::f32::consts::PI;


fn add_bias (z : &mut Vec<Vec<f64>>, b : &mut Vec<Vec<f64>>) {

    for i in 0..z[0].len(){

        z[0][i] += b[0][i];
    }
}

fn apply_tanh(z: &mut Vec<Vec<f64>>){

    for i in 0..z[0].len() {

        z[0][i] = z[0][i].tanh();
    }    
}

fn calc_exp(z : &mut Vec<Vec<f64>>) {

    for i in 0..z[0].len() {

        z[0][i] = z[0][i].exp();
    }
}

fn calc_probs(exp_scores : &mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    // sum scores
    let mut sum: f64 = 0.0;

    for i in 0..exp_scores[0].len() {

        sum += exp_scores[0][i];
    }

    let mut probs = exp_scores.clone();
    
    // prob = scores / sum 
    for i in 0..exp_scores[0].len() {
        
        probs[0][i] = exp_scores[0][i] / sum;
    }

    return probs;
}

pub fn apply_forward_propagation (model : &mut NN, x : &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>{

    // calculate z1
    let mut z1 = apply_dot_product(x, &mut model.weights_1);
    add_bias(&mut z1, &mut model.bias_1);

    // calculate a1
    let mut a1 = z1.clone();
    apply_tanh(&mut a1);

    // calculate z2
    let mut z2 = apply_dot_product(&mut a1, &mut model.weights_2);
    add_bias(&mut z2, &mut model.bias_2);

    // calculate z2
    let mut exp_scores = z2.clone();
    calc_exp(&mut exp_scores);

    // calculate probabilities
    let mut probs = calc_probs(&mut exp_scores);

    return probs;
}
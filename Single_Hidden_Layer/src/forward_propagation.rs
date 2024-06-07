use rand_distr::num_traits::ToPrimitive;
use crate::init_model::NN;
use crate::matrix_functions::apply_dot_product;


pub struct ForwardOutput {
    pub a1 : Vec<Vec<f64>>,
    pub probs: Vec<Vec<f64>>,
}

fn add_bias (z : &mut Vec<Vec<f64>>, b : &mut Vec<Vec<f64>>) {

    for i in 0..z.len(){

        for j in 0..z[0].len(){

            z[i][j] += b[0][j];
        }
    }
}

fn apply_tanh(z: &mut Vec<Vec<f64>>){

    for i in 0..z.len(){

        for j in 0..z[0].len(){

            z[i][j] = z[i][j].tanh();
        }
    }
}

fn calc_exp(z : &mut Vec<Vec<f64>>) {

    for i in 0..z.len() {
        for j in 0..z[0].len() {
            z[i][j] = z[i][j].exp();
        }
    }
}

fn calc_probs(exp_scores : &mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    // sum scores
    let mut sums = vec![0.0; exp_scores.len()];

    for i in 0..exp_scores.len(){

        let mut curr_sum:f64 = 0.0;
        for j in 0..exp_scores[0].len(){

            curr_sum += exp_scores[i][j];
        }

        sums[i] = curr_sum;
    }

    let mut probs = exp_scores.clone();
    
    // prob = scores / sum 
    for i in 0..probs.len(){

        for j in 0..probs[0].len(){

            probs[i][j] /= sums[i]
        }
    }

    return probs;
}


pub fn apply_forward_propagation (model : &mut NN, x : &Vec<Vec<f64>>) -> ForwardOutput {

    let mut fpv = ForwardOutput {
        a1: Vec::new(),
        probs: Vec::new(),
    };

    // calculate z1
    let mut z1 = apply_dot_product(x, &mut model.weights_1);
    add_bias(&mut z1, &mut model.bias_1);

    // calculate a1s
    fpv.a1 = z1.clone();
    apply_tanh(&mut fpv.a1);


    // calculate z2
    let mut z2 = apply_dot_product(&mut fpv.a1, &mut model.weights_2);
    add_bias(&mut z2, &mut model.bias_2);

    // calculate z2
    let mut exp_scores = z2.clone();
    calc_exp(&mut exp_scores);

    // calculate probabilities
    fpv.probs = calc_probs(&mut exp_scores);

    // and return probabilities
    return fpv;
}

pub fn predict(model : &mut NN, x : &Vec<Vec<f64>>) -> i32{

    let forward_output = apply_forward_propagation(model, x);

    let mut maxProb:f64 = 0.0;
    let mut maxIndex: i32 = -1;

    for i in 0..forward_output.probs[0].len(){

        if forward_output.probs[0][i] > maxProb {

            maxProb = forward_output.probs[0][i];
            maxIndex = i.to_i32().unwrap();
        }
    }

    return maxIndex;
}
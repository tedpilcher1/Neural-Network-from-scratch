use crate::init_model::NN;
use create::dot_product::apply_dot_product;
use create::dot_product::transpose;

fn sum(delta : &mut Vec<Vec<f64>>) -> Vec<Vec<f64>>{

    let mut z: Vec<Vec<f64>> = vec![vec![0.0; delta[0].len()]; 1];

    for column in 0..delta[0].len(){
        
        for row in 0..delta.len(){
            
            z[0][column] += delta[row][column];
        }
    }

    return z;
}


fn apply_backward_propagation (model : &mut NN, probs : &mut Vec<Vec<f64>>, x : &mut Vec<Vec<f64>>, y : &mut Vec<i64>) {

    // calculate delta3
    let mut delta3 = probs.clone();
    // TODO

    // calculate dW2
    let mut a1T = transpose(a1);
    let dW2 = apply_dot_product(&mut a1T, &mut delta3);

    // calculate db2
    let mut db2 = sum(delta3);

    // calculate delta2
    let w2T = transpose(model.weights_2);
    let mut delta2 = dot(delta3, w2T);

    // calculate dW1
    xT = transpose(x);
    let dW1 = apply_dot_product(&mut xT, &mut delta2);

    // calculate db1
    let db1 = sum(delta2);
}
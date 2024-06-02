use std::vec;
use rand::{distributions::{Distribution, Uniform}, rngs::ThreadRng};


struct NN {

    num_input_nodes: i32,
    num_output_nodes: i32,
    num_hidden_nodes: i32,
    weights_1: Vec<Vec<f32>>,
    weights_2: Vec<Vec<f32>>,
    bias_1: Vec<Vec<f32>>,
    bias_2: Vec<Vec<f32>>,
}


fn init_layer_weights(weights: &mut Vec<Vec<f32>>, step: &Uniform<f32>, rng: &mut ThreadRng, prev_layer_size: i32, curr_layer_size: i32){

    for _i in 0..prev_layer_size {
        
        let mut new_weights: Vec<f32> = Vec::new();

        for _j in 0..curr_layer_size {
            
            new_weights.push(step.sample(rng));
        }

        weights.push(new_weights);
    }
}

fn init_bias(bias: &mut Vec<Vec<f32>>, num_nodes : i32){
    
    let mut row: Vec<f32> = Vec::new();
    for _i in 0..num_nodes {
        
        row.push(0.0)
    }

    bias.push(row)

}

fn build_model (my_num_hidden_nodes : i32) -> NN {

    const INPUT_SIZE: i32 = 2;
    const OUTPUT_SIZE: i32 = 2;

    // instantiate model
    let mut model = NN {
        num_input_nodes: INPUT_SIZE,
        num_output_nodes: OUTPUT_SIZE,
        num_hidden_nodes: my_num_hidden_nodes,
        weights_1: Vec::new(),
        weights_2: Vec::new(),
        bias_1: Vec::new(),
        bias_2: Vec::new(),
    };

    let step = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    // initalise first layer weights
    init_layer_weights(&mut model.weights_1, &step, &mut rng, INPUT_SIZE, my_num_hidden_nodes);

    // initalise second layer weights
    init_layer_weights(&mut model.weights_2, &step, &mut rng, my_num_hidden_nodes, OUTPUT_SIZE);

    // initalise biases
    init_bias(&mut model.bias_1, my_num_hidden_nodes);
    init_bias(&mut model.bias_2, OUTPUT_SIZE);

    return model;
}

fn dot_product(a : &mut Vec<Vec<f32>>, b : &mut Vec<Vec<f32>>, z : &mut Vec<Vec<f32>>){

    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_cols = b[0].len();

    let mut result: Vec<Vec<f32>> = vec![vec![0.0; b_cols]; a_rows];

    for i in 0..a_rows {

        for j in 0..b_cols {

            let mut sum: f32 = 0.0;
            for k in 0..a_cols {

                sum += a[i][k] * b[k][j];
            }

            z[i][j] = sum;
        }
    }
}

fn add_bias () {

}

fn forward_propagation (model : &mut NN, x : &mut Vec<f32>) {

    
    // calculate z1, dot product of x and W1 then add b1 bias

    // calculate a1


    // calculate z2


}


fn main(){

    // need to define dataset to train on

    let mut model = build_model(5);

}
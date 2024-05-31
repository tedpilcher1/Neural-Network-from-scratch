use std::vec;
use rand::{distributions::{Distribution, Uniform}, rngs::ThreadRng};


struct NN {

    num_hidden_nodes: i32,
    weights_1: Vec<Vec<f32>>,
    weights_2: Vec<Vec<f32>>,
    bias_1: Vec<f32>,
    bias_2: Vec<f32>,
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

fn init_bias(bias: &mut Vec<f32>, num_nodes : i32){

    for _i in 0..num_nodes {

        bias.push(0.0)
    }
}

fn build_model (my_num_hidden_nodes : i32) -> NN {

    const INPUT_SIZE: i32 = 2;
    const OUTPUT_SIZE: i32 = 2;

    // instantiate model
    let mut model = NN {
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

fn forward_propagation (model : &mut NN) {

    
}


fn main(){

    let mut model = build_model(5);

}
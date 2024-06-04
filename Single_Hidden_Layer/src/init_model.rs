use rand::thread_rng;
use rand::rngs::ThreadRng;
use rand_distr::{Normal, Distribution};


pub struct NN {
    pub input_size: i32,
    pub output_size: i32,
    pub hidden_layer_size: i32,
    pub weights_1: Vec<Vec<f64>>,
    pub weights_2: Vec<Vec<f64>>,
    pub bias_1: Vec<Vec<f64>>,
    pub bias_2: Vec<Vec<f64>>,
}


fn init_layer_weights(weights: &mut Vec<Vec<f64>>, normal: &Normal<f64>, rng: &mut ThreadRng, prev_layer_size: i32, curr_layer_size: i32){

    for _i in 0..prev_layer_size {
        
        let mut new_weights: Vec<f64> = Vec::new();

        for _j in 0..curr_layer_size {
            new_weights.push(normal.sample(rng));
        }

        weights.push(new_weights);
    }
}

fn init_bias(bias: &mut Vec<Vec<f64>>, num_nodes : i32){
    
    let mut row: Vec<f64> = Vec::new();
    for _i in 0..num_nodes {
        
        row.push(0.0)
    }

    bias.push(row)

}

pub fn build_model (input_size: i32, hidden_layer_size : i32, output_size : i32) -> NN {


    // instantiate model
    let mut model = NN {
        input_size: input_size,
        output_size: output_size,
        hidden_layer_size: hidden_layer_size,
        weights_1: Vec::new(),
        weights_2: Vec::new(),
        bias_1: Vec::new(),
        bias_2: Vec::new(),
    };

    let normal = Normal::new(0.0, 2.2361).unwrap();
    let mut rng = thread_rng();

    // initalise first layer weights
    init_layer_weights(&mut model.weights_1, &normal, &mut rng, model.input_size, model.hidden_layer_size);

    // initalise second layer weights
    init_layer_weights(&mut model.weights_2, &normal, &mut rng, model.hidden_layer_size, model.output_size);

    // initalise biases
    init_bias(&mut model.bias_1, model.hidden_layer_size);
    init_bias(&mut model.bias_2, model.output_size);

    return model;
}
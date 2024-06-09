use rand::thread_rng;
use rand::rngs::ThreadRng;
use rand_distr::{Normal, Distribution};


pub struct NeuralNetwork {
    pub input_size: i32,
    pub output_size: i32,
    pub num_hidden_layers: usize,
    pub hidden_layer_sizes: Vec<i32>, // e.g. [10, 5]: 10 node layer followed by 5 node layer
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<Vec<f64>>>,
}


fn init_single_layer_weights(normal: &Normal<f64>, rng: &mut ThreadRng, prev_layer_size: i32, layer_size: i32, weights: &mut Vec<Vec<Vec<f64>>>) {

    let mut layer_weights: Vec<Vec<f64>> = Vec::new();

    let sqrt_input_size = f64::from(prev_layer_size).sqrt();

    for _i in 0..prev_layer_size {

        let mut new_weights: Vec<f64> = Vec::new();

        for _j in 0..layer_size {
            new_weights.push(normal.sample(rng) / sqrt_input_size);
        }

        layer_weights.push(new_weights);
    }

    weights.push(layer_weights);
}

fn init_layer_weights (model : &mut NeuralNetwork, normal: &Normal<f64>, rng: &mut ThreadRng) {

    // nothing to initialise if no hidden layers
    if model.hidden_layer_sizes.len() == 0 {
        return;
    }

    // init weights between input layer and first hidden layer
    init_single_layer_weights(normal, rng, model.input_size, model.hidden_layer_sizes[0], &mut model.weights);

    // init weights between each hidden layer
    for i in 1..model.hidden_layer_sizes.len(){

        init_single_layer_weights(normal, rng, model.hidden_layer_sizes[i - 1], model.hidden_layer_sizes[i], &mut model.weights);
    }

    // init weights between last hidden layer and output layer
    init_single_layer_weights(normal, rng, model.hidden_layer_sizes[model.hidden_layer_sizes.len() - 1], model.output_size, &mut model.weights);
}

fn init_biases (model : &mut NeuralNetwork) {

    // initialise each hidden layers biases
    for i in 0..model.hidden_layer_sizes.len(){

        model.biases.push(vec![vec![0.0; model.hidden_layer_sizes[i] as usize]; 1]);
    }

    // add last biases for output layer
    model.biases.push(vec![vec![0.0; model.output_size as usize]; 1])
}

pub fn build_model (input_size : i32, output_size: i32, hidden_layer_sizes : Vec<i32>) -> NeuralNetwork{

    let mut model = NeuralNetwork {

        input_size,
        output_size,
        num_hidden_layers: hidden_layer_sizes.len(),
        hidden_layer_sizes,
        weights: Vec::new(),
        biases: Vec::new(),
    };

    init_biases(&mut model);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    init_layer_weights(&mut model, &normal, &mut rng);

    return model;
}
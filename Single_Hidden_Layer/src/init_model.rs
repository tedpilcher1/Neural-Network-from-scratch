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

    let sqrt_input_size = f64::from(prev_layer_size).sqrt();

    for _i in 0..prev_layer_size {
        
        let mut new_weights: Vec<f64> = Vec::new();

        for _j in 0..curr_layer_size {
            new_weights.push(normal.sample(rng) / sqrt_input_size);
        }

        weights.push(new_weights);
    }
}

pub fn build_model (input_size: i32, hidden_layer_size : i32, output_size : i32) -> NN {


    // instantiate model
    let mut model = NN {
        input_size: input_size,
        output_size: output_size,
        hidden_layer_size: hidden_layer_size,
        weights_1: Vec::new(),
        weights_2: Vec::new(),
        bias_1: vec![vec![0.0; hidden_layer_size as usize]; 1],
        bias_2:  vec![vec![0.0; output_size as usize]; 1],
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    // initalise first layer weights
    init_layer_weights(&mut model.weights_1, &normal, &mut rng, model.input_size, model.hidden_layer_size);

    // initalise second layer weights
    init_layer_weights(&mut model.weights_2, &normal, &mut rng, model.hidden_layer_size, model.output_size);

    return model;
}
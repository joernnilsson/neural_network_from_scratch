 
extern crate nalgebra as na;

extern crate argparse;

mod utils;
mod nn;
mod data;


use crate::data::Dataset;
use crate::nn::NeuralNetwork;

/*

Ideas:
 - Detect stall
  - Restart with new seed
 - Detect convergance
 - Track best model



*/

fn main(){


    // Define network
    let mut nn = nn::SimpleNeuralNetwork::new(vec![2, 2, 1]);


    // Load dataset
    let dataset = data::JsonDataset::new("data/xor.json".to_string());


    // Train network
    let x = dataset.get_x();
    let y = dataset.get_y();

    nn.train(&x, &y, 5.0, 10000);


    // Test network
    {
        let mut correct = 0;
        x.column_iter().zip(y.column_iter()).for_each(|(x_col, y_col)| {

            let result = nn.predict(&x_col.into());

            let cost = nn.cost(&y_col.into(), &result);
            println!("{} XOR {} = {:.5}, expected: {} (cost: {:.5})", x_col[0], x_col[1], result[0], y_col[0], cost);
            if (result[0] - y_col[0]).abs() < 0.5 {
                correct += 1;
            }
        });
        println!("Correct: {}/{}", correct, x.ncols());
    }

    // Save model
    nn.save_model("model.json");

    // Load model from file and test it
    let nn2 = nn::SimpleNeuralNetwork::from_file("model.json").unwrap();
    {
        let mut correct = 0;
        x.column_iter().zip(y.column_iter()).for_each(|(x_col, y_col)| {

            let result = nn2.predict(&x_col.into());

            let cost = nn2.cost(&y_col.into(), &result);
            println!("{} XOR {} = {:.5}, expected: {} (cost: {:.5})", x_col[0], x_col[1], result[0], y_col[0], cost);
            if (result[0] - y_col[0]).abs() < 0.5 {
                correct += 1;
            }
        });
        println!("Correct: {}/{}", correct, x.ncols());
    }


}

// https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
// http://neuralnetworksanddeeplearning.com/chap2.html
// http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
// https://github.com/mnielsen/neural-networks-and-deep-learning

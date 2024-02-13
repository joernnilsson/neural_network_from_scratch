 
#![allow(non_snake_case)]

// use rand::Rng;
// use rayon::{prelude::*, result};

extern crate nalgebra as na;
use na::{DMatrix, DVector};

extern crate argparse;
// use argparse::{ArgumentParser, Store};
// use plotters::prelude::*;

mod mat_loader;
mod utils;
mod nn;

// use matio_rs::MatFile;
// use approx::assert_abs_diff_eq;

/*

Ideas:
 - Detect stall
  - Restart with new seed
 - Detect convergance
 - Track best model



*/

fn main(){
    let mut nn = nn::NeuralNetwork::new(vec![2, 2, 1]);

    let x = DMatrix::from_row_iterator(4, 2, [
        0., 0., 
        0., 1., 
        1., 0., 
        1., 1., 
    ].iter().cloned());

    let y = DMatrix::from_row_iterator(4, 1, [
        0.,
        1.,
        1.,
        0.,
    ].iter().cloned());

    nn.train(&x, &y, 5.0, 10000);


    let test: na::Matrix<f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>> = DMatrix::from_row_iterator(4, 2, [
        0., 0., 
        0., 1., 
        1., 0., 
        1., 1., 
    ].iter().cloned());


    let mut correct = 0;
    x.row_iter().zip(y.row_iter()).for_each(|(x_row, y_row)| {

        let col: DVector<f64> = x_row.transpose();

        // let result = nn.predict(&x_row.into());
        let result = nn.predict(&col);

        let cost = nn.cost(&y_row.into(), &result);
        println!("{} XOR {} = {:.5}, expected: {} (cost: {:.5})", x_row[0], x_row[1], result[0], y_row[0],cost);
        if (result[0] - y_row[0]).abs() < 0.5 {
            correct += 1;
        }
    });
    println!("Correct: {}/{}", correct, x.nrows());


}

// https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
// http://neuralnetworksanddeeplearning.com/chap2.html
// http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
// https://github.com/mnielsen/neural-networks-and-deep-learning

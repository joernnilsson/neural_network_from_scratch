 
#![allow(non_snake_case)]

// use rand::Rng;
// use rayon::{prelude::*, result};

extern crate nalgebra as na;
use na::{DMatrix, DVector, RowVector1};

extern crate argparse;
// use argparse::{ArgumentParser, Store};

// use plotters::prelude::*;

mod mat_loader;
mod utils;
mod nn;

// use matio_rs::MatFile;

// use approx::assert_abs_diff_eq;

macro_rules! dim {
    ($var:expr) => {
        println!("{} rows/columns: {}x{}", stringify!($var), $var.shape().0, $var.shape().1);
    };
}

macro_rules! mat {
    ($var:expr) => {
        println!("{} =\n{}", stringify!($var), $var);
    };
}

#[allow(dead_code)]
fn forward_single_layer(input: &DMatrix<f64>, weights: &DMatrix<f64>) -> DMatrix<f64> {

    // Each row of x is a training example
    dim!(input);
    dim!(weights);
    let layer_input = input * weights;

    // Apply activation function to all elements
    let layer_value = layer_input.map(|x| utils::sigmoid(x));

    layer_value
}


/**
 * x: 1d matrix
 * y: 1d matrix
 */
// #[allow(dead_code)]
// fn partial_fit(x: &na::Matrix<f64, na::Const<1>, na::Dyn, na::ViewStorage<'_, f64, na::Const<1>, na::Dyn, na::Const<1>, na::Dyn>>, y: &na::Matrix<f64, na::Const<1>, na::Dyn, na::ViewStorage<'_, f64, na::Const<1>, na::Dyn, na::Const<1>, na::Dyn>>, w: &DMatrix<f64>, alpha: f64) {

//https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/

// }

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


    // let bias = DVector::from_element(test.nrows(), 1.0);
    // let test_bias = utils::append_column(&test, &bias);

    x.row_iter().zip(y.row_iter()).for_each(|(x_row, y_row)| {
        let result = nn.predict(&x_row.into());

        let cost = nn.cost(&y_row.into(), &result);
        println!("{} XOR {} = {:.5}, expected: {} (cost: {:.5})", x_row[0], x_row[1], result[0], y_row[0],cost);
    });

    // test.row_iter().for_each(|row| {
    //     let result = nn.predict(&row.into());
    //     let cost = nn.cost(&row.into(), &result);
    //     println!("{} XOR {} = {} (cost: {})", row[0], row[1], result[0]), cost;
    // });

}

// https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
#[allow(dead_code)]
fn main_(){

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

    // 2-2-1 network with bias embedded (effectively 3-3-1)
    let bias = DVector::from_row_slice(&[1., 1., 1., 1.]);
    let x_bias = utils::append_column(&x, &bias);

    let dim_input = 3;
    let dim_layer_1 = 3;
    let dim_output = 1;

    let mut w_1 = DMatrix::from_row_iterator(dim_input, dim_layer_1, (0..dim_input*dim_layer_1).map(|_| rand::random::<f64>()));
    let mut w_2 = DMatrix::from_row_iterator(dim_layer_1, dim_output, (0..dim_layer_1*dim_output).map(|_| rand::random::<f64>()));

    {
        mat!(x);
        mat!(x_bias);
        mat!(w_1);

        // Forward propagation
        let hidden_layer_value = forward_single_layer(&x_bias, &w_1);
        let output_layer_value = forward_single_layer(&hidden_layer_value, &w_2);

        mat!(hidden_layer_value);
        mat!(output_layer_value);

    }

    // Fit
    let alpha = 100.0;
    let epochs = 10000;
    for i in 0..epochs{
        for row_idx in 0..x_bias.nrows(){
            let x_row: na::Matrix<f64, na::Const<1>, na::Dyn, na::ViewStorage<'_, f64, na::Const<1>, na::Dyn, na::Const<1>, na::Dyn>> = x_bias.row(row_idx);
            let y_row: na::Matrix<f64, na::Const<1>, na::Dyn, na::ViewStorage<'_, f64, na::Const<1>, na::Dyn, na::Const<1>, na::Dyn>> = y.row(row_idx);

            // self.partial_fit(&x_row, &y_row, &w_1, alpha);
        }
    }



}
use std::vec;

use nalgebra::{DMatrix, DVector, RowDVector};
use crate::{dim, mat, utils};


pub struct NeuralNetwork {
    spec: Vec<u64>,
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DVector<f64>>,
}

pub fn make_weights_matrix(input_dim: u64, output_dim: u64) -> DMatrix<f64> {
    DMatrix::from_row_iterator(output_dim as usize, input_dim as usize, (0..input_dim*output_dim).map(|_| rand::random::<f64>()))
}

impl NeuralNetwork {
    pub fn new(spec: Vec<u64>) -> NeuralNetwork {

        // At least 1 hidden layer
        assert!(spec.len() > 2);

        let weights = spec.windows(2).map(|s| { make_weights_matrix(s[0], s[1])}).collect();

        let biases = spec.iter().skip(1).map(|s| {DVector::from_element(*s as usize, 1.0)}).collect();

        let nn = NeuralNetwork {
            spec: spec.clone(),
            weights: weights,
            biases: biases,
        };
        nn
    }

    /**
     * Takes a single sample input and propagates it through the network
     * Return tuple of activations and z values for each layer
     */
    pub fn forward_sample(&self, input: &DVector<f64>) -> (Vec<DVector<f64>>, Vec<DVector<f64>>){

        let mut a_v: Vec<DVector<f64>> = Vec::new();
        let mut z_v: Vec<DVector<f64>> = Vec::new();

        a_v.push(input.clone());

        let _ = self.weights.iter().zip(self.biases.iter()).fold(input.clone(), |acc, (weight, bias)| {
            let z = weight * &acc + bias;
            let a = z.map(|x| utils::sigmoid(x));
            
            let a_copy = a.clone();

            z_v.push(z);
            a_v.push(a);

            a_copy
        });

        (z_v, a_v)
    }

    pub fn fit(&self, input: &RowDVector<f64>, y_target: &RowDVector<f64>, alpha: f64) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>, f64) {

        let input = input.transpose();
        let y_target = y_target.transpose();



        let (z_v, a_v) = self.forward_sample(&input);


        // Backward pass: d_l = (W_l+1)^T * d_l+1 * f'(z_l)

        // Iterate backwards from L-1 to 1 (layers, stopping at idx 1)
        let mut delta_v: Vec<DVector<f64>> = Vec::new();
        let mut dC_dW_v: Vec<DMatrix<f64>> = Vec::new();

        let cost_derivative = a_v.last().unwrap() - &y_target;

        // Add to vectors, then reverse
        {
            delta_v.push(cost_derivative.component_mul(&z_v.last().unwrap().map(|x| utils::sigmoid_derivative(x)))); // Output layer
            let v = delta_v.last().unwrap() * a_v[a_v.len() - 2].transpose();
            dC_dW_v.push(v);
        }


        for l in (1..(self.weights.len())).rev(){
            let delta_l = (self.weights[l].transpose() * delta_v.last().unwrap()).component_mul(&z_v[l-1].map(|x| utils::sigmoid_derivative(x)));

            delta_v.push(delta_l);

            let v = delta_v.last().unwrap() * a_v[l-1].transpose();

            dC_dW_v.push(v);
        }

        let cost = (1./2.)*(&y_target - a_v.last().unwrap()).norm_squared();

        // Test
        delta_v.reverse();
        dC_dW_v.reverse();

        (dC_dW_v, delta_v, cost)

 
    }


    pub fn predict(&self, input: &DVector<f64>) -> DVector<f64> {

        let (_, a_v) = self.forward_sample(input);
        a_v.last().unwrap().clone()

    }

    pub fn cost(&self, y_pred: &RowDVector<f64>, y_target: &DVector<f64>) -> f64 {
        (1./2.)*(y_target - y_pred).norm_squared()
    }

    pub fn train(&mut self, x: &DMatrix<f64>, target: &DMatrix<f64>, alpha: f64, epochs: u64) -> DMatrix<f64> {

        let mut costs: Vec<f64> = Vec::new();

        for n in 0..epochs{

            let mut cost = 0.0;
            for i in 0..x.nrows(){

                let input_row: RowDVector<f64> = x.row(i).into();
                let y_row: RowDVector<f64> = target.row(i).into();


                let (dC_dW_v, delta_v, sample_cost) = self.fit(&input_row,&y_row, alpha/x.nrows() as f64);


                // Update weights
                for l in 0..self.weights.len(){
                    self.weights[l] = &self.weights[l] - (alpha/(x.nrows() as f64)) * &dC_dW_v[l];
                }

                // Update biases
                for l in 0..self.biases.len(){
                    self.biases[l] = &self.biases[l] - (alpha/(x.nrows() as f64)) * &delta_v[l];
                }

                cost += sample_cost / (x.nrows() as f64);

            }

            if(n % 1000 == 0){
                println!("{} Cost: {}", n, cost);
            }
            costs.push(cost);

        }

        let input_transposed = x.transpose();

        self.forward_sample(&input_transposed.column(0).into());

        utils::write_vector_to_csv("costs.csv", &costs).unwrap();

        target.clone()

    }
}
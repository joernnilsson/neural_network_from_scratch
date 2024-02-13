use std::vec;

use nalgebra::{DMatrix, DVector, RowDVector};
use crate::{dim, mat, utils};


// macro_rules! dim {
//     ($var:expr) => {
//         println!("{} rows/columns: {}x{}", stringify!($var), $var.shape().0, $var.shape().1);
//     };
// }

// macro_rules! mat {
//     ($var:expr) => {
//         println!("{} =\n{}", stringify!($var), $var);
//     };
// }

pub struct NeuralNetwork {
    spec: Vec<u64>,
    // spec_internal: Vec<u64>,
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
        // pub fn fit(&self, input: &RowDVector<f64>, y_target: &RowDVector<f64>, alpha: f64) -> (DMatrix<f64>, DMatrix<f64>, DVector<f64>, DVector<f64>, f64){

        // Do forward activations and store results
        // l = 0: input
        // l = 1: hidden
        // l = 2: output (aka L)

        // TODO Reconfigure input and y_target as column vectors

        let input = input.transpose();
        let y_target = y_target.transpose();

        // // dim!(self.weights[0]);
        // // dim!(input);
        // // dim!(self.weights[1]);

        // // Forward pass: Z = W * A + b

        // // Hidden layer input
        // let hidden_in = &self.weights[0] * &input + &self.biases[0];
        // let hidden_out = &hidden_in.map(|x| utils::sigmoid(x));

        // // Output layer input
        // let output_in = &self.weights[1] * hidden_out + &self.biases[1];
        // let output_out = &output_in.map(|x| utils::sigmoid(x));

        // // dim!(output_in);
        // // dim!(output_out);

        // let a_0 = input;
        // let a_1 = hidden_out;
        // let a_2 = output_out;

        // let z_1 = hidden_in;
        // let z_2 = output_in;


        let (z_v, a_v) = self.forward_sample(&input);

        // let a_0 = a_v[0].clone();
        // let a_1 = a_v[1].clone();
        // let a_2 = a_v[2].clone();
        // let z_1 = z_v[0].clone();
        // let z_2 = z_v[1].clone();


        // Backward pass: d_l = (W_l+1)^T * d_l+1 * f'(z_l)


        // Iterate backwards from L-1 to 1 (layers, stopping at idx 1)
        let mut delta_v: Vec<DVector<f64>> = Vec::new();
        let mut dC_dW_v: Vec<DMatrix<f64>> = Vec::new();

        let cost_derivative = a_v.last().unwrap() - &y_target;

        // Add to vectors, then reverse
        // TODO replace z_2 with z_v[1], a_1 with a_v[0]
        {
            delta_v.push(cost_derivative.component_mul(&z_v.last().unwrap().map(|x| utils::sigmoid_derivative(x)))); // Output layer
            let v = delta_v.last().unwrap() * a_v[a_v.len() - 2].transpose();
            dC_dW_v.push(v);
        }


        for l in (1..(self.weights.len())).rev(){
            // println!("---------------- {} ----------------", l);
            // TODO figure out proper indexes for z_v
            let delta_l = (self.weights[l].transpose() * delta_v.last().unwrap()).component_mul(&z_v[l-1].map(|x| utils::sigmoid_derivative(x)));
            // delta_l self.weights[1]   delta L  z_v[0]
            // mat!(&delta_l);
            delta_v.push(delta_l);

            // mat!(delta_v.last().unwrap());

            let v = delta_v.last().unwrap() * a_v[l-1].transpose();
            // delta_1    av[1]
            dC_dW_v.push(v);
        }


        // Gammel impl

        // // Trenger: (error)-, z2*, produserer delta2-
        // let delta_L = (&a_2 - &y_target).component_mul(&z_2.map(|x| utils::sigmoid_derivative(x)));

        // // Trenger w1, delta2, z1* produserer delta1
        // let delta_1 = (self.weights[1].transpose() * &delta_L).component_mul(&z_1.map(|x| utils::sigmoid_derivative(x)));

        // // dC / dW

        // // l = 2
        // let dC_dW_L: DMatrix<f64> = &delta_L * a_1.transpose();

        // // l = 1
        // let dC_dW_1: DMatrix<f64> = &delta_1 * a_0.transpose();




        let cost = (1./2.)*(&y_target - a_v.last().unwrap()).norm_squared();

        // mat!(y_target);
        // mat!(a_2);

        // Test
        delta_v.reverse();
        dC_dW_v.reverse();

        // mat!(dC_dW_v[0]);
        // mat!(dC_dW_1);

        // mat!(delta_1);
        // mat!(a_0);

        // Original
        // delta_v.clear();
        // dC_dW_v.clear();

        // delta_v.push(delta_L.column(0).into());
        // delta_v.push(delta_1.column(0).into());

        // dC_dW_v.push(dC_dW_L);
        // dC_dW_v.push(dC_dW_1);

        // delta_v.reverse();
        // dC_dW_v.reverse();



        // panic!("abc");

        (dC_dW_v, delta_v, cost)

        // (dC_dW_1, dC_dW_L, delta_1, delta_L, cost)

 
    }


    pub fn predict(&self, input: &DVector<f64>) -> DVector<f64> {

        let (_, a_v) = self.forward_sample(input);
        a_v.last().unwrap().clone()

    }

    pub fn cost(&self, y_pred: &RowDVector<f64>, y_target: &DVector<f64>) -> f64 {
        (1./2.)*(y_target - y_pred).norm_squared()
    }

    pub fn train(&mut self, input: &DMatrix<f64>, target: &DMatrix<f64>, alpha: f64, epochs: u64) -> DMatrix<f64> {

        // Extend input with bias
        let x = input;

        // Forward pass
        // let y_pred = self.forward(&x);

        // Backward pass
        let mut costs: Vec<f64> = Vec::new();

        for n in 0..epochs{

            let mut cost = 0.0;
            for i in 0..x.nrows(){

                // println!("***** i: {}", i);
                let input_row: RowDVector<f64> = x.row(i).into();
                // let y_pred_row: RowDVector<f64> = y_pred.row(i).into();
                let y_row: RowDVector<f64> = target.row(i).into();

                // self.fit(&input_row,&y_row, alpha/x.nrows() as f64);

                // let (dC_dW_1, dC_dW_L, delta_1, delta_L, sample_cost) = self.fit(&input_row,&y_row, alpha/x.nrows() as f64);
                let (dC_dW_v, delta_v, sample_cost) = self.fit(&input_row,&y_row, alpha/x.nrows() as f64);

                // dim!(self.weights[1]);
                // dim!((alpha/(x.nrows() as f64)) * &dC_dW_1);

                // Update weights

                // let dC_dW_L = dC_dW_v[1].clone();
                // let w_test = (alpha/(x.nrows() as f64)) * dC_dW_L;

                // dim!(self.weights[1]);
                // dim!(w_test);

                // Update weights
                for l in 0..self.weights.len(){
                    self.weights[l] = &self.weights[l] - (alpha/(x.nrows() as f64)) * &dC_dW_v[l];
                }

                // Update biases
                for l in 0..self.biases.len(){
                    self.biases[l] = &self.biases[l] - (alpha/(x.nrows() as f64)) * &delta_v[l];
                }


                // self.weights[1] = &self.weights[1] - (alpha/(x.nrows() as f64)) * &dC_dW_v[1];
                // self.weights[0] = &self.weights[0] - (alpha/(x.nrows() as f64)) * &dC_dW_v[0];

                // // Update biases
                // self.biases[1] = &self.biases[1] - (alpha/(x.nrows() as f64)) * &delta_v[1];
                // self.biases[0] = &self.biases[0] - (alpha/(x.nrows() as f64)) * &delta_v[0];

                // // Update weights
                // self.weights[1] = &self.weights[1] - (alpha/(x.nrows() as f64)) * &dC_dW_L;
                // self.weights[0] = &self.weights[0] - (alpha/(x.nrows() as f64)) * &dC_dW_1;

                // // Update biases
                // self.biases[1] = &self.biases[1] - (alpha/(x.nrows() as f64)) * delta_L;
                // self.biases[0] = &self.biases[0] - (alpha/(x.nrows() as f64)) * delta_1;

                cost += sample_cost / (x.nrows() as f64);

            }

            if(n % 1000 == 0){
                println!("{} Cost: {}", n, cost);
            }
            costs.push(cost);
            //let delta = self.partial_derivative(&y_pred, target);

        }

        let input_transposed = input.transpose();

        dim!(input);
        dim!(input_transposed);


        self.forward_sample(&input_transposed.column(0).into());

        // mat!(self.weights[0]);
        // mat!(self.weights[1]);
        // mat!(self.biases[0]);
        // mat!(self.biases[1]);

        utils::write_vector_to_csv("costs.csv", &costs).unwrap();

        target.clone()

    }
}
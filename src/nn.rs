use std::vec;

use nalgebra::{DMatrix, DVector, RowDVector};
use crate::utils;

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

pub struct NeuralNetwork {
    spec: Vec<u64>,
    spec_internal: Vec<u64>,

    // input_dim: u64,
    // output_dim: u64,
    // hidden_layers_dim: Vec<u64>,

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

        // Only one hidden layer supported for now
        assert!(spec.len() == 3);

        // let spec_internal = vec![spec[0] + 1, spec[1] + 1, spec[spec.len()-1]];
        let spec_internal = vec![spec[0], spec[1], spec[spec.len()-1]];

        let weights = vec![
            make_weights_matrix(spec_internal[0], spec_internal[1]),
            make_weights_matrix(spec_internal[1], spec_internal[2]),
        ];

        let biases = vec![
            DVector::from_element(spec_internal[1] as usize, 1.0),
            DVector::from_element(spec_internal[2] as usize, 1.0),
        ];

        let nn = NeuralNetwork {
            spec: spec.clone(),
            spec_internal: spec_internal,
            // input_dim: spec[0] + 1,
            // output_dim: spec[spec.len()-1],
            // hidden_layers_dim: vec![spec[1] + 1],
            weights: weights,
            biases: biases,
        };
        nn
    }

    pub fn forward(&self, input: &DMatrix<f64>) -> DMatrix<f64>{

        // Propagate trhough all layers
        self.weights.iter().fold(input.clone(), |acc, weight| {
            self.forward_single_layer(&acc, weight)
        })
    }

    pub fn forward_single_layer(&self, input: &DMatrix<f64>, weights: &DMatrix<f64>) -> DMatrix<f64> {

        dim!(input);
        dim!(weights);

        // Each row of x is a training example
        let layer_input = input * weights;
    
        // Apply activation function to all elements
        let layer_value = layer_input.map(|x| utils::sigmoid(x));
    
        layer_value
    }

    pub fn partial_derivative(&self, y_pred: &DMatrix<f64>, y_target: &DMatrix<f64>) -> DMatrix<f64> {
        let error = y_pred - y_target;

        dim!(error);
        dim!(y_pred);

        // Storage for deltas
        let mut delta: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<_, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(y_pred.nrows(), self.spec.len() - 1);

        // TODO First, try to do this with a single target/y_pred pair

        // Calculate delta for output layer
        let delta_output = error.component_mul(&y_pred.map(|x| utils::sigmoid_derivative(x)));
        let delta_output_vector = DVector::from_column_slice(delta_output.as_slice());
        delta.set_column(delta.ncols()-1, &delta_output_vector);

        // Calculate delta for hidden layers
        for i in (0..(self.weights.len()-1)).rev() {
            dim!(delta.column(i+1));
            dim!(self.weights[i].transpose());
            let delta_hidden = delta.column(i+1) * self.weights[i].transpose();
            let delta_hidden_vector = DVector::from_column_slice(delta_hidden.as_slice());
            delta.set_column(i, &delta_hidden_vector);
            println!("i: {}", i);
        }         
        println!("w: {}", self.weights.len());
        mat!(delta);

        dim!(delta);



        delta
    }

    pub fn fit2(&self, input: &RowDVector<f64>, y_target: &RowDVector<f64>, alpha: f64) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>){

        // Do forward activations and store results
        // l = 0: input
        // l = 1: hidden
        // l = 2: output (aka L)

        // TODO Reconfigure input and y_target as column vectors

        let input = input.transpose();
        let y_target = y_target.transpose();

        // dim!(self.weights[0]);
        // dim!(input);
        // dim!(self.weights[1]);

        // Forward pass: Z = W * A + b

        let mut activation = input.clone();
        let mut activations: Vec<DVector<f64>> = vec![input.clone()];
        let mut zs: Vec<DVector<f64>> = Vec::new();

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = w * activation + b;
            zs.push(z.clone());
            activation = z.map(|z_elem| utils::sigmoid(z_elem));
            activations.push(activation.clone());
        }

        // backward pass
        let mut nabla_b: Vec<DVector<f64>> = self.biases.iter().map(|b| DVector::zeros(b.nrows())).collect();
        let mut nabla_w: Vec<DMatrix<f64>> = self.weights.iter().map(|w| DMatrix::zeros(w.nrows(), w.ncols())).collect();

        let mut delta = (activations.last().unwrap() - &y_target).component_mul(&zs.last().unwrap().map(utils::sigmoid_derivative));
        nabla_b.insert(nabla_b.len() - 1, delta.clone());
        nabla_w.insert(nabla_w.len() - 1, delta.clone() * activations[activations.len() - 2].transpose());

        for l in 2..self.spec.len() {
            let z = &zs[zs.len() - l];
            let sp = z.map(utils::sigmoid_derivative);
            dim!(sp);
            dim!(delta);
            let delta_local = self.weights[self.weights.len() - l + 1].transpose() * delta.component_mul(&sp);
            delta = delta_local;
            nabla_b.insert(nabla_b.len() - l, delta.clone());
            nabla_w.insert(nabla_w.len() - l, delta.clone() * activations[activations.len() - l - 1].transpose());
        }

        (nabla_b, nabla_w)



    }


    pub fn fit(&self, input: &RowDVector<f64>, y_target: &RowDVector<f64>, alpha: f64) -> (DMatrix<f64>, DMatrix<f64>, DVector<f64>, DVector<f64>, f64){

        // Do forward activations and store results
        // l = 0: input
        // l = 1: hidden
        // l = 2: output (aka L)

        // TODO Reconfigure input and y_target as column vectors

        let input = input.transpose();
        let y_target = y_target.transpose();

        // dim!(self.weights[0]);
        // dim!(input);
        // dim!(self.weights[1]);

        // Forward pass: Z = W * A + b

        // Hidden layer input
        let hidden_in = &self.weights[0] * &input + &self.biases[0];
        let hidden_out = &hidden_in.map(|x| utils::sigmoid(x));

        // Output layer input
        let output_in = &self.weights[1] * hidden_out + &self.biases[1];
        let output_out = &output_in.map(|x| utils::sigmoid(x));

        // dim!(output_in);
        // dim!(output_out);

        let a_0 = input;
        let a_1 = hidden_out;
        let a_2 = output_out;

        let z_1 = hidden_in;
        let z_2 = output_in;

        // Backward pass: d_l = (W_l+1)^T * d_l+1 * f'(z_l)

        let delta_L = (a_2 - &y_target).component_mul(&z_2.map(|x| utils::sigmoid_derivative(x)));


        let delta_1 = (self.weights[1].transpose() * &delta_L).component_mul(&z_1.map(|x| utils::sigmoid_derivative(x)));
        //let delta_0 = (self.weights[0].transpose() * &delta_1).component_mul(&a_0.map(|x| utils::sigmoid_derivative(x)));

        // nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        // dC / dW

        // l = 2
        let dC_dW_L = &delta_L * a_1.transpose();

        // l = 1
        let dC_dW_1 = &delta_1 * a_0.transpose();


        // let nabla_w = vec![dC_dW_1, dC_dW_L];
        // let nabla_b = vec![delta_1, delta_L];


        let cost = (1./2.)*(&y_target - a_2).norm_squared();

        // mat!(y_target);
        // mat!(a_2);

        (dC_dW_1, dC_dW_L, delta_1, delta_L, cost)


 
    }

    pub fn predict(&self, x: &RowDVector<f64>) -> DVector<f64> {

        let input = x.transpose();

        // dim!(self.weights[0]);
        // dim!(input);

        // Hidden layer input
        let hidden_in = &self.weights[0] * &input + &self.biases[0];
        let hidden_out = &hidden_in.map(|x| utils::sigmoid(x));

        // Output layer input
        let output_in = &self.weights[1] * hidden_out + &self.biases[1];
        let output_out = &output_in.map(|x| utils::sigmoid(x));


        // dim!(output_out);

        let vec_out: DVector<f64> = output_out.column(0).into();
        vec_out
    }

    pub fn cost(&self, y_pred: &RowDVector<f64>, y_target: &DVector<f64>) -> f64 {
        (1./2.)*(y_target - y_pred).norm_squared()
    }

    pub fn train(&mut self, input: &DMatrix<f64>, target: &DMatrix<f64>, alpha: f64, epochs: u64) -> DMatrix<f64> {

        // Extend input with bias
        // let bias = DVector::from_element(input.nrows(), 1.0);
        // let x = utils::append_column(&input, &bias);
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

                self.fit(&input_row,&y_row, alpha/x.nrows() as f64);

                let (dC_dW_1, dC_dW_L, delta_1, delta_L, sample_cost) = self.fit(&input_row,&y_row, alpha/x.nrows() as f64);

                // dim!(self.weights[1]);
                // dim!((alpha/(x.nrows() as f64)) * &dC_dW_1);

                // Update weights
                self.weights[1] = &self.weights[1] - (alpha/(x.nrows() as f64)) * &dC_dW_L;
                self.weights[0] = &self.weights[0] - (alpha/(x.nrows() as f64)) * &dC_dW_1;

                // Update biases
                self.biases[1] = &self.biases[1] - (alpha/(x.nrows() as f64)) * delta_L;
                self.biases[0] = &self.biases[0] - (alpha/(x.nrows() as f64)) * delta_1;

                cost += sample_cost / (x.nrows() as f64);
            }

            if(n % 1000 == 0){
                println!("{} Cost: {}", n, cost);
            }
            costs.push(cost);
            //let delta = self.partial_derivative(&y_pred, target);

        }

        // mat!(self.weights[0]);
        // mat!(self.weights[1]);
        // mat!(self.biases[0]);
        // mat!(self.biases[1]);

        utils::write_vector_to_csv("costs.csv", &costs).unwrap();

        target.clone()

    }
}
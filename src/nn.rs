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

        let spec_internal = vec![spec[0] + 1, spec[1] + 1, spec[spec.len()-1]];

        let weights = vec![
            make_weights_matrix(spec_internal[0], spec_internal[1]),
            make_weights_matrix(spec_internal[1], spec_internal[2]),
        ];

        let nn = NeuralNetwork {
            spec: spec.clone(),
            spec_internal: spec_internal,
            // input_dim: spec[0] + 1,
            // output_dim: spec[spec.len()-1],
            // hidden_layers_dim: vec![spec[1] + 1],
            weights: weights,
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




    pub fn fit(&self, input: &RowDVector<f64>, y_target: &RowDVector<f64>, alpha: f64) -> (DMatrix<f64>, DMatrix<f64>, f64){

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

        // Forward pass: Z = W * A

        // Hidden layer input
        let hidden_in = &self.weights[0] * &input;
        let hidden_out = &hidden_in.map(|x| utils::sigmoid(x));

        // Output layer input
        let output_in = &self.weights[1] * hidden_out;
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

        // dim!(self.weights[1].transpose());
        // dim!(delta_L);
        // dim!(z_1.map(|x| utils::sigmoid_derivative(x)));

        let delta_1 = (self.weights[1].transpose() * &delta_L).component_mul(&z_1.map(|x| utils::sigmoid_derivative(x)));
        let delta_0 = (self.weights[0].transpose() * &delta_1).component_mul(&a_0.map(|x| utils::sigmoid_derivative(x)));

        // nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        // dC / dW

        // l = 2
        let nabla_L = delta_L * a_1.transpose();

        // l = 1
        let nabla_1 = delta_1 * a_0.transpose();

        let cost = (1./2.)*(&y_target - a_2).norm_squared();

        mat!(y_target);
        mat!(a_2);

        (nabla_1, nabla_L, cost)


 
    }

    pub fn predict(&self, x: &RowDVector<f64>) -> DVector<f64> {

        let input = x.transpose();

        // dim!(self.weights[0]);
        // dim!(input);

        // Hidden layer input
        let hidden_in = &self.weights[0] * &input;
        let hidden_out = &hidden_in.map(|x| utils::sigmoid(x));

        // Output layer input
        let output_in = &self.weights[1] * hidden_out;
        let output_out = output_in.map(|x| utils::sigmoid(x));

        output_out
    }

    pub fn train(&mut self, input: &DMatrix<f64>, target: &DMatrix<f64>, alpha: f64, epochs: u64) -> DMatrix<f64> {

        // Extend input with bias
        let bias = DVector::from_element(input.nrows(), 1.0);
        let x = utils::append_column(&input, &bias);

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

                let (dC_dw_1, dC_dw_L, sample_cost) = self.fit(&input_row,&y_row, alpha/x.nrows() as f64);

                // Update weights
                self.weights[1] = &self.weights[1] - (alpha/(x.nrows() as f64)) * dC_dw_L;
                self.weights[0] = &self.weights[0] - (alpha/(x.nrows() as f64)) * dC_dw_1;

                cost += sample_cost / (x.nrows() as f64);
            }

            println!("{} Cost: {}", n, cost);
            costs.push(cost);
            //let delta = self.partial_derivative(&y_pred, target);

        }

        utils::write_vector_to_csv("costs.csv", &costs).unwrap();

        target.clone()

    }
}
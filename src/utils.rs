
use na::{DMatrix, DVector};
use csv::Writer;
use std::error::Error;
use std::fs::File;

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

pub fn append_column(matrix: &DMatrix<f64>, column: &DVector<f64>) -> DMatrix<f64> {
   let mut extended_matrix = DMatrix::<f64>::zeros(matrix.shape().0, matrix.shape().1 + 1);
   extended_matrix.slice_mut((0, 0), matrix.shape()).copy_from(&matrix); // Copy the original matrix into the new matrix
   extended_matrix.column_mut(matrix.shape().1).copy_from(&column); // Copy the new column into the new matrix
   extended_matrix
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn write_vector_to_csv(filepath: &str, data: &Vec<f64>) -> Result<(), Box<dyn Error>> {
    // Open a file in write mode
    let file = File::create(filepath)?;
    
    // Create a CSV writer
    let mut wtr = Writer::from_writer(file);

    // Iterate over the data and write each value as a record
    for value in data {
        wtr.serialize(value)?;
    }

    // Ensure all data is flushed to the disk
    wtr.flush()?;
    Ok(())
}
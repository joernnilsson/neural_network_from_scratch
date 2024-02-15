
use std::vec;

use nalgebra::{DMatrix, DVector, RowDVector};
use crate::{dim, mat, utils};
use serde_json;
use serde::{Deserialize};


// Interface trait
pub trait Dataset {

    // Returns a tuple of the training input and output data
    fn get_x(&self) -> &DMatrix<f64>;
    fn get_y(&self) -> &DMatrix<f64>;

}




// Json dataset

pub struct JsonDataset {
    filename: String,
    x: DMatrix<f64>,
    y: DMatrix<f64>,
}

#[derive(Deserialize)]
struct JsonDataSpec {
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
}


impl JsonDataset {
    pub fn new(filename: String) -> JsonDataset {

        let data = JsonDataset::load(&filename);
        JsonDataset {
            filename: filename,
            x: data.0,
            y: data.1,
        }
    }

    fn load(filename: &String) -> (DMatrix<f64>, DMatrix<f64>) {

        let file_contents = std::fs::read_to_string(filename).unwrap();
        let data: JsonDataSpec = serde_json::from_str(&file_contents).unwrap();

        // Convert Vec<Vec<f64>> into DMatrix<f64>
        let x_matrix = DMatrix::from_vec(data.x[0].len(), data.x.len(), data.x.iter().flat_map(|r| r.iter()).cloned().collect());
        let y_matrix = DMatrix::from_vec(data.y[0].len(), data.y.len(), data.y.iter().flat_map(|r| r.iter()).cloned().collect());

        // Printing the matrices to check
        println!("X Matrix:\n{}", x_matrix);
        println!("Y Matrix:\n{}", y_matrix);

        (x_matrix, y_matrix)
    }

}

impl Dataset for JsonDataset {

    fn get_x(&self) -> &DMatrix<f64> {
        &self.x
    }

    fn get_y(&self) -> &DMatrix<f64> {
        &self.y
    }

}









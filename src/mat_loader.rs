// use matio_rs::MatFile;

// pub struct MatLoader {
//     file: String,
//     mat_file: MatFile,
// }

// impl MatLoader {
//     pub fn new(file: &str) -> Result<MatLoader, Box<dyn std::error::Error>> {
//         let mat_file = MatFile::load(file)?;
//         Ok(MatLoader {
//             file: file.to_string(),
//             mat_file,
//         })
//     }

//     // pub fn get_var<T>(&self, var_name: &str) -> Result<T, Box<dyn std::error::Error>>
//     // where
//     //     T: FromMat,
//     // {
//     //     let var = self.mat_file.var(var_name)?;
//     //     Ok(T::from_mat(var))
    
//     // }
// }

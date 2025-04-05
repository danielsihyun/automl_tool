use ndarray::{Array2, Array1};
use std::error::Error;
use csv::ReaderBuilder;

pub fn load_csv(
    path: &str,
    target_column: &str,
) -> Result<(Array2<f64>, Array1<usize>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path(path)?;
    let headers = rdr.headers()?.clone();

    let target_index = headers
        .iter()
        .position(|h| h == target_column)
        .ok_or("Target column not found in headers")?;

    let mut features_vec: Vec<Vec<f64>> = vec![];
    let mut target_vec: Vec<usize> = vec![];

    for result in rdr.records() {
        let record = result?;
        let mut feature_row = vec![];

        for (i, field) in record.iter().enumerate() {
            let value: f64 = field.parse().unwrap_or(0.0);

            if i == target_index {
                target_vec.push(value as usize);
            } else {
                feature_row.push(value);
            }
        }

        features_vec.push(feature_row);
    }

    let n_rows = features_vec.len();
    let n_cols = features_vec[0].len();

    let x = Array2::from_shape_vec((n_rows, n_cols), features_vec.concat())?;
    let y = Array1::from(target_vec);

    Ok((x, y))
}

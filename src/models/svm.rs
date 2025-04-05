use linfa::prelude::*;
use linfa_svm::Svm;
use linfa::dataset::DatasetBase;
use ndarray::{Array2, Array1};

pub fn train(
    train: &DatasetBase<Array2<f64>, Array1<usize>>,
    test: &DatasetBase<Array2<f64>, Array1<usize>>,
) -> (&'static str, f32) {
    let y_train_bool = train
        .targets()
        .iter()
        .map(|&label| label != 0)
        .collect::<Vec<bool>>();

    let y_test_bool = test
        .targets()
        .iter()
        .map(|&label| label != 0)
        .collect::<Vec<bool>>();

    let train_bin = DatasetBase::new(train.records().to_owned(), Array1::from(y_train_bool));
    let test_bin = DatasetBase::new(test.records().to_owned(), Array1::from(y_test_bool));

    let model = Svm::<f64, bool>::params()
        .gaussian_kernel(50.0)
        .fit(&train_bin)
        .unwrap();

    let pred = model.predict(&test_bin);
    let acc = pred.confusion_matrix(&test_bin).unwrap().accuracy();
    let acc = if acc.is_nan() { 0.0 } else { acc };

    ("SVM (Binary)", acc)
}

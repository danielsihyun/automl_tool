use linfa::prelude::*;
use linfa_trees::DecisionTree;
use linfa::dataset::DatasetBase;
use ndarray::{Array2, Array1};

pub fn train(
    train: &DatasetBase<Array2<f64>, Array1<usize>>,
    test: &DatasetBase<Array2<f64>, Array1<usize>>,
) -> (&'static str, f32) {
    let model = DecisionTree::params().fit(train).unwrap();
    let pred = model.predict(test);
    let acc = pred.confusion_matrix(test).unwrap().accuracy();
    let acc = if acc.is_nan() { 0.0 } else { acc };

    ("Decision Tree", acc)
}

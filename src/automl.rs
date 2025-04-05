use ndarray::{Array2, Array1};
use linfa::dataset::DatasetBase;
use crate::models::{logistic, decision_tree, svm};

pub fn run(x: Array2<f64>, y: Array1<usize>) {
    println!("Splitting dataset and training models...");

    let dataset = DatasetBase::new(x, y);
    let (train, test) = dataset.split_with_ratio(0.8);

    let results = vec![
        logistic::train(&train, &test),
        decision_tree::train(&train, &test),
        svm::train(&train, &test),
    ];

    println!("\nResults:");
    let mut best_score = 0.0;
    let mut best_model = "";

    for (name, acc) in results {
        println!("{}: Accuracy = {:.2}%", name, acc * 100.0);
        if acc > best_score {
            best_score = acc;
            best_model = name;
        }
    }

    println!("\nBest Model: {} (Accuracy: {:.2}%)", best_model, best_score * 100.0);
}

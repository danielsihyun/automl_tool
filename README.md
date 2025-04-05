This project is a command-line AutoML tool written in Rust that automates the process of training, evaluating, and selecting the best machine learning model for classification tasks on tabular data.

Overview

The tool takes a CSV file and a target column name as input, splits the data into training and testing sets, trains multiple models, evaluates their accuracy, and identifies the best-performing model.

Currently Supported Models:
- Logistic Regression
- Decision Tree
- Support Vector Machine (binary classification only)

How It Works

1. Load and parse a CSV file with numeric features and a categorical target column.
2. Perform an 80/20 train-test split.
3. Train three models on the training data.
4. Predict and evaluate on the test set.
5. Print accuracy scores for each model and highlight the best one.

Getting Started

To run the tool:

cargo run -- --file data_1000.csv --target label

Make sure your CSV file has:
- A header row
- At least two numeric feature columns
- One categorical target column

Example

Input CSV (data_1000.csv):

feature1,feature2,label  
1.0,2.0,0  
3.1,4.2,1  
...  

Expected CLI Output:

Loaded data: X shape = (1000, 2), y shape = 1000  
Training models...  

Results:  
Logistic Regression: Accuracy = 87.00%  
Decision Tree: Accuracy = 90.50%  
SVM (binary): Accuracy = 92.00%  

Best Model: SVM (binary) (Accuracy: 92.00%)

Roadmap

Planned features for future releases:
- Hyperparameter tuning via grid search
- Inference mode to load saved models and make predictions on new data
- Model exporting to disk
- Support for multiclass SVM using One-vs-Rest
- Interactive WebAssembly frontend using Yew

Tech Stack

- Rust
- linfa (machine learning)
- ndarray (numerical arrays)
- clap (command-line interface)
- csv (file handling)

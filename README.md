# FD-HDP
1. Project Overview
This project presents a Feature Disentanglement based Heterogeneous Defect Prediction (FD-HDP) method. It aims to address issues like data sparsity and feature heterogeneity in cross-project defect prediction. By disentangling features into domain-related and domain-independent ones, and integrating adversarial learning and feature reconstruction, it improves the defect prediction performance of target projects.
2. Project Structure
project/
├── dataset/
├── data_preprocessing.py
├── evaluate.py
├── feature_disentanglement.py
├── loss_functions.py
├── main.py
├── prediction_layer.py
├── train.py
├── README.txt
3.Environmental Requirement
Python 3.x
PyTorch
Pandas
NumPy
Scikit-learn
imblearn
SciPy
The following command can be used to install the required libraries:
pip install torch pandas numpy scikit-learn imblearn scipy
4.Dataset Preparation
Multiple open-source datasets are used in the project, including AEEEM, NASA, PROMISE, and ReLink, etc. The dataset files should be in ARFF format and stored in the "dataset" directory. Each dataset should contain feature and "defects" label columns.
5. Method of Application
(1) Install Dependencies
First, use the following command to install the required libraries:
pip install torch numpy pandas scikit-learn scipy imbalanced-learn
(2)Configure dataset path
In the main.py file, you can modify the datasets list as needed to specify the paths of the datasets to be used. For example:
datasets = [
    'dataset/AEEEM/EQ.arff',
    'dataset/NASA/PC1.arff',
]
(3) Run the main program
Execute the following command in the terminal to run the main program:
python main.py

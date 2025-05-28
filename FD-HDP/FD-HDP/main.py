# main.py
import copy
import os

import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn

from data_preprocessing import load_arff, balance_data, convert_labels, MLPFeatureCompression
from feature_disentanglement import FeatureDisentanglement, FeatureReconstructor, DomainSharedClassifier, \
    GradientReversalLayer
from prediction_layer import Predictor
from loss_functions import LossFunctions
from train import pretrain_model, fine_tune_model
from evaluate import evaluate_model
import torch
import numpy as np
import torch.nn.functional as F


# 主函数
def main():
    # 定义数据集路径
    datasets = [
        # 'dataset/AEEEM/EQ.arff',
        # 'dataset/AEEEM/JDT.arff',
        # 'dataset/AEEEM/Lucene.arff',
        # 'dataset/AEEEM/Mylyn.arff',
        # 'dataset/AEEEM/PDE.arff',
        # 'dataset/NASA/cm1.arff',
        # 'dataset/NASA/mw1.arff',
        # 'dataset/NASA/PC1.arff',
        # 'dataset/NASA/PC3.arff',
        # 'dataset/NASA/PC4.arff',
        # 'dataset/PROMISE/ant-1.3.arff',
        # 'dataset/PROMISE/camel-1.6.arff',
        # 'dataset/PROMISE/ivy-2.0.arff',
        # 'dataset/PROMISE/jedit-4.1.arff',
        # 'dataset/PROMISE/log4j-1.2.arff',
        # 'dataset/PROMISE/poi-2.0.arff',
        # 'dataset/PROMISE/prop-6.arff',
        # 'dataset/PROMISE/synapse-1.2.arff',
        # 'dataset/PROMISE/tomcat.arff',
        # 'dataset/PROMISE/velocity-1.4.arff',
        'dataset/PROMISE/xalan-2.4.arff',
        # 'dataset/PROMISE/xerces-1.2.arff',
        # 'dataset/ReLink/Apache.arff',
        'dataset/ReLink/Safe.arff',
        # 'dataset/ReLink/Zxing.arff'

    ]

    results = []

    for source_dataset_path in datasets:
        for target_dataset_path in datasets:

            source_project = os.path.dirname(source_dataset_path).split('/')[-1]
            target_project = os.path.dirname(target_dataset_path).split('/')[-1]

            if source_project == target_project:
                continue

            print(f"源数据集: {source_dataset_path}, 目标数据集: {target_dataset_path}")

            source_data = load_arff(source_dataset_path)
            target_data = load_arff(target_dataset_path)

            X_source_raw = source_data.drop('defects', axis=1).values
            y_source_raw = convert_labels(source_data['defects'].values)
            X_target_raw = target_data.drop('defects', axis=1).values
            y_target_raw = convert_labels(target_data['defects'].values)

            X_source_balanced, y_source_balanced = X_source_raw, y_source_raw
            X_target_balanced, y_target_balanced = balance_data(X_target_raw, y_target_raw)

            X_source_normalized = MinMaxScaler().fit_transform(X_source_balanced)
            X_target_normalized = MinMaxScaler().fit_transform(X_target_balanced)

            source_compressor = MLPFeatureCompression(input_dim=X_source_raw.shape[1], compressed_dim=128)
            target_compressor = MLPFeatureCompression(input_dim=X_target_raw.shape[1], compressed_dim=128)
            X_source_compressed = source_compressor(torch.tensor(X_source_normalized).float()).detach().numpy()
            X_target_compressed = target_compressor(torch.tensor(X_target_normalized).float()).detach().numpy()

            X_target_labeled, X_target_unlabeled, y_target_labeled, _ = train_test_split(
                X_target_compressed,
                y_target_balanced,
                test_size=0.5,
                stratify=y_target_balanced
            )

            X_source_res = torch.tensor(X_source_compressed).float()
            y_source_res = torch.tensor(y_source_balanced).float()
            X_target_labeled = torch.tensor(X_target_labeled).float()
            X_target_unlabeled = torch.tensor(X_target_unlabeled).float()
            y_target_labeled = torch.tensor(y_target_labeled).float()

            source_compressor = MLPFeatureCompression(input_dim=X_source_raw.shape[1], compressed_dim=128)
            target_compressor = MLPFeatureCompression(input_dim=X_target_raw.shape[1], compressed_dim=128)
            disentanglement_layer = FeatureDisentanglement(d_in=128, hidden_dims=[64, 32], disentangle_dim=64)
            feature_reconstructor = FeatureReconstructor(d_in=128, h_specific_dim=128, h_share_dim=128)
            domain_shared_classifier = DomainSharedClassifier(disentangle_dim=128)
            prediction_layer = Predictor(d_in=128, hidden_dims=[128, 64], alpha_T=0.5, alpha_S=0.5)
            loss_functions = LossFunctions(alpha=1.0, beta=1.0, gamma=1.0)

            X_target_full = torch.cat([X_target_labeled, X_target_unlabeled])
            y_target_full = torch.cat([y_target_labeled, torch.zeros(len(X_target_unlabeled))])

            pretrain_model(
                X_source_res,
                X_target_full,
                disentanglement_layer,
                feature_reconstructor,
                domain_shared_classifier,
                loss_functions,
                epochs=50
            )

            prediction_layer.alpha_S = 0.5

            fine_tune_model(
                X_source_res, y_source_res,
                X_target_labeled, y_target_labeled,
                disentanglement_layer,
                feature_reconstructor,
                domain_shared_classifier,
                prediction_layer,
                loss_functions,
                epochs=100
            )
            metrics = evaluate_model(disentanglement_layer, prediction_layer, loss_functions, X_source_res, X_target_labeled, y_target_labeled)

            results.append({
                'Source Dataset': os.path.basename(source_dataset_path),
                'Target Dataset': os.path.basename(target_dataset_path),
                'Precision': np.mean(metrics['precision']),
                'Recall': np.mean(metrics['recall']),
                'F1': np.mean(metrics['f1']),
                'AUC': np.mean(metrics['auc']),
                'G-Mean': np.mean(metrics['g_mean']),
                'H-Mean': np.mean(metrics['h_mean'])
            })

if __name__ == "__main__":
    main()

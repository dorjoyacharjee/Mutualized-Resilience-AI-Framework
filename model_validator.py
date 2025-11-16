"""
Model Validator
Cross-validation, performance metrics, and result analysis

Evaluates:
- MREP loss forecasting accuracy
- MREP insurer retreat prediction
- HAM damage detection F1/IoU scores
- SFM solvency predictions
- Combined system performance

Author: Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.model_selection import TimeSeriesSplit, cross_validate, cross_val_predict
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    confusion_matrix, classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, jaccard_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Comprehensive model validation framework"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize validator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config['paths']['reports'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validation results storage
        self.results = {}

        logger.info("ModelValidator initialized")

    def time_series_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_func,
        n_splits: int = 5
    ) -> Dict:
        """
        Perform time series cross-validation (maintains temporal order)

        Args:
            X: Feature array
            y: Target array
            model_func: Model with fit/predict interface
            n_splits: Number of CV folds

        Returns:
            CV results dictionary
        """
        logger.info(f"Performing {n_splits}-fold time series CV...")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model_func.fit(X_train, y_train)

            # Predict
            y_pred = model_func.predict(X_test)

            # Metrics
            if len(np.unique(y)) == 2:  # Classification
                metrics = {
                    'accuracy': np.mean(y_pred == y_test),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                }
            else:  # Regression
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mape': mean_absolute_percentage_error(y_test, y_pred)
                }

            fold_results.append(metrics)

            logger.info(f"  Fold {fold+1}: {list(metrics.values())}")

        # Aggregate results
        cv_results = {}
        for metric in fold_results[0].keys():
            values = [f[metric] for f in fold_results]
            cv_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return cv_results

    def evaluate_mrep_lstm(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str = "MREP_LSTM_LossForecasting"
    ) -> Dict:
        """
        Evaluate LSTM loss forecasting component

        Args:
            y_true: True annual aggregate losses
            y_pred: Predicted losses
            metric_name: Name for results storage

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating MREP LSTM component...")

        metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
            'r2': float(1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
        }

        logger.info(f"MREP LSTM Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        self.results[metric_name] = metrics

        return metrics

    def evaluate_mrep_xgboost(
        self,
        y_true: np.ndarray,
        y_pred_class: np.ndarray,
        y_pred_proba: np.ndarray,
        metric_name: str = "MREP_XGBoost_RetreatPrediction"
    ) -> Dict:
        """
        Evaluate XGBoost insurer retreat classifier

        Args:
            y_true: True retreat labels
            y_pred_class: Predicted class labels
            y_pred_proba: Predicted probabilities
            metric_name: Name for results storage

        Returns:
            Classification metrics
        """
        logger.info("Evaluating MREP XGBoost component...")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_class)

        # Metrics
        metrics = {
            'accuracy': float(np.mean(y_pred_class == y_true)),
            'precision': float(precision_score(y_true, y_pred_class, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_class, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred_class, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'specificity': float(cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0),
            'confusion_matrix': cm.tolist()
        }

        logger.info(f"MREP XGBoost Metrics:")
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                logger.info(f"  {k}: {v:.4f}")

        self.results[metric_name] = metrics

        return metrics

    def evaluate_ham_cnn(
        self,
        y_true_masks: np.ndarray,
        y_pred_masks: np.ndarray,
        num_classes: int = 3,
        metric_name: str = "HAM_CNN_DamageDetection"
    ) -> Dict:
        """
        Evaluate U-Net CNN segmentation

        Args:
            y_true_masks: True segmentation masks (N, H, W, Classes)
            y_pred_masks: Predicted masks
            num_classes: Number of classes
            metric_name: Name for results storage

        Returns:
            Segmentation metrics
        """
        logger.info("Evaluating HAM CNN component...")

        # Convert to class predictions
        y_true_classes = np.argmax(y_true_masks, axis=-1).flatten()
        y_pred_classes = np.argmax(y_pred_masks, axis=-1).flatten()

        # Pixel-wise metrics
        metrics = {
            'accuracy': float(np.mean(y_true_classes == y_pred_classes)),
            'precision': float(precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0))
        }

        # Per-class IoU (Intersection over Union)
        iou_per_class = {}
        for class_id in range(num_classes):
            y_true_binary = (y_true_classes == class_id).astype(int)
            y_pred_binary = (y_pred_classes == class_id).astype(int)

            iou = jaccard_score(y_true_binary, y_pred_binary, zero_division=0)
            iou_per_class[f'class_{class_id}'] = float(iou)

        metrics['iou'] = iou_per_class
        metrics['mean_iou'] = float(np.mean(list(iou_per_class.values())))

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true_classes, y_pred_classes).tolist()

        logger.info(f"HAM CNN Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")

        self.results[metric_name] = metrics

        return metrics

    def evaluate_sfm_solvency(
        self,
        solvency_true: np.ndarray,
        solvency_pred: np.ndarray,
        capital_true: np.ndarray,
        capital_pred: np.ndarray,
        metric_name: str = "SFM_Solvency"
    ) -> Dict:
        """
        Evaluate SFM solvency predictions

        Args:
            solvency_true: True solvency status
            solvency_pred: Predicted solvency
            capital_true: True fund capital
            capital_pred: Predicted capital
            metric_name: Name for results storage

        Returns:
            SFM metrics
        """
        logger.info("Evaluating SFM component...")

        metrics = {
            'solvency_accuracy': float(np.mean(solvency_true == solvency_pred)),
            'solvency_f1': float(f1_score(solvency_true, solvency_pred, zero_division=0)),
            'capital_mae': float(mean_absolute_error(capital_true, capital_pred)),
            'capital_mape': float(mean_absolute_percentage_error(capital_true, capital_pred))
        }

        logger.info(f"SFM Metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        self.results[metric_name] = metrics

        return metrics

    def basis_risk_analysis(
        self,
        satellite_damage: np.ndarray,
        actual_loss: np.ndarray
    ) -> Dict:
        """
        Analyze basis risk (correlation between satellite index and actual loss)

        Args:
            satellite_damage: Satellite-detected damage percentage
            actual_loss: Actual verified losses

        Returns:
            Basis risk metrics
        """
        logger.info("Analyzing basis risk...")

        # Correlation
        correlation = np.corrcoef(satellite_damage, actual_loss)[0, 1]

        # False positives (payout but no loss)
        false_payout = np.sum((satellite_damage > 20) & (actual_loss < 0.05))

        # False negatives (loss but no payout)
        false_no_payout = np.sum((satellite_damage < 20) & (actual_loss > 0.2))

        basis_risk = {
            'correlation': float(correlation),
            'false_positive_rate': float(false_payout / len(satellite_damage)),
            'false_negative_rate': float(false_no_payout / len(actual_loss)),
            'basis_risk_score': float(1 - abs(correlation))
        }

        logger.info(f"Basis Risk:")
        logger.info(f"  Correlation: {basis_risk['correlation']:.4f}")
        logger.info(f"  Basis Risk Score: {basis_risk['basis_risk_score']:.4f}")

        self.results['BasisRisk'] = basis_risk

        return basis_risk

    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report

        Returns:
            Path to report file
        """
        logger.info("Generating validation report...")

        report_path = self.output_dir / f"validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Add metadata
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'framework': 'Climate-Adaptive Agricultural Insurance AI',
            'validation_results': self.results
        }

        # Overall summary
        summary = {
            'mrep_loss_forecast_rmse': self.results.get('MREP_LSTM_LossForecasting', {}).get('rmse', 'N/A'),
            'mrep_retreat_prediction_auc': self.results.get('MREP_XGBoost_RetreatPrediction', {}).get('roc_auc', 'N/A'),
            'ham_damage_detection_miou': self.results.get('HAM_CNN_DamageDetection', {}).get('mean_iou', 'N/A'),
            'basis_risk_score': self.results.get('BasisRisk', {}).get('basis_risk_score', 'N/A')
        }
        report['summary'] = summary

        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved: {report_path}")

        return str(report_path)

    def plot_validation_results(self) -> List[str]:
        """
        Generate validation visualization plots

        Returns:
            List of plot file paths
        """
        plot_paths = []

        # Plot 1: Model accuracies comparison
        if self.results:
            fig, ax = plt.subplots(figsize=(10, 6))

            model_names = []
            accuracies = []

            for model_name, metrics in self.results.items():
                if 'accuracy' in metrics:
                    model_names.append(model_name)
                    accuracies.append(metrics['accuracy'])

            if model_names:
                ax.bar(model_names, accuracies, color='steelblue')
                ax.set_ylabel('Accuracy')
                ax.set_title('Model Validation Accuracies')
                ax.set_ylim([0, 1])
                plt.xticks(rotation=45, ha='right')

                plot_path = self.output_dir / 'validation_accuracies.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths.append(str(plot_path))
                plt.close()

        logger.info(f"Generated {len(plot_paths)} validation plots")

        return plot_paths


if __name__ == "__main__":
    validator = ModelValidator()

    # Example evaluation
    y_true_mrep = np.random.randint(0, 2, 100)
    y_pred_mrep = np.random.rand(100)
    y_pred_class = (y_pred_mrep > 0.5).astype(int)

    metrics_mrep = validator.evaluate_mrep_xgboost(y_true_mrep, y_pred_class, y_pred_mrep)
    print(f"\nMREP XGBoost F1-Score: {metrics_mrep['f1_score']:.4f}")

    # Basis risk analysis
    sat_damage = np.random.uniform(0, 50, 100)
    actual_loss = sat_damage + np.random.normal(0, 5, 100)
    basis_risk = validator.basis_risk_analysis(sat_damage, actual_loss)
    print(f"Basis Risk Correlation: {basis_risk['correlation']:.4f}")

    # Generate report
    report_path = validator.generate_validation_report()
    print(f"\nValidation report: {report_path}")

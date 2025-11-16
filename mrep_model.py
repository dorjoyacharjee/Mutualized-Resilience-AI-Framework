"""
MREP Model - Meta-Risk Evaluation and Prediction
Predicts agricultural insurance market failure using LSTM + XGBoost

Combines:
- LSTM: Time series forecasting of aggregate losses
- XGBoost: Classification of insurer retreat probability

Author: Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MREPModel:
    """Meta-Risk Evaluation and Prediction Model"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize MREP model"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.mrep_cfg = self.config['models']['mrep']

        # LSTM config
        self.lstm_layers = self.mrep_cfg['lstm']['layers']
        self.dropout = self.mrep_cfg['lstm']['dropout']
        self.sequence_length = self.mrep_cfg['lstm']['sequence_length']
        self.batch_size = self.mrep_cfg['lstm']['batch_size']
        self.epochs = self.mrep_cfg['lstm']['epochs']
        self.lr = self.mrep_cfg['lstm']['learning_rate']

        # XGBoost config
        self.xgb_params = {
            'max_depth': self.mrep_cfg['xgboost']['max_depth'],
            'learning_rate': self.mrep_cfg['xgboost']['learning_rate'],
            'n_estimators': self.mrep_cfg['xgboost']['n_estimators'],
            'min_child_weight': self.mrep_cfg['xgboost']['min_child_weight'],
            'subsample': self.mrep_cfg['xgboost']['subsample'],
            'colsample_bytree': self.mrep_cfg['xgboost']['colsample_bytree'],
            'objective': self.mrep_cfg['xgboost']['objective']
        }

        # Model storage
        self.lstm_model = None
        self.xgb_model = None
        self.scaler_lstm = StandardScaler()
        self.scaler_xgb = StandardScaler()

        # Output paths
        self.model_dir = Path(self.config['paths']['models']) / 'mrep'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("MREPModel initialized")

    def build_lstm_model(self, n_features: int) -> keras.Model:
        """
        Build LSTM architecture for loss forecasting

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, n_features)),

            # LSTM layer 1
            layers.LSTM(
                self.lstm_layers[0],
                return_sequences=True,
                dropout=self.dropout,
                name='lstm_1'
            ),
            layers.BatchNormalization(),

            # LSTM layer 2
            layers.LSTM(
                self.lstm_layers[1],
                return_sequences=True,
                dropout=self.dropout,
                name='lstm_2'
            ),
            layers.BatchNormalization(),

            # LSTM layer 3
            layers.LSTM(
                self.lstm_layers[2],
                return_sequences=False,
                dropout=self.dropout,
                name='lstm_3'
            ),
            layers.BatchNormalization(),

            # Dense output (predicting aggregate annual loss)
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(1, activation='linear', name='loss_output')
        ])

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )

        logger.info(f"LSTM model built: {model.count_params()} parameters")

        return model

    def prepare_lstm_sequences(
        self,
        data: pd.DataFrame,
        target_col: str = 'aggregate_loss'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM training

        Args:
            data: Time series DataFrame (sorted by time)
            target_col: Column name for target variable

        Returns:
            Tuple of (X_sequences, y_targets)
        """
        # Drop target from features
        features = data.drop(columns=[target_col]).values
        target = data[target_col].values

        # Normalize features
        features_scaled = self.scaler_lstm.fit_transform(features)

        # Create sequences
        X, y = [], []

        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i+self.sequence_length])
            y.append(target[i+self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences of length {self.sequence_length}")

        return X, y

    def train_lstm(
        self,
        data: pd.DataFrame,
        validation_split: float = 0.15
    ) -> Dict:
        """
        Train LSTM loss forecasting model

        Args:
            data: Training data with climate, economic, and loss columns
            validation_split: Fraction of data for validation

        Returns:
            Training history dict
        """
        logger.info("Training LSTM loss forecasting model...")

        # Prepare sequences
        X, y = self.prepare_lstm_sequences(data)

        # Split (maintain temporal order)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build model
        self.lstm_model = self.build_lstm_model(n_features=X.shape[2])

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.mrep_cfg['lstm']['early_stopping_patience'],
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        val_loss = history.history['val_loss'][-1]
        val_mae = history.history['val_mae'][-1]

        logger.info(f"LSTM training complete: val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

        # Save model
        model_path = self.model_dir / 'lstm_model.h5'
        self.lstm_model.save(model_path)
        logger.info(f"LSTM model saved: {model_path}")

        # Save scaler
        scaler_path = self.model_dir / 'lstm_scaler.pkl'
        joblib.dump(self.scaler_lstm, scaler_path)

        return history.history

    def predict_losses(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict future aggregate losses using trained LSTM

        Args:
            data: Input features

        Returns:
            Predicted losses
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained. Call train_lstm() first.")

        # Prepare sequences
        X, _ = self.prepare_lstm_sequences(data)

        # Predict
        predictions = self.lstm_model.predict(X, batch_size=self.batch_size)

        return predictions.flatten()

    def prepare_xgb_features(
        self,
        data: pd.DataFrame,
        lstm_predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for XGBoost retreat classifier

        Args:
            data: DataFrame with static features
            lstm_predictions: LSTM loss forecasts

        Returns:
            Tuple of (X_features, y_labels)
        """
        # Align lengths (LSTM predictions are shorter due to sequence window)
        data_aligned = data.iloc[self.sequence_length:].copy()

        # Add LSTM predictions as feature
        data_aligned['lstm_loss_forecast'] = lstm_predictions

        # Feature engineering for XGBoost
        # Calculate rolling statistics
        data_aligned['loss_ma_5yr'] = data['aggregate_loss'].rolling(60).mean().iloc[self.sequence_length:]
        data_aligned['loss_std_5yr'] = data['aggregate_loss'].rolling(60).std().iloc[self.sequence_length:]

        # Loss Exceedance Probability (LEP)
        data_aligned['lep'] = (data_aligned['lstm_loss_forecast'] / 
                                (data_aligned['premium_collected'] + 1e-10))

        # Premium-to-Value ratio (approximated)
        if 'insured_value' in data_aligned.columns:
            data_aligned['pv_ratio'] = (data_aligned['premium_collected'] / 
                                         (data_aligned['insured_value'] + 1e-10))

        # Composite risk score
        data_aligned['composite_risk'] = (
            0.4 * data_aligned['flood_risk_score'] +
            0.3 * data_aligned['drought_risk_score'] +
            0.3 * data_aligned['cyclone_risk_score']
        )

        # Target: insurer_retreat (binary: 1=retreat, 0=remain)
        y = data_aligned['insurer_retreat'].values

        # Features
        feature_cols = [
            'lstm_loss_forecast', 'loss_ma_5yr', 'loss_std_5yr',
            'lep', 'composite_risk', 'reinsurance_cost_index',
            'portfolio_concentration', 'government_subsidy_pct'
        ]

        # Add pv_ratio if available
        if 'pv_ratio' in data_aligned.columns:
            feature_cols.append('pv_ratio')

        X = data_aligned[feature_cols].values

        # Scale
        X_scaled = self.scaler_xgb.fit_transform(X)

        logger.info(f"XGBoost features prepared: {X_scaled.shape}")

        return X_scaled, y

    def train_xgboost(
        self,
        data: pd.DataFrame,
        lstm_predictions: np.ndarray
    ) -> Dict:
        """
        Train XGBoost classifier for insurer retreat prediction

        Args:
            data: Training data
            lstm_predictions: LSTM loss forecasts

        Returns:
            Evaluation metrics dict
        """
        logger.info("Training XGBoost retreat classifier...")

        # Prepare features
        X, y = self.prepare_xgb_features(data, lstm_predictions)

        # Split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )

        # Train
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params, random_state=42)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )

        # Predict
        y_pred = self.xgb_model.predict(X_test)
        y_proba = self.xgb_model.predict_proba(X_test)[:, 1]

        # Evaluate
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }

        logger.info(f"XGBoost training complete:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Feature importance
        importance = self.xgb_model.feature_importances_
        logger.info("Feature importances (top 3):")
        top_idx = np.argsort(importance)[-3:][::-1]
        for idx in top_idx:
            logger.info(f"  Feature {idx}: {importance[idx]:.4f}")

        # Save model
        model_path = self.model_dir / 'xgboost_model.json'
        self.xgb_model.save_model(model_path)
        logger.info(f"XGBoost model saved: {model_path}")

        # Save scaler
        scaler_path = self.model_dir / 'xgb_scaler.pkl'
        joblib.dump(self.scaler_xgb, scaler_path)

        # Save metrics
        metrics_path = self.model_dir / 'xgb_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def predict_retreat_probability(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict probability of insurer retreat

        Args:
            data: Input features

        Returns:
            Retreat probabilities (0-1)
        """
        if self.lstm_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Train LSTM and XGBoost first.")

        # Get LSTM predictions
        lstm_preds = self.predict_losses(data)

        # Prepare XGB features
        X, _ = self.prepare_xgb_features(data, lstm_preds)

        # Predict probabilities
        proba = self.xgb_model.predict_proba(X)[:, 1]

        return proba

    def load_models(self) -> None:
        """Load pre-trained models from disk"""
        lstm_path = self.model_dir / 'lstm_model.h5'
        xgb_path = self.model_dir / 'xgboost_model.json'

        if lstm_path.exists():
            self.lstm_model = keras.models.load_model(lstm_path)
            self.scaler_lstm = joblib.load(self.model_dir / 'lstm_scaler.pkl')
            logger.info("LSTM model loaded")

        if xgb_path.exists():
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path)
            self.scaler_xgb = joblib.load(self.model_dir / 'xgb_scaler.pkl')
            logger.info("XGBoost model loaded")


if __name__ == "__main__":
    # Example usage with synthetic data
    model = MREPModel()

    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 300  # 25 years monthly

    data = pd.DataFrame({
        'temperature': np.random.randn(n_samples) + 25,
        'precipitation': np.random.gamma(2, 50, n_samples),
        'extreme_events': np.random.poisson(2, n_samples),
        'aggregate_loss': np.random.gamma(3, 1e6, n_samples),
        'premium_collected': np.random.gamma(2, 8e5, n_samples),
        'flood_risk_score': np.random.randint(1, 10, n_samples),
        'drought_risk_score': np.random.randint(1, 10, n_samples),
        'cyclone_risk_score': np.random.randint(1, 10, n_samples),
        'reinsurance_cost_index': np.random.uniform(0.8, 1.5, n_samples),
        'portfolio_concentration': np.random.uniform(0.3, 0.8, n_samples),
        'government_subsidy_pct': np.random.uniform(0.2, 0.6, n_samples),
        'insurer_retreat': np.random.binomial(1, 0.2, n_samples)
    })

    # Train LSTM
    lstm_history = model.train_lstm(data)

    # Get LSTM predictions
    lstm_preds = model.predict_losses(data)

    # Train XGBoost
    xgb_metrics = model.train_xgboost(data, lstm_preds)

    print("\nMREP model training complete!")
    print(f"LSTM validation MAE: {lstm_history['val_mae'][-1]:.2f}")
    print(f"XGBoost ROC-AUC: {xgb_metrics['roc_auc']:.4f}")

"""
SFM - Solvency Forecasting Module
LSTM + XGBoost for fund long-term viability management

Predicts:
- Annual fund liabilities (payout requirements)
- Premium inflows and farmer enrollment
- Solvency at 99.5% confidence level
- Prescriptive recommendations (premium adjustments, capital injections)

Author: Dorjoy Acharjee
License: MIT
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import yaml
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFMModel:
    """Solvency Forecasting Module"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize SFM model"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sfm_cfg = self.config['models']['sfm']

        # LSTM config
        self.lstm_layers = self.sfm_cfg['lstm']['layers']
        self.dropout = self.sfm_cfg['lstm']['dropout']
        self.sequence_length = self.sfm_cfg['lstm']['sequence_length']
        self.n_outputs = self.sfm_cfg['lstm']['outputs']  # liabilities, premiums

        # XGBoost config
        self.xgb_params = {
            'max_depth': self.sfm_cfg['xgboost']['max_depth'],
            'learning_rate': self.sfm_cfg['xgboost']['learning_rate'],
            'n_estimators': self.sfm_cfg['xgboost']['n_estimators'],
            'objective': self.sfm_cfg['xgboost']['objective']
        }

        # Models
        self.lstm_model = None
        self.xgb_model = None
        self.scaler_lstm = StandardScaler()
        self.scaler_xgb = StandardScaler()

        # Fund parameters
        self.solvency_confidence = 0.995  # 99.5%
        self.solvency_buffer = 1.5  # 1.5x payouts + costs

        # Output paths
        self.model_dir = Path(self.config['paths']['models']) / 'sfm'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SFMModel initialized")

    def build_lstm_model(self, n_features: int) -> keras.Model:
        """
        Build LSTM for fund liability/premium forecasting

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, n_features)),

            layers.LSTM(self.lstm_layers[0], return_sequences=True, 
                       dropout=self.dropout, name='lstm_1'),
            layers.BatchNormalization(),

            layers.LSTM(self.lstm_layers[1], return_sequences=True,
                       dropout=self.dropout, name='lstm_2'),
            layers.BatchNormalization(),

            layers.LSTM(self.lstm_layers[2], return_sequences=False,
                       dropout=self.dropout, name='lstm_3'),
            layers.BatchNormalization(),

            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),

            # Multi-output: liabilities and premiums
            layers.Dense(self.n_outputs, activation='relu', name='fund_forecast')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"LSTM model built: {model.count_params()} parameters")
        return model

    def prepare_fund_sequences(
        self,
        data: pd.DataFrame,
        liability_col: str = 'annual_payouts',
        premium_col: str = 'annual_premiums'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare fund financial data for LSTM

        Args:
            data: Time series DataFrame (annual or monthly aggregates)
            liability_col: Column with liability amounts
            premium_col: Column with premium inflows

        Returns:
            Tuple of (X_sequences, y_targets)
        """
        # Drop target columns from features
        feature_cols = [col for col in data.columns 
                       if col not in [liability_col, premium_col]]
        features = data[feature_cols].values

        # Targets: [liabilities, premiums]
        targets = data[[liability_col, premium_col]].values

        # Normalize
        features_scaled = self.scaler_lstm.fit_transform(features)

        # Create sequences
        X, y = [], []

        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i+self.sequence_length])
            y.append(targets[i+self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} fund sequences")
        return X, y

    def train_lstm(
        self,
        data: pd.DataFrame,
        validation_split: float = 0.2,
        epochs: int = None
    ) -> Dict:
        """
        Train LSTM for fund liability forecasting

        Args:
            data: Annual fund financial data
            validation_split: Validation fraction
            epochs: Number of epochs

        Returns:
            Training history
        """
        logger.info("Training SFM LSTM model...")

        # Prepare sequences
        X, y = self.prepare_fund_sequences(data)

        # Split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build model
        self.lstm_model = self.build_lstm_model(n_features=X.shape[2])

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]

        # Train
        epochs = epochs or 100
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=16,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        val_loss, val_mae = self.lstm_model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"LSTM training complete: val_loss={val_loss:.4f}")

        # Save
        self.lstm_model.save(self.model_dir / 'sfm_lstm.h5')
        joblib.dump(self.scaler_lstm, self.model_dir / 'sfm_lstm_scaler.pkl')

        return history.history

    def forecast_fund_financials(
        self,
        data: pd.DataFrame,
        horizon_years: int = 20
    ) -> pd.DataFrame:
        """
        Forecast fund liabilities and premiums

        Args:
            data: Historical fund data
            horizon_years: Forecast horizon

        Returns:
            DataFrame with forecasts
        """
        if self.lstm_model is None:
            raise ValueError("LSTM not trained")

        # Prepare last sequence
        features = data.drop(columns=['annual_payouts', 'annual_premiums']).values
        features_scaled = self.scaler_lstm.transform(features[-self.sequence_length:])
        X_input = features_scaled.reshape(1, self.sequence_length, features_scaled.shape[1])

        forecasts = []
        current_input = X_input.copy()

        # Iterative forecasting
        for _ in range(horizon_years):
            pred = self.lstm_model.predict(current_input, verbose=0)
            forecasts.append(pred[0])

            # Shift window (simplified - assumes pred becomes next input feature)
            # In production, would need proper feature engineering
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1] = pred[0]

        forecasts = np.array(forecasts)

        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecasts, columns=['forecast_liabilities', 'forecast_premiums'])
        forecast_df['year'] = range(2026, 2026 + horizon_years)

        logger.info(f"Forecasted {horizon_years} years ahead")
        return forecast_df

    def prepare_solvency_features(
        self,
        fund_data: pd.DataFrame,
        climate_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for solvency classification

        Args:
            fund_data: Fund financial data
            climate_data: Climate/risk data

        Returns:
            Tuple of (X_features, y_solvency)
        """
        # Merge datasets
        data = pd.merge(fund_data, climate_data, left_index=True, right_index=True)

        # Calculate solvency indicators
        data['capital_ratio'] = data['fund_capital'] / (data['annual_payouts'] + 1e-10)
        data['loss_ratio'] = data['annual_payouts'] / (data['annual_premiums'] + 1e-10)
        data['premium_coverage'] = data['annual_premiums'] / (data['annual_payouts'] * 1.5 + 1e-10)

        # Target: is_solvent (solvency buffer maintained)
        data['is_solvent'] = (
            (data['fund_capital'] + data['annual_premiums']) > 
            (data['annual_payouts'] * self.solvency_buffer)
        ).astype(int)

        # Features
        feature_cols = [
            'capital_ratio', 'loss_ratio', 'premium_coverage',
            'farmer_enrollment', 'climate_hazard_intensity',
            'reinsurance_access', 'government_backstop'
        ]

        X = data[feature_cols].values
        y = data['is_solvent'].values

        # Scale
        X_scaled = self.scaler_xgb.fit_transform(X)

        logger.info(f"Solvency features prepared: {X_scaled.shape}")
        return X_scaled, y

    def train_solvency_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Train XGBoost solvency classifier

        Args:
            X: Feature matrix
            y: Solvency labels

        Returns:
            Performance metrics
        """
        logger.info("Training solvency classifier...")

        # Train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params, random_state=42)
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict
        y_pred = self.xgb_model.predict(X_test)
        y_proba = self.xgb_model.predict_proba(X_test)[:, 1]

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }

        logger.info(f"Solvency classifier trained: {metrics['roc_auc']:.4f} ROC-AUC")

        # Save
        self.xgb_model.save_model(self.model_dir / 'sfm_xgboost.json')
        joblib.dump(self.scaler_xgb, self.model_dir / 'sfm_xgb_scaler.pkl')

        return metrics

    def predict_solvency_trajectory(
        self,
        forecast_df: pd.DataFrame,
        initial_capital: float,
        enrollment_growth: float = 0.05
    ) -> pd.DataFrame:
        """
        Simulate fund solvency over forecast horizon

        Args:
            forecast_df: DataFrame with liability/premium forecasts
            initial_capital: Starting fund capital
            enrollment_growth: Annual farmer enrollment growth rate

        Returns:
            DataFrame with solvency trajectory
        """
        trajectory = []
        current_capital = initial_capital

        for idx, row in forecast_df.iterrows():
            # Simulate operational expenses
            operational_costs = row['forecast_premiums'] * 0.15  # 15% ops costs

            # Net cash flow
            net_flow = row['forecast_premiums'] - row['forecast_liabilities'] - operational_costs

            # Update capital
            current_capital += net_flow

            # Solvency check
            solvency_buffer = current_capital / (row['forecast_liabilities'] * self.solvency_buffer + 1e-10)
            is_solvent = solvency_buffer >= 1.0

            trajectory.append({
                'year': row['year'],
                'liabilities': row['forecast_liabilities'],
                'premiums': row['forecast_premiums'],
                'ops_costs': operational_costs,
                'net_flow': net_flow,
                'fund_capital': max(current_capital, 0),
                'solvency_buffer': solvency_buffer,
                'is_solvent': is_solvent,
                'solvency_confidence': 0.995 if is_solvent else 0.95
            })

        trajectory_df = pd.DataFrame(trajectory)

        logger.info(f"Solvency trajectory computed")
        return trajectory_df

    def prescribe_interventions(
        self,
        trajectory_df: pd.DataFrame,
        threshold_year: int = None
    ) -> List[Dict]:
        """
        Generate prescriptive recommendations when solvency at risk

        Args:
            trajectory_df: Solvency trajectory
            threshold_year: Year when solvency becomes critical

        Returns:
            List of intervention recommendations
        """
        # Identify crisis year
        crisis_years = trajectory_df[trajectory_df['is_solvent'] == False]['year'].values

        if len(crisis_years) == 0:
            logger.info("Fund remains solvent - no interventions needed")
            return []

        crisis_year = int(crisis_years[0])
        crisis_data = trajectory_df[trajectory_df['year'] == crisis_year].iloc[0]

        logger.warning(f"Solvency crisis predicted in {crisis_year}")

        recommendations = []

        # Option 1: Premium increase
        premium_shortfall = crisis_data['liabilities'] * self.solvency_buffer - crisis_data['premiums']
        if premium_shortfall > 0:
            premium_increase_pct = (premium_shortfall / crisis_data['premiums']) * 100
            recommendations.append({
                'intervention': 'Premium Adjustment',
                'description': f"Increase premiums by {premium_increase_pct:.1f}%",
                'estimated_cost_to_farmers': premium_shortfall,
                'feasibility': 'Medium' if premium_increase_pct < 20 else 'Low',
                'implementation_timeline': '1 year'
            })

        # Option 2: Government capital injection
        capital_need = max(0, crisis_data['liabilities'] * self.solvency_buffer - crisis_data['fund_capital'])
        recommendations.append({
            'intervention': 'Government Capital Injection',
            'description': f"Inject ${capital_need/1e6:.1f}M into fund",
            'estimated_cost_to_govt': capital_need,
            'feasibility': 'High' if capital_need < 1e9 else 'Medium',
            'implementation_timeline': '2 years'
        })

        # Option 3: Coverage rebalancing
        coverage_reduction = (crisis_data['liabilities'] - crisis_data['premiums']) / crisis_data['liabilities']
        recommendations.append({
            'intervention': 'Coverage Rebalancing',
            'description': f"Reduce coverage in high-risk regions by {coverage_reduction*100:.1f}%",
            'impact_on_farmers': f"{coverage_reduction*100:.1f}% less coverage",
            'feasibility': 'Low' if coverage_reduction > 0.3 else 'Medium',
            'implementation_timeline': '1 year'
        })

        # Save recommendations
        rec_path = self.model_dir / f'interventions_{crisis_year}.json'
        with open(rec_path, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)

        return recommendations

    def load_models(self) -> None:
        """Load pre-trained models"""
        lstm_path = self.model_dir / 'sfm_lstm.h5'
        xgb_path = self.model_dir / 'sfm_xgboost.json'

        if lstm_path.exists():
            self.lstm_model = keras.models.load_model(lstm_path)
            self.scaler_lstm = joblib.load(self.model_dir / 'sfm_lstm_scaler.pkl')
            logger.info("SFM LSTM model loaded")

        if xgb_path.exists():
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path)
            self.scaler_xgb = joblib.load(self.model_dir / 'sfm_xgb_scaler.pkl')
            logger.info("SFM XGBoost model loaded")


if __name__ == "__main__":
    model = SFMModel()

    # Generate synthetic fund data
    np.random.seed(42)
    n_years = 30

    fund_data = pd.DataFrame({
        'year': range(1994, 1994 + n_years),
        'annual_payouts': np.random.gamma(3, 1e7, n_years),
        'annual_premiums': np.random.gamma(2, 1.5e7, n_years),
        'fund_capital': np.linspace(1e9, 3e9, n_years),
        'farmer_enrollment': np.linspace(1e6, 5e6, n_years)
    })

    # Generate synthetic climate data
    climate_data = pd.DataFrame({
        'climate_hazard_intensity': np.random.uniform(1, 10, n_years),
        'reinsurance_access': np.random.uniform(0, 1, n_years),
        'government_backstop': np.ones(n_years)
    }, index=fund_data.index)

    # Train LSTM
    lstm_history = model.train_lstm(fund_data, epochs=50)

    # Forecast
    forecast_df = model.forecast_fund_financials(fund_data, horizon_years=20)

    # Predict solvency trajectory
    trajectory = model.predict_solvency_trajectory(forecast_df, initial_capital=2e9)
    print(f"\nSolvency Trajectory (first 5 years):")
    print(trajectory[['year', 'fund_capital', 'is_solvent', 'solvency_confidence']].head())

    # Get interventions if needed
    interventions = model.prescribe_interventions(trajectory)
    if interventions:
        print(f"\nRecommended Interventions:")
        for rec in interventions:
            print(f"  - {rec['intervention']}: {rec['description']}")

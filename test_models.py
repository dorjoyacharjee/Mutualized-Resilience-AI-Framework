"""
Unit Tests for AI Models
Tests for MREP, HAM, and SFM modules

Run with: pytest tests/test_models.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.mrep_model import MREPModel
from models.ham_model import HAMModel
from models.sfm_model import SFMModel


class TestMREPModel:
    """Tests for MREP model"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'temperature': np.random.randn(n) + 25,
            'precipitation': np.random.gamma(2, 50, n),
            'aggregate_loss': np.random.gamma(3, 1e6, n),
            'premium_collected': np.random.gamma(2, 8e5, n),
            'flood_risk_score': np.random.randint(1, 10, n),
            'drought_risk_score': np.random.randint(1, 10, n),
            'cyclone_risk_score': np.random.randint(1, 10, n),
            'reinsurance_cost_index': np.random.uniform(0.8, 1.5, n),
            'portfolio_concentration': np.random.uniform(0.3, 0.8, n),
            'government_subsidy_pct': np.random.uniform(0.2, 0.6, n),
            'insurer_retreat': np.random.binomial(1, 0.2, n)
        })
        return data

    def test_initialization(self):
        """Test model initialization"""
        model = MREPModel()
        assert model.lstm_model is None
        assert model.xgb_model is None
        assert model.sequence_length == 60

    def test_sequence_preparation(self, sample_data):
        """Test LSTM sequence preparation"""
        model = MREPModel()
        X, y = model.prepare_lstm_sequences(sample_data)

        assert X.shape[0] == len(sample_data) - model.sequence_length
        assert X.shape[1] == model.sequence_length
        assert len(y) == len(X)

    def test_lstm_build(self):
        """Test LSTM model architecture"""
        model = MREPModel()
        lstm = model.build_lstm_model(n_features=5)

        assert lstm is not None
        assert len(lstm.layers) > 0
        assert lstm.output_shape[-1] == 1

    def test_predict_losses(self, sample_data):
        """Test loss prediction (requires trained model)"""
        model = MREPModel()

        # Should raise error if model not trained
        with pytest.raises(ValueError):
            model.predict_losses(sample_data)


class TestHAMModel:
    """Tests for HAM U-Net CNN"""

    @pytest.fixture
    def sample_images(self):
        """Generate sample satellite images"""
        n_samples = 10
        img_size = 64
        n_channels = 3
        n_classes = 3

        X = np.random.rand(n_samples, img_size, img_size, n_channels).astype(np.float32)
        y = np.random.randint(0, n_classes, (n_samples, img_size, img_size))

        import tensorflow as tf
        y = tf.keras.utils.to_categorical(y, num_classes=n_classes)

        return X, y

    def test_initialization(self):
        """Test HAM initialization"""
        model = HAMModel()
        assert model.model is None
        assert model.input_shape == (64, 64, 3)
        assert model.n_classes == 3

    def test_unet_build(self):
        """Test U-Net architecture"""
        model = HAMModel()
        unet = model.build_unet()

        assert unet is not None
        assert unet.input_shape[1:] == model.input_shape
        assert unet.output_shape[-1] == model.n_classes

    def test_damage_calculation(self, sample_images):
        """Test damage percentage calculation"""
        model = HAMModel()
        X, y = sample_images

        # Use ground truth as "prediction"
        damage_pct = model.calculate_damage_percentage(y[0])

        assert 0 <= damage_pct <= 100

    def test_payout_trigger(self, sample_images):
        """Test automated payout logic"""
        model = HAMModel()
        X, y = sample_images

        # Test payout trigger
        trigger, damage = model.trigger_payout([y[0], y[1]], threshold=20.0)

        assert isinstance(trigger, bool)
        assert 0 <= damage <= 100


class TestSFMModel:
    """Tests for SFM solvency forecasting"""

    @pytest.fixture
    def sample_fund_data(self):
        """Generate sample fund financial data"""
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({
            'year': range(2000, 2000 + n),
            'annual_payouts': np.random.gamma(3, 1e7, n),
            'annual_premiums': np.random.gamma(2, 1.5e7, n),
            'fund_capital': np.linspace(1e9, 3e9, n),
            'farmer_enrollment': np.linspace(1e6, 5e6, n),
            'climate_hazard_intensity': np.random.uniform(1, 10, n),
            'reinsurance_access': np.random.uniform(0, 1, n),
            'government_backstop': np.ones(n)
        })
        return data

    def test_initialization(self):
        """Test SFM initialization"""
        model = SFMModel()
        assert model.lstm_model is None
        assert model.xgb_model is None
        assert model.solvency_confidence == 0.995

    def test_sequence_preparation(self, sample_fund_data):
        """Test fund sequence preparation"""
        model = SFMModel()
        X, y = model.prepare_fund_sequences(sample_fund_data)

        assert X.shape[0] == len(sample_fund_data) - model.sequence_length
        assert y.shape[1] == 2  # liabilities and premiums

    def test_solvency_trajectory(self, sample_fund_data):
        """Test solvency trajectory computation"""
        model = SFMModel()

        # Create simple forecast
        forecast_df = pd.DataFrame({
            'year': range(2025, 2030),
            'forecast_liabilities': np.random.gamma(3, 1e7, 5),
            'forecast_premiums': np.random.gamma(2, 1.5e7, 5)
        })

        trajectory = model.predict_solvency_trajectory(
            forecast_df, initial_capital=2e9
        )

        assert len(trajectory) == 5
        assert 'is_solvent' in trajectory.columns
        assert 'solvency_buffer' in trajectory.columns


class TestIntegration:
    """Integration tests across modules"""

    def test_pipeline_flow(self):
        """Test basic pipeline flow"""
        # This would test data flow between modules
        # Simplified for demonstration
        assert True

    def test_config_loading(self):
        """Test configuration loading"""
        from pathlib import Path
        import yaml

        config_path = Path('config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            assert 'models' in config
            assert 'data_sources' in config


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])

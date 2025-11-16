"""
HAM - Hazard Assessment Module
U-Net CNN for satellite-based crop damage detection

Processes:
- Sentinel-1 SAR for flood detection
- Sentinel-2 optical for drought detection
- Pixel-wise segmentation (healthy / flood-damaged / drought-stressed)

Author: Research Team
License: MIT
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
import yaml
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HAMModel:
    """Hazard Assessment Module - U-Net CNN"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize HAM model"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.ham_cfg = self.config['models']['ham']

        # Architecture params
        self.input_shape = tuple(self.ham_cfg['input_shape'])
        self.n_classes = self.ham_cfg['output_classes']
        self.encoder_filters = self.ham_cfg['encoder_filters']
        self.decoder_filters = self.ham_cfg['decoder_filters']

        # Training params
        self.batch_size = self.ham_cfg['batch_size']
        self.epochs = self.ham_cfg['epochs']
        self.lr = self.ham_cfg['learning_rate']

        # Model
        self.model = None

        # Output paths
        self.model_dir = Path(self.config['paths']['models']) / 'ham'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("HAMModel initialized")

    def build_unet(self) -> keras.Model:
        """
        Build U-Net architecture for semantic segmentation

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)

        # Encoder (contracting path)
        skip_connections = []
        x = inputs

        for i, filters in enumerate(self.encoder_filters):
            # Conv block
            x = layers.Conv2D(filters, 3, padding='same', activation='relu', 
                            name=f'enc_conv1_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                            name=f'enc_conv2_{i}')(x)
            x = layers.BatchNormalization()(x)

            # Save for skip connection
            skip_connections.append(x)

            # Downsample
            x = layers.MaxPooling2D(2, name=f'enc_pool_{i}')(x)
            x = layers.Dropout(0.2)(x)

        # Bottleneck
        x = layers.Conv2D(self.encoder_filters[-1] * 2, 3, padding='same', 
                        activation='relu', name='bottleneck_conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(self.encoder_filters[-1] * 2, 3, padding='same',
                        activation='relu', name='bottleneck_conv2')(x)
        x = layers.BatchNormalization()(x)

        # Decoder (expanding path)
        for i, filters in enumerate(self.decoder_filters):
            # Upsample
            x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same',
                                      name=f'dec_upconv_{i}')(x)

            # Skip connection
            skip = skip_connections[-(i+1)]
            x = layers.Concatenate(name=f'dec_concat_{i}')([x, skip])

            # Conv block
            x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                            name=f'dec_conv1_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                            name=f'dec_conv2_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Output layer
        outputs = layers.Conv2D(
            self.n_classes, 1, 
            activation='softmax' if self.n_classes > 1 else 'sigmoid',
            name='output'
        )(x)

        # Build model
        model = keras.Model(inputs=inputs, outputs=outputs, name='HAM_UNet')

        # Compile
        loss = (keras.losses.CategoricalCrossentropy() if self.n_classes > 1 
                else keras.losses.BinaryCrossentropy())

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.MeanIoU(num_classes=self.n_classes),
                self._dice_coef
            ]
        )

        logger.info(f"U-Net model built: {model.count_params()} parameters")
        model.summary(print_fn=logger.info)

        return model

    @staticmethod
    def _dice_coef(y_true, y_pred, smooth=1.0):
        """Dice coefficient metric"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
        )

    def load_tiles_from_dir(
        self,
        tiles_dir: str,
        labels_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load pre-processed image tiles

        Args:
            tiles_dir: Directory containing .npy tile files
            labels_dir: Optional directory with corresponding labels

        Returns:
            Tuple of (images, labels) or (images, None)
        """
        tiles_path = Path(tiles_dir)
        tile_files = sorted(tiles_path.glob('*.npy'))

        images = []
        for tile_file in tile_files:
            img = np.load(tile_file)
            images.append(img)

        images = np.array(images)
        logger.info(f"Loaded {len(images)} tiles from {tiles_dir}")

        # Load labels if provided
        labels = None
        if labels_dir is not None:
            labels_path = Path(labels_dir)
            label_files = sorted(labels_path.glob('*.npy'))

            labels_list = []
            for label_file in label_files:
                label = np.load(label_file)
                labels_list.append(label)

            labels = np.array(labels_list)
            logger.info(f"Loaded {len(labels)} labels from {labels_dir}")

            # Convert to categorical if multi-class
            if self.n_classes > 1:
                labels = keras.utils.to_categorical(labels, num_classes=self.n_classes)

        return images, labels

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.15,
        class_weights: Optional[Dict] = None
    ) -> Dict:
        """
        Train HAM U-Net model

        Args:
            X_train: Training images (N, H, W, C)
            y_train: Training labels (N, H, W, Classes)
            validation_split: Fraction for validation
            class_weights: Optional class weights for imbalanced data

        Returns:
            Training history dict
        """
        logger.info(f"Training HAM model on {len(X_train)} samples...")

        # Build model
        self.model = self.build_unet()

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'ham_best.h5'),
                monitor='val_mean_io_u',
                save_best_only=True,
                mode='max'
            )
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Evaluate on validation
        val_loss, val_acc, val_iou, val_dice = self.model.evaluate(
            X_val, y_val, verbose=0
        )

        logger.info(f"Training complete:")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Accuracy: {val_acc:.4f}")
        logger.info(f"  Val IoU: {val_iou:.4f}")
        logger.info(f"  Val Dice: {val_dice:.4f}")

        # Save final model
        model_path = self.model_dir / 'ham_final.h5'
        self.model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Save training history
        history_path = self.model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] 
                      for k, vals in history.history.items()}, f, indent=2)

        return history.history

    def predict(
        self,
        images: np.ndarray,
        batch_size: int = None
    ) -> np.ndarray:
        """
        Predict damage segmentation masks

        Args:
            images: Input images (N, H, W, C)
            batch_size: Batch size for prediction

        Returns:
            Predicted masks (N, H, W, Classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load_model().")

        batch_size = batch_size or self.batch_size

        predictions = self.model.predict(images, batch_size=batch_size, verbose=0)

        return predictions

    def calculate_damage_percentage(
        self,
        prediction: np.ndarray,
        damage_classes: List[int] = [1, 2]
    ) -> float:
        """
        Calculate percentage of damaged area

        Args:
            prediction: Predicted mask (H, W, Classes)
            damage_classes: Class indices considered as "damaged"

        Returns:
            Damage percentage (0-100)
        """
        # Get class with max probability for each pixel
        class_map = np.argmax(prediction, axis=-1)

        # Count damaged pixels
        damaged_pixels = np.isin(class_map, damage_classes).sum()
        total_pixels = class_map.size

        damage_pct = (damaged_pixels / total_pixels) * 100

        return damage_pct

    def trigger_payout(
        self,
        cooperative_predictions: List[np.ndarray],
        threshold: float = 20.0
    ) -> Tuple[bool, float]:
        """
        Determine if cooperative qualifies for payout

        Args:
            cooperative_predictions: List of prediction masks for cooperative's farmland
            threshold: Damage threshold (%) for payout trigger

        Returns:
            Tuple of (trigger_payout, damage_percentage)
        """
        # Calculate damage for all tiles
        damage_percentages = [
            self.calculate_damage_percentage(pred) 
            for pred in cooperative_predictions
        ]

        # Average across cooperative
        avg_damage = np.mean(damage_percentages)

        # Trigger if exceeds threshold
        trigger = avg_damage >= threshold

        if trigger:
            logger.info(f"PAYOUT TRIGGERED: {avg_damage:.2f}% damage (threshold: {threshold}%)")
        else:
            logger.info(f"No payout: {avg_damage:.2f}% damage (threshold: {threshold}%)")

        return trigger, avg_damage

    def evaluate_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Comprehensive evaluation on test set

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)

        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating on test set...")

        # Predict
        predictions = self.predict(X_test)

        # Get class indices
        y_true_classes = np.argmax(y_test, axis=-1).flatten()
        y_pred_classes = np.argmax(predictions, axis=-1).flatten()

        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Classification report
        report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=['Healthy', 'Flood-damaged', 'Drought-stressed'][:self.n_classes],
            output_dict=True
        )

        # Pixel-wise metrics
        test_loss, test_acc, test_iou, test_dice = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_iou': float(test_iou),
            'test_dice': float(test_dice),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        logger.info(f"Test metrics:")
        logger.info(f"  Accuracy: {test_acc:.4f}")
        logger.info(f"  IoU: {test_iou:.4f}")
        logger.info(f"  Dice: {test_dice:.4f}")

        # Save metrics
        metrics_path = self.model_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load pre-trained model"""
        if model_path is None:
            model_path = self.model_dir / 'ham_best.h5'
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = keras.models.load_model(
            model_path,
            custom_objects={'_dice_coef': self._dice_coef}
        )
        logger.info(f"Model loaded: {model_path}")


if __name__ == "__main__":
    # Example usage with synthetic data
    model = HAMModel()

    # Generate synthetic training data (replace with real satellite tiles)
    n_samples = 100
    img_size = 64
    n_channels = 3
    n_classes = 3

    X_train = np.random.rand(n_samples, img_size, img_size, n_channels).astype(np.float32)
    y_train = np.random.randint(0, n_classes, (n_samples, img_size, img_size))
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)

    # Train
    history = model.train(X_train, y_train, validation_split=0.2)

    # Test prediction
    X_test = np.random.rand(10, img_size, img_size, n_channels).astype(np.float32)
    predictions = model.predict(X_test)

    # Calculate damage
    damage_pct = model.calculate_damage_percentage(predictions[0])
    print(f"\nSample damage: {damage_pct:.2f}%")

    # Test payout trigger
    trigger, avg_damage = model.trigger_payout(
        predictions[:5], threshold=20.0
    )
    print(f"Payout triggered: {trigger}, Average damage: {avg_damage:.2f}%")

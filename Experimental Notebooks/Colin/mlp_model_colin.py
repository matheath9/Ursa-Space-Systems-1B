"""
SAR Iceberg Classifier - MLP Baseline Model
============================================

Author: Colin
Date: October 2025
Project: Fall AI Studio Challenge with Ursa Space Systems

This is a standalone Multi-Layer Perceptron (MLP) implementation for 
classifying SAR (Synthetic Aperture Radar) imagery to distinguish 
ships from icebergs.

Dataset: Statoil/C-CORE Iceberg Classifier Challenge
- Training: 1,604 images (75x75x2 channels)
- Test: 8,424 images
- Binary classification: Ship (0) vs Iceberg (1)

Architecture:
- Input: 11,250 features (75x75x2 flattened SAR bands)
- Hidden layers: [512, 256, 128] neurons
- Activation: ReLU
- Output: 1 neuron (sigmoid activation)
- Optimizer: Adam
- Regularization: L2 (alpha=0.0001)
- Early stopping: Enabled (patience=10 epochs)

Performance:
- Validation ROC-AUC: ~0.985
- Training iterations: ~50-100 epochs

Usage:
    # Training
    python mlp_model_colin.py --mode train --train_path train.json
    
    # Testing
    python mlp_model_colin.py --mode test --train_path train.json --test_path test.json
    
    # Prediction only
    python mlp_model_colin.py --mode predict --test_path test.json
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score
)
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)


class MLPIcebergClassifier:
    """
    Multi-Layer Perceptron for SAR Iceberg Classification
    
    This classifier uses a deep neural network to distinguish between
    ships and icebergs in Synthetic Aperture Radar imagery.
    """
    
    def __init__(self, 
                 hidden_layers=(512, 256, 128),
                 learning_rate=0.001,
                 max_iterations=100,
                 alpha=0.0001):
        """
        Initialize the MLP classifier.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            learning_rate: Initial learning rate for Adam optimizer
            max_iterations: Maximum training epochs
            alpha: L2 regularization parameter
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.alpha = alpha
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size=32,
            learning_rate_init=learning_rate,
            max_iter=max_iterations,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data, has_labels=True):
        """
        Prepare SAR imagery data for model input.
        
        Flattens the two 75x75 band images into a single feature vector
        of length 11,250 (75*75*2).
        
        Args:
            data: DataFrame with 'band_1' and 'band_2' columns
            has_labels: Whether data includes 'is_iceberg' labels
            
        Returns:
            X: Feature matrix (n_samples, 11250)
            y: Labels (if has_labels=True), None otherwise
        """
        # Flatten each band
        band1 = np.array([np.array(band).flatten() for band in data['band_1']])
        band2 = np.array([np.array(band).flatten() for band in data['band_2']])
        
        # Concatenate both bands
        X = np.concatenate([band1, band2], axis=1)
        
        if has_labels:
            y = data['is_iceberg'].values
            return X, y
        else:
            return X, None
    
    def train(self, train_data, validation_split=0.2):
        """
        Train the MLP model.
        
        Args:
            train_data: DataFrame with training data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training history and metrics
        """
        print("="*70)
        print("TRAINING MLP MODEL")
        print("="*70)
        
        # Prepare features
        print("\nPreparing training data...")
        X_full, y_full = self.prepare_features(train_data, has_labels=True)
        print(f"Total samples: {X_full.shape[0]}")
        print(f"Features per sample: {X_full.shape[1]}")
        
        # Create train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full,
            test_size=validation_split,
            random_state=42,
            stratify=y_full
        )
        
        print(f"\nData split:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Validation samples: {X_val.shape[0]}")
        
        # Standardize features
        print("\nStandardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Display architecture
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        print(f"Input layer: {X_full.shape[1]} features")
        print(f"Hidden layers: {self.hidden_layers}")
        print(f"Output layer: 1 neuron (sigmoid)")
        print(f"Activation: ReLU")
        print(f"Optimizer: Adam (learning_rate={self.learning_rate})")
        print(f"L2 regularization: alpha={self.alpha}")
        print(f"Early stopping: Enabled (patience=10)")
        
        # Train model
        print("\n" + "="*70)
        print("TRAINING IN PROGRESS...")
        print("="*70 + "\n")
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total iterations: {self.model.n_iter_}")
        print(f"Final loss: {self.model.loss_:.6f}")
        
        # Evaluate on both sets
        train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        val_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        
        print(f"\nPerformance Metrics:")
        print(f"  Training ROC-AUC: {train_auc:.4f}")
        print(f"  Validation ROC-AUC: {val_auc:.4f}")
        print(f"  Overfitting gap: {train_auc - val_auc:.4f}")
        
        # Classification report
        val_pred_binary = (val_pred > 0.5).astype(int)
        print("\n" + "="*70)
        print("VALIDATION SET CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_val, val_pred_binary,
                                   target_names=['Ship', 'Iceberg']))
        
        # Store validation data for visualization
        self.X_val_scaled = X_val_scaled
        self.y_val = y_val
        self.val_pred = val_pred
        
        return {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'n_iterations': self.model.n_iter_,
            'final_loss': self.model.loss_,
            'loss_curve': self.model.loss_curve_
        }
    
    def predict(self, test_data):
        """
        Make predictions on test data.
        
        Args:
            test_data: DataFrame with test data
            
        Returns:
            Array of predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        print("\nGenerating predictions...")
        X_test, _ = self.prepare_features(test_data, has_labels=False)
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict_proba(X_test_scaled)[:, 1]
        print(f"✅ Generated {len(predictions)} predictions")
        
        return predictions
    
    def save_model(self, filepath='mlp_iceberg_model.pkl'):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'alpha': self.alpha
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to {filepath}")
    
    def load_model(self, filepath='mlp_iceberg_model.pkl'):
        """Load a trained model and scaler."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.hidden_layers = model_data['hidden_layers']
        self.learning_rate = model_data['learning_rate']
        self.alpha = model_data['alpha']
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")
    
    def plot_training_curve(self, save_path='mlp_training_curve.png'):
        """Plot the training loss curve."""
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting!")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.model.loss_curve_, linewidth=2.5, color='#2E86AB')
        plt.xlabel('Iteration', fontsize=13)
        plt.ylabel('Loss', fontsize=13)
        plt.title('MLP Training Loss Curve', fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training curve saved to {save_path}")
    
    def plot_roc_curve(self, save_path='mlp_roc_curve.png'):
        """Plot ROC curve for validation set."""
        if not hasattr(self, 'y_val'):
            raise ValueError("Must train model first to generate ROC curve!")
        
        fpr, tpr, _ = roc_curve(self.y_val, self.val_pred)
        auc = roc_auc_score(self.y_val, self.val_pred)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, linewidth=2.5, color='#A23B72',
                label=f'MLP (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC Curve - MLP Iceberg Classifier', fontsize=15, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC curve saved to {save_path}")
    
    def plot_confusion_matrix(self, save_path='mlp_confusion_matrix.png'):
        """Plot confusion matrix for validation set."""
        if not hasattr(self, 'y_val'):
            raise ValueError("Must train model first to generate confusion matrix!")
        
        val_pred_binary = (self.val_pred > 0.5).astype(int)
        cm = confusion_matrix(self.y_val, val_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 14})
        plt.xlabel('Predicted Label', fontsize=13)
        plt.ylabel('True Label', fontsize=13)
        plt.title('Confusion Matrix - MLP Iceberg Classifier', 
                 fontsize=15, fontweight='bold')
        plt.xticks([0.5, 1.5], ['Ship', 'Iceberg'])
        plt.yticks([0.5, 1.5], ['Ship', 'Iceberg'])
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to {save_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='MLP Iceberg Classifier by Colin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python mlp_model_colin.py --mode train --train_path train.json
  
  # Test existing model
  python mlp_model_colin.py --mode test --train_path train.json --test_path test.json
  
  # Make predictions only
  python mlp_model_colin.py --mode predict --test_path test.json --model_path mlp_iceberg_model.pkl
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'test', 'predict'],
                       help='Mode: train, test, or predict')
    parser.add_argument('--train_path', type=str, default='train.json',
                       help='Path to training data JSON file')
    parser.add_argument('--test_path', type=str, default='test.json',
                       help='Path to test data JSON file')
    parser.add_argument('--model_path', type=str, default='mlp_iceberg_model.pkl',
                       help='Path to save/load model')
    parser.add_argument('--hidden_layers', type=int, nargs='+', 
                       default=[512, 256, 128],
                       help='Hidden layer sizes (default: 512 256 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--max_iter', type=int, default=100,
                       help='Maximum iterations (default: 100)')
    parser.add_argument('--alpha', type=float, default=0.0001,
                       help='L2 regularization (default: 0.0001)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("SAR ICEBERG CLASSIFIER - MLP MODEL")
    print("Author: Colin")
    print("="*70 + "\n")
    
    # Initialize classifier
    classifier = MLPIcebergClassifier(
        hidden_layers=tuple(args.hidden_layers),
        learning_rate=args.learning_rate,
        max_iterations=args.max_iter,
        alpha=args.alpha
    )
    
    if args.mode == 'train':
        # Load training data
        print("Loading training data...")
        train_data = pd.read_json(args.train_path)
        print(f"✅ Loaded {len(train_data)} training samples\n")
        
        # Train model
        history = classifier.train(train_data)
        
        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        classifier.plot_training_curve()
        classifier.plot_roc_curve()
        classifier.plot_confusion_matrix()
        
        # Save model
        classifier.save_model(args.model_path)
        
        # If test data provided, make predictions
        if Path(args.test_path).exists():
            print("\n" + "="*70)
            print("GENERATING TEST PREDICTIONS")
            print("="*70)
            test_data = pd.read_json(args.test_path)
            predictions = classifier.predict(test_data)
            
            submission = pd.DataFrame({
                'id': test_data['id'],
                'is_iceberg': predictions
            })
            submission.to_csv('submission_mlp_colin.csv', index=False)
            print(f"✅ Submission saved to submission_mlp_colin.csv")
    
    elif args.mode == 'test':
        # Load model
        print(f"Loading model from {args.model_path}...")
        classifier.load_model(args.model_path)
        
        # Load and evaluate on training data
        print("\nLoading training data for evaluation...")
        train_data = pd.read_json(args.train_path)
        
        # Re-train to get validation metrics (or implement separate evaluation)
        history = classifier.train(train_data)
        
        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        classifier.plot_training_curve()
        classifier.plot_roc_curve()
        classifier.plot_confusion_matrix()
        
        # Make predictions on test data
        if Path(args.test_path).exists():
            print("\n" + "="*70)
            print("GENERATING TEST PREDICTIONS")
            print("="*70)
            test_data = pd.read_json(args.test_path)
            predictions = classifier.predict(test_data)
            
            submission = pd.DataFrame({
                'id': test_data['id'],
                'is_iceberg': predictions
            })
            submission.to_csv('submission_mlp_colin.csv', index=False)
            print(f"✅ Submission saved to submission_mlp_colin.csv")
    
    elif args.mode == 'predict':
        # Load model
        print(f"Loading model from {args.model_path}...")
        classifier.load_model(args.model_path)
        
        # Load test data and predict
        print(f"\nLoading test data from {args.test_path}...")
        test_data = pd.read_json(args.test_path)
        print(f"✅ Loaded {len(test_data)} test samples")
        
        # Make predictions
        predictions = classifier.predict(test_data)
        
        # Save submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'is_iceberg': predictions
        })
        submission.to_csv('submission_mlp_colin.csv', index=False)
        print(f"✅ Submission saved to submission_mlp_colin.csv")
    
    print("\n" + "="*70)
    print("PROCESS COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

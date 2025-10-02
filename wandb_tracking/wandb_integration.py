#!/usr/bin/env python3
"""
Weights & Biases (W&B) Integration Demo

This module demonstrates how to use W&B for experiment tracking,
visualization, and model management.
"""

import wandb
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train_with_wandb(config=None):
    """
    Train a model with W&B tracking
    
    Args:
        config: Dictionary with hyperparameters
    """
    # Initialize W&B run
    with wandb.init(project="ai-development-tools", config=config) as run:
        # Access hyperparameters
        config = wandb.config
        
        # Generate dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log metrics to W&B
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        })
        
        # Log feature importances
        feature_importance = model.feature_importances_
        wandb.log({
            "feature_importance": wandb.Histogram(feature_importance)
        })
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"W&B Run: {run.url}")
        
        return model


def run_sweep():
    """
    Run hyperparameter sweep with W&B
    """
    # Define sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'n_estimators': {
                'values': [50, 100, 150, 200]
            },
            'learning_rate': {
                'min': 0.01,
                'max': 0.3
            },
            'max_depth': {
                'values': [3, 5, 7, 10]
            }
        }
    }
    
    # Initialize sweep
    # sweep_id = wandb.sweep(sweep_config, project="ai-development-tools")
    # wandb.agent(sweep_id, train_with_wandb, count=10)
    
    print("Sweep configuration ready!")
    print("Uncomment the sweep code to run hyperparameter optimization")


if __name__ == "__main__":
    print("Training model with W&B tracking...\n")
    
    # Example configuration
    config = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    }
    
    # NOTE: Set your W&B API key before running:
    # export WANDB_API_KEY=your_api_key
    # or use: wandb.login()
    
    # Uncomment to run:
    # train_with_wandb(config)
    
    print("\nTo run this demo:")
    print("1. Install wandb: pip install wandb")
    print("2. Login: wandb login")
    print("3. Uncomment the train_with_wandb() call")
    print("\nFor hyperparameter sweeps, see run_sweep() function")

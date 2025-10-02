#!/usr/bin/env python3
"""
MLflow Experiment Tracking Demo

This module demonstrates how to use MLflow for tracking machine learning experiments,
logging parameters, metrics, and models.
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def train_model(n_estimators=100, max_depth=5, random_state=42):
    """
    Train a Random Forest model and log metrics with MLflow
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility
    """
    # Set experiment name
    mlflow.set_experiment("random-forest-demo")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Generate synthetic dataset
        X, y = make_classification(n_samples=1000, n_features=20, 
                                   n_informative=15, random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate and log metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return model


if __name__ == "__main__":
    # Example: Train multiple models with different hyperparameters
    print("Training models with different configurations...\n")
    
    configurations = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
    ]
    
    for config in configurations:
        print(f"Training with config: {config}")
        train_model(**config)
        print("-" * 50)
    
    print("\nAll experiments logged to MLflow!")
    print("Run 'mlflow ui' to view the tracking UI")

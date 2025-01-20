from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train model with hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        logger.info("Starting model training with GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"Best parameters found: {self.best_params}")
        
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'roc_auc': roc_auc
        } 
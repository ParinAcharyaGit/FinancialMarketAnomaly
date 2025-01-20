from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    
    # Load and preprocess data
    df = preprocessor.load_data('Financial_Market_Data.xlsx')
    df = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    df = feature_engineer.create_technical_indicators(df)
    
    # Prepare features and target
    X = df.drop('Y', axis=1)
    y = df['Y']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model_trainer.train_model(X_train, y_train)
    
    # Evaluate model
    results = model_trainer.evaluate_model(X_test, y_test)
    
    logger.info(f"Model Accuracy: {results['accuracy']}")
    logger.info("\nClassification Report:")
    logger.info(results['classification_report'])
    logger.info(f"ROC AUC Score: {results['roc_auc']}")

if __name__ == "__main__":
    main() 
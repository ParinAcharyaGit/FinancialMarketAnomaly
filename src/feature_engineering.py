import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    @staticmethod
    def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for financial data"""
        numerical_features = df.select_dtypes(include=['number']).columns
        
        for col in numerical_features:
            logger.info(f"Creating technical indicators for column: {col}")
            
            # Moving averages
            df[f'{col}_MA7'] = df[col].rolling(window=7).mean()
            df[f'{col}_MA30'] = df[col].rolling(window=30).mean()
            
            # RSI
            df[f'{col}_RSI'] = self.calculate_rsi(df[col])
            
            # Volatility
            df[f'{col}_volatility'] = df[col].rolling(window=30).std()
            
            # Momentum
            df[f'{col}_momentum'] = df[col].pct_change(periods=14)
        
        # Handle NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df 
"""
Preprocessing pipeline module
Creates reusable sklearn pipelines for data transformation
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger(__name__)

class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for ColumnTransformer that preserves DataFrame with column names
    
    Fixes: LightGBM warning about feature names
    """
    
    def __init__(self, column_transformer):
        self.column_transformer = column_transformer
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        self.column_transformer.fit(X, y)
        
        # Extract feature names from transformers
        self.feature_names_out_ = self._get_feature_names_out()
        
        return self
    
    def transform(self, X) -> pd.DataFrame:
        X_transformed = self.column_transformer.transform(X)
        
        # Convert to DataFrame with proper column names
        return pd.DataFrame(
            X_transformed,
            columns=self.feature_names_out_,
            index=X.index
        )
    
    # CORREÇÃO: Remover type hint incompatível e usar tipo genérico
    def fit_transform(self, X, y=None, **fit_params):  # type: ignore[override]
        """
        Fit to data, then transform it.
        
        Returns DataFrame instead of ndarray (intentional override)
        """
        self.fit(X, y)
        return self.transform(X)
    
    def set_output(self, *, transform=None):
        """Override to indicate DataFrame output"""
        return self
    
    def _get_feature_names_out(self):
        """
        Extract feature names from all transformers
        """
        feature_names = []
        
        for name, transformer, columns in self.column_transformer.transformers_:
            if name == 'remainder':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                # For OneHotEncoder and similar
                try:
                    names = transformer.get_feature_names_out(columns)
                    feature_names.extend(names)
                except:
                    # Fallback
                    if hasattr(transformer, 'categories_'):
                        # OneHotEncoder
                        for i, col in enumerate(columns):
                            for cat in transformer.categories_[i]:
                                feature_names.append(f"{col}_{cat}")
                    else:
                        # Numeric transformers
                        feature_names.extend(columns)
            else:
                # Simple transformers (numeric, binary)
                if isinstance(columns, list):
                    feature_names.extend(columns)
                else:
                    feature_names.append(columns)
        
        return feature_names
    
    def get_feature_names_out(self, input_features=None):
        """
        Compatibility with sklearn API
        """
        return self.feature_names_out_

def create_preprocessing_pipeline(X):
    """
    Creates reusable preprocessing pipeline
    
    Args:
        X: Feature dataframe
        
    Returns:
        preprocessor: ColumnTransformer object
        feature_info: Dict with column types
    """
    
    logger.info("\nStep 4: Preprocessing Pipeline")
    logger.info("-" * 70)
    
    # === IDENTIFY COLUMN TYPES ===
    
    numeric_cols = []
    binary_cols = []
    categorical_cols = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            unique_vals = set(X[col].dropna().unique())
            if len(unique_vals) == 2 and unique_vals.issubset({0, 1}):
                binary_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    logger.info(f"  Numeric: {len(numeric_cols)} columns")
    logger.info(f"  Binary: {len(binary_cols)} columns")
    logger.info(f"  Categorical: {len(categorical_cols)} columns")
    
    # === CREATE TRANSFORMERS ===
    
    numeric_transformer = ImbPipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = ImbPipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            max_categories=10
        ))
    ])
    
    binary_transformer = SimpleImputer(strategy='most_frequent')
    
    # === COMPOSE PREPROCESSOR ===
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('bin', binary_transformer, binary_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    feature_info = {
        'numeric_cols': numeric_cols,
        'binary_cols': binary_cols,
        'categorical_cols': categorical_cols,
        'n_features_in': len(X.columns)
    }
    
    logger.info(f"\n✅ Preprocessing pipeline created")
    
    return preprocessor, feature_info

def create_preprocessing_pipeline_enhanced(X, target='stroke'):
    """
    Enhanced preprocessing that preserves feature names
    
    Returns DataFrame-preserving preprocessor
    """
    
    feature_cols = [c for c in X.columns if c != target]
    
    num_cols = []
    bin_cols = []
    cat_cols = []
    
    for col in feature_cols:
        if X[col].dtype in ['int64', 'float64']:
            if X[col].nunique() == 2 and set(X[col].dropna().unique()).issubset({0, 1}):
                bin_cols.append(col)
            else:
                num_cols.append(col)
        else:
            cat_cols.append(col)
    
    print(f"✅ Feature types detected:")
    print(f"   Numeric: {len(num_cols)} | Binary: {len(bin_cols)} | Categorical: {len(cat_cols)}")
    
    # Define transformers
    numeric_transformer = ImbPipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = ImbPipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    binary_transformer = SimpleImputer(strategy='most_frequent')
    
    # Create base ColumnTransformer
    base_transformer = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
        ('bin', binary_transformer, bin_cols)
    ], remainder='drop')
    
    # Wrap with DataFrame-preserving wrapper
    preprocessor = DataFrameColumnTransformer(base_transformer)
    
    return preprocessor, {'numeric': num_cols, 'categorical': cat_cols, 'binary': bin_cols}

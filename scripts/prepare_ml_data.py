import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Create result directory if it doesn't exist
Path('results/prepare_ml_data').mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/prepare_ml_data/data_preparation.log'),
        logging.StreamHandler()
    ]
)

def prepare_ml_data():
    """
    Prepare extracted interface data for ML training to predict SRH interface recombination rates.
    """
    try:
        # Load the extracted interface data
        input_file = 'results/extract/extracted_data.csv'
        logging.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        logging.info(f"Original data shape: {df.shape}")
        logging.info(f"Interface distribution: {df['lid'].value_counts().sort_index().to_dict()}")
        
        # Define features and targets
        # Exclude spatial coordinates, voltage, and layer ID from features
        exclude_features = ['x', 'V', 'lid']
        
        # Target variables
        targets = ['IntSRHn', 'IntSRHp']
        
        # Feature columns (all columns except excluded ones and targets)
        feature_columns = [col for col in df.columns if col not in exclude_features + targets]
        
        logging.info(f"Number of features: {len(feature_columns)}")
        logging.info(f"Target variables: {targets}")
        
        # Separate features and targets
        X = df[feature_columns]
        y_n = df['IntSRHn']
        y_p = df['IntSRHp']
        
        # Log target statistics
        logging.info(f"IntSRHn statistics:")
        logging.info(f"  Mean: {y_n.mean():.2e}")
        logging.info(f"  Std: {y_n.std():.2e}")
        logging.info(f"  Min: {y_n.min():.2e}")
        logging.info(f"  Max: {y_n.max():.2e}")
        
        logging.info(f"IntSRHp statistics:")
        logging.info(f"  Mean: {y_p.mean():.2e}")
        logging.info(f"  Std: {y_p.std():.2e}")
        logging.info(f"  Min: {y_p.min():.2e}")
        logging.info(f"  Max: {y_p.max():.2e}")
        
        # Split data for each target
        X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
            X, y_n, test_size=0.2, random_state=42, stratify=df['lid']
        )
        
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X, y_p, test_size=0.2, random_state=42, stratify=df['lid']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled_n = scaler.fit_transform(X_train_n)
        X_test_scaled_n = scaler.transform(X_test_n)
        
        X_train_scaled_p = scaler.fit_transform(X_train_p)
        X_test_scaled_p = scaler.transform(X_test_p)
        
        # Convert back to DataFrames with column names
        X_train_n_df = pd.DataFrame(X_train_scaled_n, columns=feature_columns, index=X_train_n.index)
        X_test_n_df = pd.DataFrame(X_test_scaled_n, columns=feature_columns, index=X_test_n.index)
        
        X_train_p_df = pd.DataFrame(X_train_scaled_p, columns=feature_columns, index=X_train_p.index)
        X_test_p_df = pd.DataFrame(X_test_scaled_p, columns=feature_columns, index=X_test_p.index)
        
        # Save prepared data
        output_dir = Path('results/prepare_ml_data')
        
        # Save IntSRHn data
        X_train_n_df.to_csv(output_dir / 'X_train_IntSRHn.csv', index=False)
        X_test_n_df.to_csv(output_dir / 'X_test_IntSRHn.csv', index=False)
        y_train_n.to_csv(output_dir / 'y_train_IntSRHn.csv', index=False)
        y_test_n.to_csv(output_dir / 'y_test_IntSRHn.csv', index=False)
        
        # Save IntSRHp data
        X_train_p_df.to_csv(output_dir / 'X_train_IntSRHp.csv', index=False)
        X_test_p_df.to_csv(output_dir / 'X_test_IntSRHp.csv', index=False)
        y_train_p.to_csv(output_dir / 'y_train_IntSRHp.csv', index=False)
        y_test_p.to_csv(output_dir / 'y_test_IntSRHp.csv', index=False)
        
        # Save scaler for IntSRHn
        joblib.dump(scaler, output_dir / 'X_scaler_IntSRHn.joblib')
        
        # Create new scaler for IntSRHp
        scaler_p = StandardScaler()
        scaler_p.fit(X_train_p)
        joblib.dump(scaler_p, output_dir / 'X_scaler_IntSRHp.joblib')
        
        # Save feature names
        feature_names_df = pd.DataFrame({'feature_name': feature_columns})
        feature_names_df.to_csv(output_dir / 'feature_names.csv', index=False)
        
        # Log summary
        logging.info(f"Data preparation completed successfully!")
        logging.info(f"Training set sizes:")
        logging.info(f"  IntSRHn: {len(X_train_n_df)} samples")
        logging.info(f"  IntSRHp: {len(X_train_p_df)} samples")
        logging.info(f"Test set sizes:")
        logging.info(f"  IntSRHn: {len(X_test_n_df)} samples")
        logging.info(f"  IntSRHp: {len(X_test_p_df)} samples")
        
        # Log feature categories
        layer_features = [col for col in feature_columns if any(prefix in col for prefix in ['L1_', 'L2_', 'L3_'])]
        interface_features = [col for col in feature_columns if any(prefix in col for prefix in ['left_', 'right_'])]
        physics_features = [col for col in feature_columns if col not in layer_features + interface_features]
        
        logging.info(f"Feature categories:")
        logging.info(f"  Layer parameters: {len(layer_features)} features")
        logging.info(f"  Interface parameters: {len(interface_features)} features")
        logging.info(f"  Physics parameters: {len(physics_features)} features")
        
        return {
            'feature_columns': feature_columns,
            'layer_features': layer_features,
            'interface_features': interface_features,
            'physics_features': physics_features,
            'train_size_n': len(X_train_n_df),
            'test_size_n': len(X_test_n_df),
            'train_size_p': len(X_train_p_df),
            'test_size_p': len(X_test_p_df)
        }
        
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    results = prepare_ml_data()
    print(f"\nData preparation summary:")
    print(f"Features: {results['feature_columns']}")
    print(f"Training samples (IntSRHn): {results['train_size_n']}")
    print(f"Training samples (IntSRHp): {results['train_size_p']}")
    print(f"Test samples (IntSRHn): {results['test_size_n']}")
    print(f"Test samples (IntSRHp): {results['test_size_p']}") 
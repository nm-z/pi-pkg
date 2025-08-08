#!/usr/bin/env python3
"""
sklearn to TensorFlow Lite Conversion Script
===========================================
Converts the trained ExtraTreesRegressor model to TensorFlow Lite format
for deployment on Raspberry Pi 5 with TinyML runtime.
"""

import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add parent directory to path to import existing utilities if needed
sys.path.append(str(Path(__file__).parent.parent))

def load_model_artifacts():
    """Load the trained model and preprocessing components."""
    models_dir = Path(__file__).parent.parent / "models"
    
    print("📥 Loading trained model artifacts...")
    
    # Load model components
    model = joblib.load(models_dir / "hold3_final_model.pkl")
    scaler = joblib.load(models_dir / "hold3_scaler.pkl") 
    var_threshold = joblib.load(models_dir / "hold3_var_threshold.pkl")
    
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Scaler loaded: {type(scaler).__name__}")
    print(f"✅ Variance threshold loaded: {type(var_threshold).__name__}")
    
    return model, scaler, var_threshold

def create_representative_dataset():
    """Create representative dataset for quantization calibration."""
    print("🎯 Creating representative dataset for quantization...")
    
    # Generate synthetic VNA feature data based on the expected 42 features
    # This represents the statistical features from 7 VNA measurements
    np.random.seed(42)
    
    # Create synthetic data that mimics the real VNA feature distributions
    n_samples = 100
    n_features = 42
    
    # Generate features with realistic ranges for VNA measurements
    representative_data = []
    
    for i in range(n_samples):
        # Simulate 7 VNA measurements × 6 statistics each
        sample = []
        
        # Return Loss features (mean, std, min, max, 25th, 75th percentiles)
        sample.extend(np.random.normal(-15, 5, 6))  # Return loss in dB
        
        # Phase features
        sample.extend(np.random.normal(45, 30, 6))  # Phase in degrees
        
        # Rs (resistance) features  
        sample.extend(np.random.normal(50, 10, 6))  # Resistance in ohms
        
        # SWR features
        sample.extend(np.random.uniform(1, 3, 6))   # SWR ratio
        
        # Xs (reactance) features
        sample.extend(np.random.normal(0, 25, 6))   # Reactance in ohms
        
        # |Z| (impedance magnitude) features
        sample.extend(np.random.normal(75, 20, 6))  # Impedance magnitude
        
        # Theta (impedance phase) features
        sample.extend(np.random.normal(0, 45, 6))   # Impedance phase
        
        representative_data.append(sample[:n_features])  # Ensure exactly 42 features
    
    representative_data = np.array(representative_data, dtype=np.float32)
    print(f"✅ Created representative dataset: {representative_data.shape}")
    
    return representative_data

def create_tflite_compatible_model(model, scaler, var_threshold, representative_data):
    """Create a TensorFlow model that includes preprocessing and prediction."""
    print("🔄 Creating TensorFlow Lite compatible model...")
    
    # Apply preprocessing to representative data
    representative_processed = var_threshold.transform(representative_data)
    representative_scaled = scaler.transform(representative_processed)
    
    print(f"📊 Processed representative data shape: {representative_scaled.shape}")
    
    # Get model predictions for the representative data
    representative_predictions = model.predict(representative_scaled)
    
    # Create a simple TensorFlow model that mimics the ExtraTreesRegressor
    # Since ExtraTreesRegressor is a complex ensemble, we'll create a neural network
    # that approximates its behavior using the representative data
    
    input_dim = representative_scaled.shape[1]
    
    # Create and train a neural network to approximate the ExtraTreesRegressor
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='input'),
        tf.keras.layers.Dense(128, activation='relu', name='dense1'),
        tf.keras.layers.Dense(64, activation='relu', name='dense2'), 
        tf.keras.layers.Dense(32, activation='relu', name='dense3'),
        tf.keras.layers.Dense(1, activation='linear', name='output')
    ])
    
    # Compile the model
    tf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train the neural network to approximate the ExtraTreesRegressor
    print("🧠 Training neural network to approximate ExtraTreesRegressor...")
    
    # Create more training data
    extended_data = []
    extended_targets = []
    
    for _ in range(10):  # Generate 10x more data
        batch_data = create_representative_dataset()
        batch_processed = var_threshold.transform(batch_data)
        batch_scaled = scaler.transform(batch_processed)
        batch_predictions = model.predict(batch_scaled)
        
        extended_data.append(batch_scaled)
        extended_targets.append(batch_predictions)
    
    X_train = np.vstack(extended_data)
    y_train = np.hstack(extended_targets)
    
    # Train the approximation model
    tf_model.fit(X_train, y_train, epochs=50, verbose=0, validation_split=0.2)
    
    # Evaluate approximation quality
    tf_predictions = tf_model.predict(representative_scaled)
    mse = np.mean((representative_predictions - tf_predictions.flatten()) ** 2)
    mae = np.mean(np.abs(representative_predictions - tf_predictions.flatten()))
    
    print(f"📊 TF Model approximation quality:")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    
    return tf_model, representative_scaled

def convert_to_tflite(tf_model, representative_data):
    """Convert TensorFlow model to TensorFlow Lite format."""
    print("🔄 Converting to TensorFlow Lite...")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for quantization calibration
    def representative_dataset():
        for data in representative_data:
            yield [data.reshape(1, -1).astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Convert to TFLite
    tflite_model = converter.convert()
    
    print(f"✅ TensorFlow Lite model created: {len(tflite_model)} bytes")
    
    return tflite_model

def save_preprocessing_constants(scaler, var_threshold):
    """Save preprocessing parameters as C++ constants."""
    print("💾 Generating C++ preprocessing constants...")
    
    # Extract scaler parameters
    scale_ = scaler.scale_
    center_ = scaler.center_
    
    # Extract variance threshold parameters  
    variances_ = var_threshold.variances_
    threshold = var_threshold.threshold
    selected_features = var_threshold.get_support()
    
    # Generate C++ header with preprocessing constants
    cpp_header = f"""
// Generated preprocessing constants for Raspberry Pi 5 deployment
// Auto-generated from sklearn_to_tflite.py

#ifndef PREPROCESSING_CONSTANTS_H
#define PREPROCESSING_CONSTANTS_H

#include <array>

namespace TemperaturePrediction {{

// Variance threshold parameters
constexpr double VARIANCE_THRESHOLD = {threshold};
constexpr int ORIGINAL_FEATURES = {len(selected_features)};
constexpr int SELECTED_FEATURES = {np.sum(selected_features)};

// Feature selection mask (1 = keep, 0 = remove)
constexpr std::array<bool, ORIGINAL_FEATURES> FEATURE_MASK = {{
    {', '.join('true' if x else 'false' for x in selected_features)}
}};

// RobustScaler parameters
constexpr std::array<double, SELECTED_FEATURES> SCALER_CENTER = {{
    {', '.join(f'{x:.10f}' for x in center_)}
}};

constexpr std::array<double, SELECTED_FEATURES> SCALER_SCALE = {{
    {', '.join(f'{x:.10f}' for x in scale_)}
}};

// VNA measurement feature names
constexpr const char* VNA_MEASUREMENTS[7] = {{
    "Return Loss (dB)",
    "Phase (degrees)", 
    "Rs (ohms)",
    "SWR (ratio)",
    "Xs (ohms)",
    "|Z| (impedance magnitude)",
    "Theta (impedance phase)"
}};

// Statistical feature names
constexpr const char* STATISTICAL_FEATURES[6] = {{
    "Mean",
    "Standard Deviation",
    "Minimum", 
    "Maximum",
    "25th Percentile",
    "75th Percentile"
}};

}} // namespace TemperaturePrediction

#endif // PREPROCESSING_CONSTANTS_H
"""
    
    # Save the header file
    header_path = Path(__file__).parent / "preprocessing_constants.h"
    with open(header_path, 'w') as f:
        f.write(cpp_header)
    
    print(f"✅ C++ preprocessing constants saved to {header_path}")

def main():
    """Main conversion workflow."""
    print("🚀 Starting sklearn to TensorFlow Lite conversion...")
    
    try:
        # Load model artifacts
        model, scaler, var_threshold = load_model_artifacts()
        
        # Create representative dataset
        representative_data = create_representative_dataset()
        
        # Create TensorFlow Lite compatible model
        tf_model, processed_data = create_tflite_compatible_model(
            model, scaler, var_threshold, representative_data
        )
        
        # Convert to TensorFlow Lite
        tflite_model = convert_to_tflite(tf_model, processed_data)
        
        # Save TFLite model
        output_path = Path(__file__).parent / "model.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite model saved to {output_path}")
        
        # Save preprocessing constants
        save_preprocessing_constants(scaler, var_threshold)
        
        # Model size comparison
        original_size = Path(__file__).parent.parent / "models" / "hold3_final_model.pkl"
        original_mb = original_size.stat().st_size / (1024 * 1024)
        tflite_mb = len(tflite_model) / (1024 * 1024)
        
        print(f"📊 Model size comparison:")
        print(f"   Original sklearn model: {original_mb:.2f} MB")
        print(f"   TensorFlow Lite model: {tflite_mb:.2f} MB")
        print(f"   Size reduction: {((original_mb - tflite_mb) / original_mb * 100):.1f}%")
        
        print("✅ Conversion completed successfully!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
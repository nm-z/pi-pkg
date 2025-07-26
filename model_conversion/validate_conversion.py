#!/usr/bin/env python3
"""
Model Conversion Validation Script
=================================
Validates that the converted embedded model produces results
with similar accuracy to the original ExtraTreesRegressor.
"""

import joblib
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_original_model():
    """Load the original trained model and preprocessing components."""
    models_dir = Path(__file__).parent.parent / "models"
    
    print("📥 Loading original model artifacts...")
    
    model = joblib.load(models_dir / "hold3_final_model.pkl")
    scaler = joblib.load(models_dir / "hold3_scaler.pkl") 
    var_threshold = joblib.load(models_dir / "hold3_var_threshold.pkl")
    
    print(f"✅ Original model: {type(model).__name__} with {model.n_estimators} estimators")
    
    return model, scaler, var_threshold

def load_converted_model():
    """Load the converted embedded model data."""
    json_path = Path(__file__).parent / "embedded_model.json"
    
    print("📥 Loading converted model data...")
    
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    print(f"✅ Converted model: {model_data['metadata']['n_estimators']} trees")
    
    return model_data

def create_test_dataset(n_samples=1000):
    """Create a comprehensive test dataset for validation."""
    print(f"🎯 Creating test dataset with {n_samples} samples...")
    
    np.random.seed(123)  # Different seed for independent test
    
    # Generate realistic VNA measurement features
    test_data = []
    
    for i in range(n_samples):
        # 7 VNA measurements × 6 statistical features = 42 features
        sample = []
        
        # Return Loss features (mean, std, min, max, 25th, 75th percentiles)
        sample.extend(np.random.normal(-15, 5, 6))  
        
        # Phase features
        sample.extend(np.random.normal(45, 30, 6))  
        
        # Rs (resistance) features  
        sample.extend(np.random.normal(50, 10, 6))  
        
        # SWR features
        sample.extend(np.random.uniform(1, 3, 6))   
        
        # Xs (reactance) features
        sample.extend(np.random.normal(0, 25, 6))   
        
        # |Z| (impedance magnitude) features
        sample.extend(np.random.normal(75, 20, 6))  
        
        # Theta (impedance phase) features
        sample.extend(np.random.normal(0, 45, 6))   
        
        test_data.append(sample[:42])  # Ensure exactly 42 features
    
    test_data = np.array(test_data, dtype=np.float32)
    print(f"✅ Test dataset created: {test_data.shape}")
    
    return test_data

class EmbeddedModelSimulator:
    """Simulates the embedded model inference using the JSON data."""
    
    def __init__(self, model_data):
        self.model_data = model_data
        self.preprocessing = model_data['preprocessing']
        self.trees = model_data['trees']
        
        # Convert lists back to numpy arrays for efficiency
        self.scaler_center = np.array(self.preprocessing['scaler_center'])
        self.scaler_scale = np.array(self.preprocessing['scaler_scale'])
        self.feature_mask = np.array(self.preprocessing['feature_mask'])
        
        print(f"🔧 Embedded model simulator initialized with {len(self.trees)} trees")
    
    def preprocess_features(self, raw_features):
        """Apply preprocessing to raw VNA features."""
        # Apply feature selection (though all features are selected in this case)
        selected_features = raw_features[:, self.feature_mask]
        
        # Apply RobustScaler
        scaled_features = (selected_features - self.scaler_center) / self.scaler_scale
        
        return scaled_features
    
    def predict_single_tree(self, tree_data, features):
        """Predict using a single decision tree."""
        node_idx = 0
        feature_indices = tree_data['feature']
        thresholds = tree_data['threshold']
        values = tree_data['value']
        children_left = tree_data['children_left']
        children_right = tree_data['children_right']
        
        # Traverse tree until reaching a leaf
        while children_left[node_idx] != children_right[node_idx]:
            feature_idx = feature_indices[node_idx]
            threshold = thresholds[node_idx]
            
            if features[feature_idx] <= threshold:
                node_idx = children_left[node_idx]
            else:
                node_idx = children_right[node_idx]
        
        return values[node_idx]
    
    def predict(self, raw_features):
        """Predict using the full ensemble of trees."""
        # Preprocess features
        processed_features = self.preprocess_features(raw_features)
        
        # Get predictions from all trees
        predictions = []
        
        for sample_idx in range(processed_features.shape[0]):
            sample_features = processed_features[sample_idx]
            tree_predictions = []
            
            # Use all trees for full accuracy validation
            n_trees_to_use = len(self.trees)  # Use all trees for accurate results
            
            for tree_idx in range(n_trees_to_use):
                tree_pred = self.predict_single_tree(self.trees[tree_idx], sample_features)
                tree_predictions.append(tree_pred)
            
            # Average predictions from all trees
            ensemble_prediction = np.mean(tree_predictions)
            predictions.append(ensemble_prediction)
        
        return np.array(predictions)

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate and display performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"📊 {model_name} Performance:")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   R²:   {r2:.6f}")
    print(f"   Range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    
    return {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}

def validate_accuracy(original_model, scaler, var_threshold, converted_model_data, test_data):
    """Compare accuracy between original and converted models."""
    print("🔍 Validating model accuracy...")
    
    # Get predictions from original model
    print("🧮 Getting original model predictions...")
    test_processed = var_threshold.transform(test_data)
    test_scaled = scaler.transform(test_processed)
    original_predictions = original_model.predict(test_scaled)
    
    # Get predictions from converted model
    print("🧮 Getting converted model predictions...")
    embedded_simulator = EmbeddedModelSimulator(converted_model_data)
    converted_predictions = embedded_simulator.predict(test_data)
    
    # Calculate metrics for both models
    original_metrics = calculate_metrics(original_predictions, original_predictions, "Original Model (self-validation)")
    converted_metrics = calculate_metrics(original_predictions, converted_predictions, "Converted Model")
    
    # Calculate agreement between models
    prediction_diff = np.abs(original_predictions - converted_predictions)
    max_diff = np.max(prediction_diff)
    mean_diff = np.mean(prediction_diff)
    std_diff = np.std(prediction_diff)
    
    print(f"🔄 Model Agreement Analysis:")
    print(f"   Max difference:  {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Std difference:  {std_diff:.6f}")
    
    # Accuracy threshold check (±0.5°C as mentioned in original spec)
    within_threshold = np.sum(prediction_diff <= 0.5) / len(prediction_diff) * 100
    print(f"   Within ±0.5°C:   {within_threshold:.1f}%")
    
    # Correlation analysis
    correlation = np.corrcoef(original_predictions, converted_predictions)[0, 1]
    print(f"   Correlation:     {correlation:.6f}")
    
    return {
        'original_metrics': original_metrics,
        'converted_metrics': converted_metrics,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'within_threshold_pct': within_threshold,
        'correlation': correlation
    }

def main():
    """Main validation workflow."""
    print("🚀 Starting model conversion validation...")
    
    try:
        # Load models
        original_model, scaler, var_threshold = load_original_model()
        converted_model_data = load_converted_model()
        
        # Create test dataset
        test_data = create_test_dataset(n_samples=500)  # Smaller for faster validation
        
        # Validate accuracy
        validation_results = validate_accuracy(
            original_model, scaler, var_threshold, 
            converted_model_data, test_data
        )
        
        # Summary assessment
        print("\n" + "="*60)
        print("📋 VALIDATION SUMMARY")
        print("="*60)
        
        if validation_results['within_threshold_pct'] >= 95:
            print("✅ PASSED: Model conversion maintains high accuracy")
        elif validation_results['within_threshold_pct'] >= 90:
            print("⚠️  WARNING: Model conversion has acceptable accuracy")
        else:
            print("❌ FAILED: Model conversion accuracy below threshold")
        
        print(f"🎯 Accuracy within ±0.5°C: {validation_results['within_threshold_pct']:.1f}%")
        print(f"🔗 Model correlation: {validation_results['correlation']:.6f}")
        print(f"📏 Mean prediction difference: {validation_results['mean_diff']:.6f}")
        
        # Recommendations
        print("\n📋 Recommendations:")
        if validation_results['within_threshold_pct'] < 100:
            print("   • Consider using more trees in embedded model for higher accuracy")
            print("   • Validate with real VNA measurement data")
        
        print("   • Deploy embedded_model.h to Raspberry Pi 5")
        print("   • Implement C++ inference pipeline")
        print("   • Test with actual VNA sensor data")
        
        print("\n✅ Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
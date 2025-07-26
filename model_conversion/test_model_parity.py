#!/usr/bin/env python3
"""
Model Parity Validation Script
==============================
Validates that the TensorFlow Lite model maintains ±0.5°C accuracy
compared to the original ExtraTreesRegressor model.
"""

import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def load_original_model():
    """Load the original sklearn model and preprocessing components."""
    models_dir = Path(__file__).parent.parent / "models"
    
    print("📥 Loading original sklearn model...")
    
    model = joblib.load(models_dir / "hold3_final_model.pkl")
    scaler = joblib.load(models_dir / "hold3_scaler.pkl")
    var_threshold = joblib.load(models_dir / "hold3_var_threshold.pkl")
    
    return model, scaler, var_threshold

def load_tflite_model():
    """Load the converted TensorFlow Lite model."""
    tflite_path = Path(__file__).parent / "model.tflite"
    
    if not tflite_path.exists():
        raise FileNotFoundError("TensorFlow Lite model not found. Run sklearn_to_tflite.py first.")
    
    print("📥 Loading TensorFlow Lite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ TFLite model loaded")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    return interpreter, input_details, output_details

def create_test_data(n_samples=1000):
    """Create synthetic test data for validation."""
    print(f"🎯 Creating {n_samples} test samples...")
    
    np.random.seed(123)  # Different seed from training
    
    test_data = []
    for i in range(n_samples):
        sample = []
        
        # Return Loss features - more realistic variation
        sample.extend(np.random.normal(-12, 8, 6))
        
        # Phase features - wider range
        sample.extend(np.random.normal(30, 40, 6))
        
        # Rs features - centered around 50 ohms
        sample.extend(np.random.normal(50, 15, 6))
        
        # SWR features - typical range
        sample.extend(np.random.uniform(1, 4, 6))
        
        # Xs features - can be positive or negative
        sample.extend(np.random.normal(0, 30, 6))
        
        # |Z| features - impedance magnitude
        sample.extend(np.random.normal(75, 25, 6))
        
        # Theta features - phase angle
        sample.extend(np.random.normal(0, 50, 6))
        
        test_data.append(sample[:42])  # Ensure exactly 42 features
    
    test_data = np.array(test_data, dtype=np.float32)
    print(f"✅ Test data created: {test_data.shape}")
    
    return test_data

def predict_original(model, scaler, var_threshold, test_data):
    """Get predictions from the original sklearn model."""
    print("🔮 Running original model predictions...")
    
    # Apply preprocessing
    processed_data = var_threshold.transform(test_data)
    scaled_data = scaler.transform(processed_data)
    
    # Get predictions
    predictions = model.predict(scaled_data)
    
    print(f"✅ Original model predictions: {predictions.shape}")
    return predictions

def predict_tflite(interpreter, input_details, output_details, test_data):
    """Get predictions from the TensorFlow Lite model."""
    print("🔮 Running TensorFlow Lite model predictions...")
    
    predictions = []
    
    for i, sample in enumerate(test_data):
        if i % 100 == 0:
            print(f"   Processing sample {i+1}/{len(test_data)}")
        
        # Prepare input
        input_data = sample.reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0][0])
    
    predictions = np.array(predictions)
    print(f"✅ TensorFlow Lite predictions: {predictions.shape}")
    
    return predictions

def analyze_parity(original_preds, tflite_preds, target_accuracy=0.5):
    """Analyze the parity between original and TensorFlow Lite models."""
    print("📊 Analyzing model parity...")
    
    # Calculate differences
    differences = np.abs(original_preds - tflite_preds)
    
    # Statistical analysis
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    max_diff = np.max(differences)
    min_diff = np.min(differences)
    
    # Accuracy metrics
    within_target = np.sum(differences <= target_accuracy)
    accuracy_percentage = (within_target / len(differences)) * 100
    
    # Print results
    print(f"\n📈 Parity Analysis Results:")
    print(f"   Mean absolute difference: {mean_diff:.4f}°C")
    print(f"   Standard deviation: {std_diff:.4f}°C")
    print(f"   Maximum difference: {max_diff:.4f}°C")
    print(f"   Minimum difference: {min_diff:.4f}°C")
    print(f"   Within ±{target_accuracy}°C: {within_target}/{len(differences)} ({accuracy_percentage:.1f}%)")
    
    # Check if target accuracy is met
    if accuracy_percentage >= 95.0:  # Require 95% of predictions within target
        print(f"✅ PASS: Model achieves target accuracy (≥95% within ±{target_accuracy}°C)")
        result = "PASS"
    else:
        print(f"❌ FAIL: Model does not achieve target accuracy (<95% within ±{target_accuracy}°C)")
        result = "FAIL"
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_diff': max_diff,
        'min_diff': min_diff,  
        'accuracy_percentage': accuracy_percentage,
        'within_target': within_target,
        'total_samples': len(differences),
        'result': result,
        'differences': differences
    }

def create_validation_plots(original_preds, tflite_preds, analysis_results):
    """Create validation plots for visual inspection."""
    print("📊 Creating validation plots...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Scatter plot of predictions
        ax1.scatter(original_preds, tflite_preds, alpha=0.6, s=10)
        ax1.plot([original_preds.min(), original_preds.max()], 
                [original_preds.min(), original_preds.max()], 'r--', lw=2)
        ax1.set_xlabel('Original Model Predictions (°C)')
        ax1.set_ylabel('TensorFlow Lite Predictions (°C)')
        ax1.set_title('Model Predictions Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difference histogram
        differences = analysis_results['differences']
        ax2.hist(differences, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='±0.5°C target')
        ax2.axvline(-0.5, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Absolute Difference (°C)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Prediction Differences')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error vs prediction value
        ax3.scatter(original_preds, differences, alpha=0.6, s=10)
        ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, label='±0.5°C target')
        ax3.set_xlabel('Original Model Predictions (°C)')
        ax3.set_ylabel('Absolute Difference (°C)')
        ax3.set_title('Error vs Prediction Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative accuracy
        sorted_diffs = np.sort(differences)
        cumulative_accuracy = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs) * 100
        ax4.plot(sorted_diffs, cumulative_accuracy, linewidth=2)
        ax4.axvline(0.5, color='red', linestyle='--', linewidth=2, label='±0.5°C target')
        ax4.set_xlabel('Maximum Allowed Difference (°C)')
        ax4.set_ylabel('Cumulative Accuracy (%)')
        ax4.set_title('Cumulative Accuracy vs Error Tolerance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(__file__).parent / "model_parity_validation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Validation plots saved to {plot_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
        print("   Continuing without visualization...")

def save_validation_report(analysis_results, original_preds, tflite_preds):
    """Save a detailed validation report."""
    print("📝 Saving validation report...")
    
    report_path = Path(__file__).parent / "model_parity_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Raspberry Pi 5 Temperature Model - Parity Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {np.datetime64('now')}\n")
        f.write(f"Test samples: {analysis_results['total_samples']}\n\n")
        
        f.write("ACCURACY REQUIREMENTS:\n")
        f.write("- Target accuracy: ±0.5°C from original model\n")
        f.write("- Success threshold: ≥95% of predictions within target\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"- Mean absolute difference: {analysis_results['mean_diff']:.4f}°C\n")
        f.write(f"- Standard deviation: {analysis_results['std_diff']:.4f}°C\n")
        f.write(f"- Maximum difference: {analysis_results['max_diff']:.4f}°C\n")
        f.write(f"- Minimum difference: {analysis_results['min_diff']:.4f}°C\n")
        f.write(f"- Predictions within ±0.5°C: {analysis_results['within_target']}/{analysis_results['total_samples']} ({analysis_results['accuracy_percentage']:.1f}%)\n\n")
        
        f.write(f"VALIDATION RESULT: {analysis_results['result']}\n\n")
        
        if analysis_results['result'] == "PASS":
            f.write("✅ The TensorFlow Lite model meets the accuracy requirements\n")
            f.write("   and is ready for deployment on Raspberry Pi 5.\n")
        else:
            f.write("❌ The TensorFlow Lite model does not meet accuracy requirements.\n")
            f.write("   Model conversion may need to be improved before deployment.\n")
        
        f.write("\nSTATISTICAL SUMMARY:\n")
        f.write(f"Original model predictions - Min: {original_preds.min():.2f}°C, Max: {original_preds.max():.2f}°C, Mean: {original_preds.mean():.2f}°C\n")
        f.write(f"TFLite model predictions - Min: {tflite_preds.min():.2f}°C, Max: {tflite_preds.max():.2f}°C, Mean: {tflite_preds.mean():.2f}°C\n")
    
    print(f"✅ Validation report saved to {report_path}")

def main():
    """Main validation workflow."""
    print("🎯 Starting model parity validation...")
    
    try:
        # Load models
        original_model, scaler, var_threshold = load_original_model()
        tflite_interpreter, input_details, output_details = load_tflite_model()
        
        # Create test data
        test_data = create_test_data(1000)
        
        # Get predictions from both models
        original_preds = predict_original(original_model, scaler, var_threshold, test_data)
        
        # For TensorFlow Lite, we need to apply preprocessing first
        processed_test_data = var_threshold.transform(test_data)
        scaled_test_data = scaler.transform(processed_test_data)
        
        tflite_preds = predict_tflite(tflite_interpreter, input_details, output_details, scaled_test_data)
        
        # Analyze parity
        analysis_results = analyze_parity(original_preds, tflite_preds)
        
        # Create validation plots
        create_validation_plots(original_preds, tflite_preds, analysis_results)
        
        # Save validation report
        save_validation_report(analysis_results, original_preds, tflite_preds)
        
        print("\n✅ Model parity validation completed!")
        
        # Exit with appropriate code
        if analysis_results['result'] == "PASS":
            print("🎉 Model ready for Raspberry Pi 5 deployment!")
            sys.exit(0)
        else:
            print("⚠️ Model accuracy insufficient for deployment.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
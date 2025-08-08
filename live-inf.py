#!/usr/bin/env python3
"""
Live VNA Monitoring and Temperature Inference using hold5 model
==============================================================
Real-time script for VNA data monitoring and temperature prediction.
Uses Java VNAhl command to capture VNA data, then runs inference using hold5 model.

VERSION: 1.0.6
"""

import argparse
import time
import threading
import subprocess
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

console = Console()

# Print versions immediately when script starts
console.print("[bold blue]=== VERSION INFO ===[/bold blue]")
console.print(f"Script Version: 1.0.6")
console.print(f"Python: {sys.version}")
console.print(f"NumPy: {np.__version__}")
console.print(f"Pandas: {pd.__version__}")
console.print(f"Joblib: {joblib.__version__}")
console.print(f"Script path: {Path(__file__).resolve()}")
console.print(f"Working directory: {Path.cwd()}")
console.print("[bold blue]==================[/bold blue]")

class VNADataHandler(FileSystemEventHandler):
    """Handles new VNA CSV files and triggers inference."""

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.processed_files = set()

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() == '.csv' and str(file_path) not in self.processed_files:
            self.processed_files.add(str(file_path))
            console.print(f"[green]New VNA data detected: {file_path.name}[/green]")

            # Wait a moment for file to be fully written
            time.sleep(1)

            # Process the VNA data
            threading.Thread(
                target=self.inference_engine.process_vna_file,
                args=(file_path,),
                daemon=True
            ).start()

class LiveVnaInference:
    """Live VNA monitoring and temperature inference using hold5 model."""

    def __init__(self):
        # Use hold5 model directory
        self.model_dir = Path("best_model_hold5")
        
        # VNA Live Data monitoring
        self.vna_data_path = Path("vna-live")
        self.vna_data_path.mkdir(parents=True, exist_ok=True)
        
        # VNAhl Java command configuration
        self.vna_jar_path = "vnaJ-hl.3.3.3_jp.jar"
        self.vna_cal_file = "NATES-miniVNA_Tiny.cal"
        
        self.live_monitoring = False
        self.last_prediction = None
        self.prediction_count = 0
        self.vna_process = None

        # Model components
        self.model = None
        self.scaler = None
        self.var_threshold = None
        self.kbest_selector = None
        # Raw features expected from CSV before preprocessing (4 cols × 10,001 points)
        self.expected_raw_features = 40004

    def load_model_components(self):
        """Load all model components from hold5 saved artifacts."""
        if self.model is None:
            with console.status("[bold blue]Loading hold5 model components..."):
                # Load preprocessing first so we can derive final input dimensionality
                self.scaler = joblib.load(self.model_dir / "hold5_scaler.pkl")
                self.var_threshold = joblib.load(self.model_dir / "hold5_var_threshold.pkl")
                self.kbest_selector = joblib.load(self.model_dir / "hold5_kbest_selector.pkl")

                # Define the correct ResNet architecture (must match training)
                class ResNetBlock(nn.Module):
                    def __init__(self, dim, dropout=0.1):
                        super().__init__()
                        self.norm1 = nn.LayerNorm(dim)
                        self.linear1 = nn.Linear(dim, dim)
                        self.dropout1 = nn.Dropout(dropout)
                        self.norm2 = nn.LayerNorm(dim)
                        self.linear2 = nn.Linear(dim, dim)
                        self.dropout2 = nn.Dropout(dropout)
                        
                    def forward(self, x):
                        residual = x
                        x = self.norm1(x)
                        x = torch.relu(self.linear1(x))
                        x = self.dropout1(x)
                        x = self.norm2(x)
                        x = self.linear2(x)
                        x = self.dropout2(x)
                        return x + residual
                
                class CustomResNet(nn.Module):
                    def __init__(self, input_dim, hidden_dim, num_blocks, dropout):
                        super().__init__()
                        self.input_proj = nn.Linear(input_dim, hidden_dim)
                        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
                        self.output = nn.Linear(hidden_dim, 1)
                        
                    def forward(self, x):
                        x = self.input_proj(x)
                        for block in self.blocks:
                            x = block(x)
                        return self.output(x).squeeze(-1)
                
                # Determine final feature count after preprocessing pipeline so
                # the PyTorch model input dimension matches the checkpoint
                # (avoids size-mismatch errors)
                try:
                    dummy = np.zeros((1, self.expected_raw_features), dtype=np.float32)
                    vt_out = self.var_threshold.transform(dummy)
                    kb_out = self.kbest_selector.transform(vt_out)
                    final_input_dim = kb_out.shape[1]
                except Exception:
                    # Fallback: try to infer from selector attributes
                    final_input_dim = getattr(self.kbest_selector, 'k', None)
                    if final_input_dim is None or final_input_dim == 'all':
                        support = getattr(self.kbest_selector, 'get_support', None)
                        if callable(support):
                            final_input_dim = int(support().sum())
                    if not isinstance(final_input_dim, int) or final_input_dim <= 0:
                        # Last resort – keep previous behavior (may error)
                        final_input_dim = self.expected_raw_features

                # Create model instance with correct parameters from training
                # Best parameters: hidden_dim=128, num_blocks=2, dropout=0.010716112128622697
                try:
                    self.model = CustomResNet(
                        input_dim=final_input_dim,
                        hidden_dim=128,   # From best_params
                        num_blocks=2,     # From best_params  
                        dropout=0.010716112128622697  # From best_params
                    )
                    console.print(
                        f"Model created with input_dim={final_input_dim}, derived from preprocessing",
                        style="blue"
                    )
                    console.print("Model architecture created successfully", style="green")
                    
                    # Load the model weights
                    model_path = self.model_dir / "hold5_final_model.pt"
                    console.print(f"Loading model from: {model_path}", style="blue")
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    self.model.eval()  # Set to evaluation mode
                    console.print("Model loaded and set to evaluation mode", style="green")
                    
                except Exception as e:
                    console.print(f"Error loading model: {e}", style="red")
                    raise

            # Create success table
            table = Table(title="Hold5 Model Components Loaded", show_header=True, header_style="bold magenta")
            table.add_column("Component", style="cyan")
            table.add_column("Type", style="green")

            table.add_row("Model", f"{type(self.model).__name__}")
            table.add_row("Scaler", f"{type(self.scaler).__name__}")
            table.add_row("Variance Threshold", f"{type(self.var_threshold).__name__}")
            table.add_row("KBest Selector", f"{type(self.kbest_selector).__name__}")

            console.print(table)

    def start_vna_capture(self):
        """Start VNAhl Java application for data capture."""
        if not os.path.exists(self.vna_jar_path):
            console.print(f"Error: VNA JAR file not found at {self.vna_jar_path}", style="red")
            return False
            
        if not os.path.exists(self.vna_cal_file):
            console.print(f"Error: VNA calibration file not found: {self.vna_cal_file}", style="red")
            return False

        try:
            # Start VNAhl with the correct Java command and parameters
            # Adjusted frequency range and steps to ensure we get ~10,001 data points
            cmd = [
                "java",
                "-Dpurejavacomm.log=false",
                "-Dpurejavacomm.debug=false",
                "-Dfstart=45000000",
                "-Dfstop=55000000",  # Reduced range to get more points
                "-Dfsteps=10001",
                "-DdriverId=20",
                "-Dcalfile=NATES-miniVNA_Tiny.cal",
                "-Dexports=csv",
                "-DexportDirectory=vna-live",
                "-DexportFilename=live-vna{0,date,yyMMdd}_{0,time,HHmmss}",
                "-Dscanmode=REFL",
                "-DdriverPort=ttyUSB0",
                "-DkeepGeneratorOn",
                "-jar", self.vna_jar_path
            ]
            
            console.print(f"Starting VNAhl: {' '.join(cmd)}", style="blue")
            
            # Start VNA process
            self.vna_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path.cwd())
            )
            
            console.print("VNAhl started successfully", style="green")
            return True
            
        except Exception as e:
            console.print(f"Error starting VNAhl: {e}", style="red")
            return False

    def stop_vna_capture(self):
        """Stop VNAhl Java application."""
        if self.vna_process:
            console.print("Stopping VNAhl...", style="yellow")
            self.vna_process.terminate()
            self.vna_process.wait()
            self.vna_process = None
            console.print("VNAhl stopped", style="green")

    def run_inference(self, features):
        """Run inference on the provided features using hold5 PyTorch model."""
        with console.status("[bold blue]Running hold5 model inference..."):
            # Convert to numpy array and reshape for single sample
            x_sample = np.array(features).reshape(1, -1)

            # Apply preprocessing pipeline (same as training)
            # Apply variance threshold
            x_var_threshold = self.var_threshold.transform(x_sample)
            
            # Apply KBest feature selection
            x_kbest = self.kbest_selector.transform(x_var_threshold)
            
            # Apply scaler
            x_scaled = self.scaler.transform(x_kbest)
            
            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x_scaled)
            
            # Run prediction with PyTorch model
            with torch.no_grad():
                prediction = self.model(x_tensor).item()
            
            return prediction

    def extract_vna_features(self, vna_df):
        """Extract features from VNA data for hold5 model - Return Loss, Phase, Rs, and Xs."""
        try:
            # The hold5 model expects 40,004 features from 4 measurement columns with 10,001 values each
            # Extract raw values from Return Loss(dB), Phase(deg), Rs, and Xs (NOT frequency)
            features = []
            
            # Show all available columns for debugging
            console.print(f"Available columns: {list(vna_df.columns)}", style="cyan")
            
            # Try different possible column names for Return Loss
            return_loss_col = None
            for col in vna_df.columns:
                if 'return' in col.lower() or 'loss' in col.lower() or 's11' in col.lower():
                    return_loss_col = col
                    break
            
            if return_loss_col:
                return_loss = vna_df[return_loss_col].values
                features.extend(return_loss.tolist())
                console.print(f"Added {len(return_loss)} Return Loss features from column '{return_loss_col}'", style="green")
            else:
                console.print("Missing Return Loss column - tried: return, loss, s11", style="red")
                return None
            
            # Try different possible column names for Phase
            phase_col = None
            for col in vna_df.columns:
                if 'phase' in col.lower():
                    phase_col = col
                    break
            
            if phase_col:
                phase = vna_df[phase_col].values
                features.extend(phase.tolist())
                console.print(f"Added {len(phase)} Phase features from column '{phase_col}'", style="green")
            else:
                console.print("Missing Phase column", style="red")
                return None
            
            # Try different possible column names for Rs (Resistance)
            rs_col = None
            for col in vna_df.columns:
                if col.lower() == 'rs' or 'resistance' in col.lower():
                    rs_col = col
                    break
            
            if rs_col:
                rs = vna_df[rs_col].values
                features.extend(rs.tolist())
                console.print(f"Added {len(rs)} Rs (Resistance) features from column '{rs_col}'", style="green")
            else:
                console.print("Missing Rs column", style="red")
                return None
            
            # Try different possible column names for Xs (Reactance)
            xs_col = None
            for col in vna_df.columns:
                if col.lower() == 'xs' or 'reactance' in col.lower():
                    xs_col = col
                    break
            
            if xs_col:
                xs = vna_df[xs_col].values
                features.extend(xs.tolist())
                console.print(f"Added {len(xs)} Xs (Reactance) features from column '{xs_col}'", style="green")
            else:
                console.print("Missing Xs column", style="red")
                return None
            
            console.print(f"Total features extracted: {len(features)}", style="green")
            return features
            
        except Exception as e:
            console.print(f"Error extracting features: {e}", style="red")
            return None

    def process_vna_file(self, file_path):
        """Process a VNA CSV file and run inference."""
        try:
            console.print(f"Processing VNA file: {file_path.name}", style="blue")
            
            # Use the python engine for robustness and handle various CSV formats
            # Force reading ALL rows - don't let pandas truncate
            try:
                vna_df = pd.read_csv(file_path, engine='python', encoding='utf-8')
            except Exception as e:
                console.print(f"Error reading CSV with python engine: {e}", style="red")
                # Fallback to default engine with explicit parameters
                try:
                    vna_df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                except Exception as e2:
                    console.print(f"Error reading CSV with default engine: {e2}", style="red")
                    # Last fallback - read line by line if needed
                    vna_df = pd.read_csv(file_path, engine='python', encoding='latin-1')
            
            # ADD THIS LINE TO DEBUG
            console.print(f"VNA CSV has {len(vna_df.columns)} columns", style="cyan")
            console.print("Pandas sees these columns:", vna_df.columns.tolist(), style="yellow")
            console.print(f"DataFrame shape: {vna_df.shape}", style="yellow")
            console.print("First few rows:", style="yellow")
            console.print(vna_df.head())
            
            console.print(f"Loaded VNA data: {vna_df.shape}", style="green")
            
            # Extract features
            features = self.extract_vna_features(vna_df)
            if features is None:
                console.print("Failed to extract features", style="red")
                return
                
            console.print(f"Extracted {len(features)} features", style="green")
            
            # Check if we have the expected number of features
            if len(features) != 40004:
                console.print(f"ERROR: Expected 40,004 features but got {len(features)}", style="red")
                console.print("This will cause a shape mismatch with the model", style="red")
                return
            
            # Run inference
            prediction = self.run_inference(features)
            
            # Display results
            self.display_vna_results(file_path.name, prediction)
            
            # Save results
            self.save_vna_result(file_path.name, prediction)
            
            # Update counters
            self.last_prediction = prediction
            self.prediction_count += 1
            
        except Exception as e:
            console.print(f"Error processing VNA file: {e}", style="red")

    def display_vna_results(self, filename, prediction):
        """Display inference results."""
        # Create results panel
        results_text = f"""
File: {filename}
Temperature Prediction: {prediction:.2f}°C
Model: Hold5 CustomResNet (2 residual blocks, 128 hidden dim)
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        panel = Panel(
            results_text,
            title="[bold green]Hold5 Model Inference Results[/bold green]",
            border_style="green"
        )
        console.print(panel)

    def save_vna_result(self, filename, prediction):
        """Save inference results to file."""
        results_file = Path("vna_inference_results.txt")
        
        with open(results_file, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {filename} | {prediction:.2f}°C\n")
        
        console.print(f"Results saved to {results_file}", style="green")

    def start_vna_monitoring(self):
        """Start VNA monitoring - VNAhl runs continuously."""
        console.print("VNAhl is running continuously - monitoring for new files...", style="green")
        console.print("Press Ctrl+C to stop monitoring", style="yellow")

    def start_live_monitoring(self):
        """Start live monitoring with VNAhl integration."""
        console.print("Starting Live VNA Monitoring with Hold5 Model", style="bold blue")
        
        # Load model components
        self.load_model_components()
        
        # Start VNAhl capture
        if not self.start_vna_capture():
            console.print("Failed to start VNAhl. Exiting.", style="red")
            return
        
        # Start file monitoring
        event_handler = VNADataHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.vna_data_path), recursive=False)
        observer.start()
        
        self.live_monitoring = True
        
        # Start VNA monitoring
        self.start_vna_monitoring()
        
        console.print("Live monitoring started!", style="green")
        console.print("Press Ctrl+C to stop", style="yellow")
        
        try:
            while self.live_monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\nStopping live monitoring...", style="yellow")
            self.stop_vna_capture()
            observer.stop()
            observer.join()
            self.live_monitoring = False
            console.print("Live monitoring stopped", style="green")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Live VNA Monitoring with Hold5 Model")
    args = parser.parse_args()
    
    # Create inference engine
    inference_engine = LiveVnaInference()
    
    # Start live monitoring
    inference_engine.start_live_monitoring()

if __name__ == "__main__":
    main()

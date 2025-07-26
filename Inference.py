#!/usr/bin/env python3
"""
Interactive Inference Script + Live VNA Monitoring
=================================================
Interactive script to run inference on any sample from the trained Dataset 3 model.
Now includes live VNA data monitoring and F12 automation for real-time inference.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import random
import time
import threading
from pathlib import Path
from InquirerPy import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

# Live VNA monitoring imports
import keyboard
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

console = Console()

class VNADataHandler(FileSystemEventHandler):
    """Handles new VNA CSV files and triggers inference."""
    
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.processed_files = set()
        
    def on_created(self, event):
        if event.is_dir:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() == '.csv' and str(file_path) not in self.processed_files:
            self.processed_files.add(str(file_path))
            console.print(f"[green]🆕 New VNA data detected: {file_path.name}[/green]")
            
            # Wait a moment for file to be fully written
            time.sleep(1)
            
            # Process the VNA data
            threading.Thread(
                target=self.inference_engine.process_vna_file,
                args=(file_path,),
                daemon=True
            ).start()

class DatasetInference:
    def __init__(self, dataset=None, row=None, random_row=False):
        self.data_splits_path = Path("/home/nate/Desktop/nates_recipe-V2/data_splits/Combined")
        self.model_dir = Path("best_model_hold3")
        
        # VNA Live Data monitoring
        self.vna_data_path = Path("/home/nate/Desktop/nates_recipe-V2/PI_PKG/VNA Live Data")
        self.vna_data_path.mkdir(parents=True, exist_ok=True)
        self.live_monitoring = False
        self.f12_automation = False
        self.last_prediction = None
        self.prediction_count = 0
        
        # Hard-coded data split information
        self.datasets = {
            "train": {
                "name": "Training Set",
                "description": "Data used for model training",
                "max_rows": 2781,
                "features_file": "train_features.csv",
                "targets_file": "train_targets.csv"
            },
            "test": {
                "name": "Test Set", 
                "description": "Data used for performance evaluation",
                "max_rows": 1012,
                "features_file": "test_features.csv",
                "targets_file": "test_targets.csv"
            },
            "validation": {
                "name": "Validation Set",
                "description": "Data used for cross-validation",
                "max_rows": 759,
                "features_file": "validation_features.csv", 
                "targets_file": "validation_targets.csv"
            },
            "holdout": {
                "name": "Holdout Set",
                "description": "Final evaluation set (never seen during training)",
                "max_rows": 506,
                "features_file": "holdout_features.csv",
                "targets_file": "holdout_targets.csv"
            }
        }
        
        # Command line arguments
        self.preset_dataset = dataset
        self.preset_row = row
        self.preset_random = random_row
        
        self.model = None
        self.scaler = None
        self.var_threshold = None
    
    def load_model_components(self):
        """Load all model components from the saved artifacts."""
        if self.model is None:
            with console.status("[bold blue]Loading trained model components..."):
                # Load the trained model
                self.model = joblib.load(self.model_dir / "hold3_final_model.pkl")
                
                # Load preprocessing components  
                self.scaler = joblib.load(self.model_dir / "hold3_scaler.pkl")
                self.var_threshold = joblib.load(self.model_dir / "hold3_var_threshold.pkl")
            
            # Create success table
            table = Table(title="Model Components Loaded", show_header=True, header_style="bold magenta")
            table.add_column("Component", style="cyan")
            table.add_column("Type", style="green")
            
            table.add_row("Model", f"{type(self.model).__name__}")
            table.add_row("Scaler", f"{type(self.scaler).__name__}")
            table.add_row("Variance Threshold", f"{type(self.var_threshold).__name__}")
            
            console.print(table)
    
    def display_dataset_info(self):
        """Display available datasets in a rich table."""
        table = Table(title="Available Datasets", show_header=True, header_style="bold magenta")
        table.add_column("Key", style="cyan", width=12)
        table.add_column("Name", style="green", width=20)
        table.add_column("Samples", style="yellow", justify="right", width=10)
        table.add_column("Description", style="white")
        
        for key, info in self.datasets.items():
            table.add_row(
                key,
                info["name"],
                str(info["max_rows"]),
                info["description"]
            )
        
        console.print(table)
    
    def select_dataset(self):
        """Interactive dataset selection."""
        if self.preset_dataset:
            if self.preset_dataset in self.datasets:
                console.print(f"[green]Using preset dataset: {self.preset_dataset}[/green]")
                return self.preset_dataset
            else:
                console.print(f"[red]Invalid preset dataset: {self.preset_dataset}[/red]")
                console.print("[yellow]Available options: train, test, validation, holdout[/yellow]")
                return None
        
        self.display_dataset_info()
        
        choices = []
        for key, info in self.datasets.items():
            choice_text = f"{info['name']} ({info['max_rows']} samples)"
            choices.append({"name": choice_text, "value": key})
        
        dataset_choice = inquirer.select(
            message="Select which dataset to run inference on:",
            choices=choices,
            default="holdout"
        ).execute()
        
        return dataset_choice
    
    def select_row_method(self, dataset_key):
        """Ask user if they want to choose specific row or random."""
        if self.preset_random:
            console.print("[green]Using preset: Random row selection[/green]")
            return "random"
        
        if self.preset_row is not None:
            console.print(f"[green]Using preset row: {self.preset_row}[/green]")
            return "specific"
        
        max_rows = self.datasets[dataset_key]["max_rows"]
        
        choices = [
            {"name": f"Choose specific row (1-{max_rows})", "value": "specific"},
            {"name": "Random row selection", "value": "random"}
        ]
        
        method = inquirer.select(
            message="How would you like to select the sample?",
            choices=choices,
            default="random"
        ).execute()
        
        return method
    
    def select_specific_row(self, dataset_key):
        """Get specific row number from user."""
        if self.preset_row is not None:
            max_rows = self.datasets[dataset_key]["max_rows"]
            if 1 <= self.preset_row <= max_rows:
                return self.preset_row
            else:
                console.print(f"[red]Invalid preset row: {self.preset_row}. Max rows: {max_rows}[/red]")
                return None
        
        max_rows = self.datasets[dataset_key]["max_rows"]
        
        row_num = inquirer.number(
            message=f"Enter row number (1-{max_rows}):",
            min_allowed=1,
            max_allowed=max_rows,
            default=1
        ).execute()
        
        return int(row_num)
    
    def get_random_row(self, dataset_key):
        """Get random row number."""
        max_rows = self.datasets[dataset_key]["max_rows"]
        row_num = random.randint(1, max_rows)
        console.print(f"[yellow]🎲 Randomly selected row: {row_num}[/yellow]")
        return row_num
    
    def load_sample_data(self, dataset_key, row_num):
        """Load the specific sample from the dataset files."""
        dataset_info = self.datasets[dataset_key]
        dataset_path = self.data_splits_path / dataset_key
        
        with console.status(f"[bold blue]Loading sample from {dataset_info['name']}, row {row_num}..."):
            # Load features (row_num + 1 because of header)
            features_file = dataset_path / dataset_info["features_file"]
            features_df = pd.read_csv(features_file)
            sample_features = features_df.iloc[row_num - 1].values  # -1 for 0-indexed
            
            # Load target
            targets_file = dataset_path / dataset_info["targets_file"] 
            targets_df = pd.read_csv(targets_file)
            actual_temp = targets_df.iloc[row_num - 1, 0]  # First column, 0-indexed
        
        return sample_features, actual_temp
    
    def run_inference(self, features):
        """Run inference on the provided features."""
        with console.status("[bold blue]Running inference..."):
            # Convert to numpy array and reshape for single sample
            X_sample = np.array(features).reshape(1, -1)
            
            # Apply preprocessing pipeline (same as training)
            # Apply variance threshold
            X_processed = self.var_threshold.transform(X_sample)
            
            # Apply scaling  
            X_scaled = self.scaler.transform(X_processed)
            
            # Run inference
            prediction = self.model.predict(X_scaled)[0]
        
        return prediction
    
    def display_results(self, dataset_key, row_num, prediction, actual_temp):
        """Display inference results with rich formatting."""
        dataset_name = self.datasets[dataset_key]["name"]
        
        # Calculate error metrics
        error = abs(prediction - actual_temp)
        percentage_error = (error / actual_temp) * 100 if actual_temp != 0 else float('inf')
        
        # Model confidence assessment
        if percentage_error < 1.0:
            confidence = "EXCELLENT 🌟"
            confidence_color = "green"
        elif percentage_error < 2.0:
            confidence = "VERY GOOD 🎯"
            confidence_color = "bright_green"
        elif percentage_error < 5.0:
            confidence = "GOOD ✅"
            confidence_color = "yellow"
        elif percentage_error < 10.0:
            confidence = "FAIR ⚡"
            confidence_color = "orange1"
        else:
            confidence = "NEEDS IMPROVEMENT ⚠️"
            confidence_color = "red"
        
        # Create results table
        table = Table(title="🎯 INFERENCE RESULTS", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=25)
        
        table.add_row("📊 Dataset", dataset_name)
        table.add_row("📍 Row", str(row_num))
        table.add_row("🔮 Predicted Temperature", f"{prediction:.3f}°C")
        table.add_row("🌡️ Actual Temperature", f"{actual_temp:.3f}°C")
        table.add_row("📏 Absolute Error", f"{error:.3f}°C")
        table.add_row("📊 Percentage Error", f"{percentage_error:.2f}%")
        table.add_row("✨ Prediction Confidence", Text(confidence, style=confidence_color))
        
        # Create panel with results
        panel = Panel(
            table,
            title="[bold blue]Model Inference Results[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        
        console.print(panel)
        
        return {
            'dataset': dataset_name,
            'row': row_num,
            'predicted_temp': prediction,
            'actual_temp': actual_temp,
            'error': error,
            'percentage_error': percentage_error,
            'confidence': confidence
        }
    
    def ask_continue(self):
        """Ask if user wants to continue."""
        return inquirer.confirm(
            message="Would you like to run another inference?",
            default=True
        ).execute()
    
    def run_single_inference(self):
        """Run a single inference (for command line usage)."""
        console.print(Panel(
            "[bold blue]🚀 Dataset 3 Model Inference[/bold blue]",
            title="Interactive ML Inference Tool",
            border_style="blue"
        ))
        
        # Load model components
        self.load_model_components()
        
        # Select dataset
        dataset_key = self.select_dataset()
        if dataset_key is None:
            return None
        
        # Select row method
        row_method = self.select_row_method(dataset_key)
        
        # Get row number
        if row_method == "specific":
            row_num = self.select_specific_row(dataset_key)
            if row_num is None:
                return None
        else:
            row_num = self.get_random_row(dataset_key)
        
        # Load sample data
        sample_features, actual_temp = self.load_sample_data(dataset_key, row_num)
        
        # Run inference
        prediction = self.run_inference(sample_features)
        
        # Display results
        results = self.display_results(dataset_key, row_num, prediction, actual_temp)
        
        return results
    
    def run_interactive_inference(self):
        """Main interactive inference workflow."""
        console.print(Panel(
            "[bold blue]🚀 Interactive Dataset 3 Model Inference[/bold blue]",
            title="Interactive ML Inference Tool",
            border_style="blue"
        ))
        
        # Load model components
        self.load_model_components()
        
        while True:
            try:
                # Select dataset
                dataset_key = self.select_dataset()
                if dataset_key is None:
                    continue
                
                # Select row method
                row_method = self.select_row_method(dataset_key)
                
                # Get row number
                if row_method == "specific":
                    row_num = self.select_specific_row(dataset_key)
                    if row_num is None:
                        continue
                else:
                    row_num = self.get_random_row(dataset_key)
                
                # Load sample data
                sample_features, actual_temp = self.load_sample_data(dataset_key, row_num)
                
                # Run inference
                prediction = self.run_inference(sample_features)
                
                # Display results
                results = self.display_results(dataset_key, row_num, prediction, actual_temp)
                
                # Ask if user wants to continue
                if not self.ask_continue():
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                if not self.ask_continue():
                    break
        
        console.print("[green]✅ Inference session completed![/green]")
    
    def extract_vna_features(self, vna_df):
        """Extract statistical features from VNA measurement data."""
        try:
            # Expected columns in VNA CSV (adjust based on actual format)
            # Assuming columns: Frequency, Return_Loss, Phase, Rs, SWR, Xs, Z_magnitude, Theta
            expected_measurements = [
                'Return_Loss', 'Phase', 'Rs', 'SWR', 'Xs', 'Z_magnitude', 'Theta'
            ]
            
            features = []
            
            for measurement in expected_measurements:
                if measurement in vna_df.columns:
                    data = vna_df[measurement].dropna()
                    
                    if len(data) > 0:
                        # Calculate 6 statistical features per measurement
                        features.extend([
                            data.mean(),           # Mean
                            data.std(),            # Standard deviation
                            data.min(),            # Minimum
                            data.max(),            # Maximum
                            data.quantile(0.25),   # 25th percentile
                            data.quantile(0.75)    # 75th percentile
                        ])
                    else:
                        # Fill with zeros if no data
                        features.extend([0.0] * 6)
                else:
                    console.print(f"[yellow]⚠️ Column '{measurement}' not found in VNA data[/yellow]")
                    features.extend([0.0] * 6)
            
            # Ensure we have exactly 42 features (7 measurements × 6 statistics)
            while len(features) < 42:
                features.append(0.0)
            features = features[:42]
            
            return np.array(features)
            
        except Exception as e:
            console.print(f"[red]❌ Error extracting VNA features: {e}[/red]")
            return np.zeros(42)  # Return zeros on error
    
    def process_vna_file(self, file_path):
        """Process a new VNA CSV file and run inference."""
        try:
            console.print(f"[blue]📊 Processing VNA file: {file_path.name}[/blue]")
            
            # Load VNA data
            vna_df = pd.read_csv(file_path)
            console.print(f"[cyan]📋 Loaded {len(vna_df)} rows, {len(vna_df.columns)} columns[/cyan]")
            
            # Extract features
            features = self.extract_vna_features(vna_df)
            
            # Run inference
            prediction = self.run_inference(features)
            self.prediction_count += 1
            self.last_prediction = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'file': file_path.name,
                'temperature': prediction,
                'count': self.prediction_count
            }
            
            # Display results
            self.display_vna_results(file_path.name, prediction, features)
            
            # Save result to log
            self.save_vna_result(file_path.name, prediction, features)
            
        except Exception as e:
            console.print(f"[red]❌ Error processing VNA file: {e}[/red]")
    
    def display_vna_results(self, filename, prediction, features):
        """Display VNA inference results."""
        # Create results table
        table = Table(title="🌡️ LIVE VNA TEMPERATURE PREDICTION", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=25)
        
        table.add_row("📁 VNA File", filename)
        table.add_row("⏰ Timestamp", time.strftime('%Y-%m-%d %H:%M:%S'))
        table.add_row("🌡️ Predicted Temperature", f"{prediction:.3f}°C")
        table.add_row("📊 Prediction Count", str(self.prediction_count))
        table.add_row("🔧 Features Extracted", f"{len(features)} statistical features")
        
        # Feature summary
        non_zero_features = np.count_nonzero(features)
        table.add_row("📈 Non-zero Features", f"{non_zero_features}/42")
        table.add_row("📏 Feature Range", f"[{features.min():.3f}, {features.max():.3f}]")
        
        # Create panel with results
        panel = Panel(
            table,
            title="[bold blue]Live VNA Inference Results[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        
        console.print(panel)
    
    def save_vna_result(self, filename, prediction, features):
        """Save VNA inference result to log file."""
        try:
            log_file = self.vna_data_path / "inference_log.csv"
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Create log entry
            log_entry = {
                'timestamp': timestamp,
                'vna_file': filename,
                'predicted_temperature': prediction,
                'prediction_count': self.prediction_count
            }
            
            # Save to CSV
            log_df = pd.DataFrame([log_entry])
            if log_file.exists():
                log_df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                log_df.to_csv(log_file, index=False)
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not save result to log: {e}[/yellow]")
    
    def start_f12_automation(self):
        """Start F12 automation for VNA data collection."""
        console.print("[green]🎹 Starting F12 automation - Press ESC to stop[/green]")
        self.f12_automation = True
        
        def f12_loop():
            while self.f12_automation:
                try:
                    console.print("[blue]⏱️ Pressing F12 to trigger VNA export...[/blue]")
                    keyboard.press_and_release('f12')
                    
                    # Wait 10 seconds as specified by user
                    for i in range(10, 0, -1):
                        if not self.f12_automation:
                            break
                        console.print(f"[yellow]⏳ Waiting {i}s for VNA export...[/yellow]", end='\r')
                        time.sleep(1)
                    
                    console.print("")  # New line after countdown
                    
                    # Wait additional time between cycles (adjustable)
                    time.sleep(5)
                    
                except Exception as e:
                    console.print(f"[red]❌ F12 automation error: {e}[/red]")
                    time.sleep(1)
        
        threading.Thread(target=f12_loop, daemon=True).start()
    
    def start_live_monitoring(self, enable_f12=True):
        """Start live VNA data monitoring."""
        console.print(Panel(
            "[bold green]🔴 LIVE VNA MONITORING STARTED[/bold green]\n\n"
            f"📁 Monitoring folder: {self.vna_data_path}\n"
            "📊 Waiting for new CSV files...\n"
            "🎹 F12 automation: " + ("ENABLED" if enable_f12 else "DISABLED") + "\n\n"
            "Press ESC to stop monitoring",
            title="Live VNA Temperature Prediction",
            border_style="green"
        ))
        
        # Load model components
        self.load_model_components()
        
        # Start file monitoring
        event_handler = VNADataHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.vna_data_path), recursive=False)
        observer.start()
        
        self.live_monitoring = True
        
        # Start F12 automation if requested
        if enable_f12:
            self.start_f12_automation()
        
        try:
            # Monitor for ESC key to stop
            console.print("[cyan]👀 Monitoring active - Press ESC to stop[/cyan]")
            while self.live_monitoring:
                if keyboard.is_pressed('esc'):
                    console.print("[yellow]🛑 ESC pressed - Stopping monitoring...[/yellow]")
                    break
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            console.print("[yellow]🛑 Ctrl+C pressed - Stopping monitoring...[/yellow]")
        
        finally:
            # Cleanup
            self.live_monitoring = False
            self.f12_automation = False
            observer.stop()
            observer.join()
            
            # Show final stats
            if self.last_prediction:
                console.print(Panel(
                    f"[bold blue]Final Stats[/bold blue]\n\n"
                    f"📊 Total predictions: {self.prediction_count}\n"
                    f"🌡️ Last temperature: {self.last_prediction['temperature']:.3f}°C\n"
                    f"📁 Last file: {self.last_prediction['file']}\n"
                    f"⏰ Last timestamp: {self.last_prediction['timestamp']}",
                    title="Live Monitoring Summary",
                    border_style="blue"
                ))
            
            console.print("[green]✅ Live VNA monitoring stopped[/green]")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Interactive inference tool for Dataset 3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Inference.py                           # Interactive mode
  python Inference.py --dataset holdout        # Use holdout dataset
  python Inference.py --dataset test --row 165 # Specific row in test set
  python Inference.py --dataset train --random # Random row in training set
  python Inference.py --single --dataset validation --row 100  # Single inference, no loop
  
  # Live VNA monitoring mode:
  python Inference.py --live                    # Start live VNA monitoring with F12 automation
  python Inference.py --live --no-f12          # Live monitoring without F12 automation

Available datasets: train, test, validation, holdout
VNA Live Data folder: /home/nate/Desktop/nates_recipe-V2/PI_PKG/VNA Live Data
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        choices=["train", "test", "validation", "holdout"],
        help="Dataset to use for inference"
    )
    
    parser.add_argument(
        "--row", "-r",
        type=int,
        help="Specific row number to use for inference"
    )
    
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random row selection"
    )
    
    parser.add_argument(
        "--single", "-s",
        action="store_true",
        help="Run single inference without looping"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    parser.add_argument(
        "--live", "-l",
        action="store_true",
        help="Start live VNA monitoring mode"
    )
    
    parser.add_argument(
        "--no-f12",
        action="store_true",
        help="Disable F12 automation in live mode"
    )
    
    args = parser.parse_args()
    
    # Create inference object
    inferencer = DatasetInference(
        dataset=args.dataset,
        row=args.row,
        random_row=args.random
    )
    
    # Handle list datasets option
    if args.list_datasets:
        console.print("[bold blue]Available Datasets:[/bold blue]")
        inferencer.display_dataset_info()
        return
    
    # Handle live monitoring mode
    if args.live:
        enable_f12 = not args.no_f12
        inferencer.start_live_monitoring(enable_f12=enable_f12)
        return
    
    # Validate conflicting arguments
    if args.row is not None and args.random:
        console.print("[red]❌ Error: Cannot use both --row and --random together[/red]")
        return
    
    # Run inference
    if args.single:
        result = inferencer.run_single_inference()
        if result is None:
            console.print("[red]❌ Inference failed[/red]")
    else:
        inferencer.run_interactive_inference()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Live VNA Monitoring and Temperature Inference
=============================================
Real-time script for VNA data monitoring and temperature prediction.
Processes live VNA CSV files using trained ML model for temperature inference.
"""

import argparse
import time
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import keyboard
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
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
    """Live VNA monitoring and temperature inference class."""

    def __init__(self):
        self.model_dir = Path("models")

        # VNA Live Data monitoring
        self.vna_data_path = Path("/home/nate/Desktop/nates_recipe-V2/PI_PKG/VNA Live Data")
        self.vna_data_path.mkdir(parents=True, exist_ok=True)
        self.live_monitoring = False
        self.f12_automation = False
        self.last_prediction = None
        self.prediction_count = 0

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



    def run_inference(self, features):
        """Run inference on the provided features."""
        with console.status("[bold blue]Running inference..."):
            # Convert to numpy array and reshape for single sample
            x_sample = np.array(features).reshape(1, -1)

            # Apply preprocessing pipeline (same as training)
            # Apply variance threshold
            x_processed = self.var_threshold.transform(x_sample)

            # Apply scaling
            x_scaled = self.scaler.transform(x_processed)

            # Run inference
            prediction = self.model.predict(x_scaled)[0]

        return prediction



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
                    warning_msg = f"Column '{measurement}' not found in VNA data"
                    console.print(f"[yellow]{warning_msg}[/yellow]")
                    features.extend([0.0] * 6)

            # Ensure we have exactly 42 features (7 measurements × 6 statistics)
            while len(features) < 42:
                features.append(0.0)
            features = features[:42]

            return np.array(features)

        except (ValueError, KeyError, AttributeError) as exc:
            console.print(f"[red]Error extracting VNA features: {exc}[/red]")
            return np.zeros(42)  # Return zeros on error

    def process_vna_file(self, file_path):
        """Process a new VNA CSV file and run inference."""
        try:
            console.print(f"[blue]Processing VNA file: {file_path.name}[/blue]")

            # Load VNA data
            vna_df = pd.read_csv(file_path)
            console.print(f"[cyan]Loaded {len(vna_df)} rows, {len(vna_df.columns)} columns[/cyan]")

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
            self.display_vna_results(file_path.name, prediction)

            # Save result to log
            self.save_vna_result(file_path.name, prediction)

        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
            console.print(f"[red]Error processing VNA file: {exc}[/red]")

    def display_vna_results(self, filename, prediction):
        """Display VNA inference results."""
        # Create results table
        table = Table(title="LIVE VNA TEMPERATURE PREDICTION", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=25)

        table.add_row("VNA File", filename)
        table.add_row("Timestamp", time.strftime('%Y-%m-%d %H:%M:%S'))
        table.add_row("Predicted Temperature", f"{prediction:.3f}°C")
        table.add_row("Prediction Count", str(self.prediction_count))

        # Create panel with results
        panel = Panel(
            table,
            title="[bold blue]Live VNA Inference Results[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )

        console.print(panel)

    def save_vna_result(self, filename, prediction):
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

        except (OSError, PermissionError) as exc:
            console.print(f"[yellow]Could not save result to log: {exc}[/yellow]")

    def start_f12_automation(self):
        """Start F12 automation for VNA data collection."""
        console.print("[green]Starting F12 automation - Press ESC to stop[/green]")
        self.f12_automation = True

        def f12_loop():
            while self.f12_automation:
                try:
                    console.print("[blue]Pressing F12 to trigger VNA export...[/blue]")
                    keyboard.press_and_release('f12')

                    # Wait 10 seconds as specified by user
                    for i in range(10, 0, -1):
                        if not self.f12_automation:
                            break
                        console.print(f"[yellow]Waiting {i}s for VNA export...[/yellow]", end='\r')
                        time.sleep(1)

                    console.print("")  # New line after countdown

                    # Wait additional time between cycles (adjustable)
                    time.sleep(5)

                except (OSError, ImportError) as exc:
                    console.print(f"[red]F12 automation error: {exc}[/red]")
                    time.sleep(1)

        threading.Thread(target=f12_loop, daemon=True).start()

    def start_live_monitoring(self, enable_f12=True):
        """Start live VNA data monitoring."""
        console.print(Panel(
            "[bold green]LIVE VNA MONITORING STARTED[/bold green]\n\n"
            f"Monitoring folder: {self.vna_data_path}\n"
            "Waiting for new CSV files...\n"
            "F12 automation: " + ("ENABLED" if enable_f12 else "DISABLED") + "\n\n"
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
            console.print("[cyan]Monitoring active - Press ESC to stop[/cyan]")
            while self.live_monitoring:
                if keyboard.is_pressed('esc'):
                    console.print("[yellow]ESC pressed - Stopping monitoring...[/yellow]")
                    break
                time.sleep(0.1)

        except KeyboardInterrupt:
            console.print("[yellow]Ctrl+C pressed - Stopping monitoring...[/yellow]")

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
                    f"Total predictions: {self.prediction_count}\n"
                    f"Last temperature: {self.last_prediction['temperature']:.3f}°C\n"
                    f"Last file: {self.last_prediction['file']}\n"
                    f"Last timestamp: {self.last_prediction['timestamp']}",
                    title="Live Monitoring Summary",
                    border_style="blue"
                ))

            console.print("[green]Live VNA monitoring stopped[/green]")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Live VNA inference tool for real-time temperature prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Inference.py                    # Start live VNA monitoring with F12 automation
  python Inference.py --no-f12          # Live monitoring without F12 automation

VNA Live Data folder: /home/nate/Desktop/nates_recipe-V2/PI_PKG/VNA Live Data
        """
    )

    parser.add_argument(
        "--no-f12",
        action="store_true",
        help="Disable F12 automation in live mode"
    )

    args = parser.parse_args()

    # Create inference object
    inferencer = LiveVnaInference()

    # Start live monitoring mode
    enable_f12 = not args.no_f12
    inferencer.start_live_monitoring(enable_f12=enable_f12)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Live VNA Monitoring and Temperature Inference using hold5 model
==============================================================
Real-time script for VNA data monitoring and temperature prediction.
Uses Java VNAhl command to capture VNA data, then runs inference using hold5 model.

VERSION: 1.0.9
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
import re
try:
    import serial
except Exception:
    serial = None
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

console = Console()

# Print versions immediately when script starts
console.print("[bold blue]=== VERSION INFO ===[/bold blue]")
console.print(f"Script Version: 1.0.9")
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

    def __init__(self, hw_points: int = 10001, read_arduino: bool = False, arduino_port: str = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_Due_Prog._Port_24336303633351406111-if00", arduino_baud: int = 115200, model_dir: str | Path | None = None):
        # Use hold5 model directory (allow override via CLI)
        self.model_dir = Path(model_dir) if model_dir else Path("best_model_hold5")
        
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
        self.using_inference_ready = False
        self.inference_ready_model_path = None
        self.model_params: dict | None = None
        # Hardware sweep points (what we ask VNA to capture)
        self.hw_points = int(hw_points)
        # Model expects features built from 4 columns × 10,001 points
        self.model_points = 10001
        # Expected raw features will be fixed after scaler is loaded (n_features_in_)
        self.expected_raw_features = 4 * self.model_points
        # Arduino config
        self.read_arduino = bool(read_arduino)
        self.arduino_port = arduino_port
        self.arduino_baud = int(arduino_baud)
        self.temp_regex = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?C?", re.IGNORECASE)
        self.latest_arduino_temp = None
        self._arduino_thread = None
        self._arduino_stop = threading.Event()
        # Do not auto-detect or fallback; use fixed by-id port when reading on demand

    def load_model_components(self):
        """Load all model components from hold5 saved artifacts or inference-ready folder."""
        if self.model is None:
            with console.status("[bold blue]Loading hold5 model components..."):
                # Load preprocessing first so we can derive final input dimensionality
                # Support both legacy (var_threshold + kbest + scaler) and inference-ready (scaler only)
                scaler_path_candidates = [
                    self.model_dir / "scaler.pkl",
                    self.model_dir / "hold5_scaler.pkl",
                ]
                for spath in scaler_path_candidates:
                    if spath.exists():
                        self.scaler = joblib.load(spath)
                        self.using_inference_ready = (spath.name == "scaler.pkl") or (self.model_dir / "model" / "model_def.py").exists()
                        break
                if self.scaler is None:
                    raise FileNotFoundError("No scaler file found (scaler.pkl or hold5_scaler.pkl)")

                # Optional preselectors
                vt_candidates = [self.model_dir / "var_thresh.pkl", self.model_dir / "hold5_var_threshold.pkl"]
                kb_candidates = [self.model_dir / "kbest_selector.pkl", self.model_dir / "hold5_kbest_selector.pkl"]
                self.var_threshold = next((joblib.load(p) for p in vt_candidates if p.exists()), None)
                self.kbest_selector = next((joblib.load(p) for p in kb_candidates if p.exists()), None)
                # Load model params if present (controls preprocessor order/toggles)
                params_path = self.model_dir / "model_params.json"
                if params_path.exists():
                    try:
                        import json
                        with open(params_path, 'r', encoding='utf-8') as f:
                            self.model_params = json.load(f)
                    except Exception:
                        self.model_params = None

                # Debug scaler parameters to verify correctness
                try:
                    scaler_info = {
                        "type": type(self.scaler).__name__,
                        "n_features_in_": getattr(self.scaler, "n_features_in_", None),
                        "with_mean": getattr(self.scaler, "with_mean", None),
                        "with_std": getattr(self.scaler, "with_std", None),
                    }
                    # Adjust expected_raw_features to match scaler input size when it's a clean multiple of model_points
                    try:
                        n_in = int(getattr(self.scaler, "n_features_in_", 0))
                        if n_in > 0 and n_in % self.model_points == 0:
                            self.expected_raw_features = n_in
                    except Exception:
                        pass
                    mean_attr = getattr(self.scaler, "mean_", None)
                    scale_attr = getattr(self.scaler, "scale_", None)
                    if mean_attr is not None and hasattr(mean_attr, "shape"):
                        scaler_info.update({
                            "mean_shape": tuple(mean_attr.shape),
                            "mean_min": float(np.min(mean_attr)),
                            "mean_max": float(np.max(mean_attr)),
                        })
                    if scale_attr is not None and hasattr(scale_attr, "shape"):
                        scaler_info.update({
                            "scale_shape": tuple(scale_attr.shape),
                            "scale_min": float(np.min(scale_attr)),
                            "scale_max": float(np.max(scale_attr)),
                        })
                    console.print("Scaler details:", scaler_info, style="cyan")
                except Exception as _e:
                    console.print(f"Failed to inspect scaler: {_e}", style="yellow")

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
                
                # Try loading a FULL PyTorch model first (torch.save(model, ...))
                # Falls back to constructing model from state_dict if needed.
                full_model_loaded = False
                # Support both legacy and inference-ready model filenames
                model_pt = self.model_dir / "hold5_final_model.pt"
                model_pt_alt = self.model_dir / "resnet_model.pth"
                model_dir = self.model_dir / "hold5_final_model"
                try:
                    if model_pt.exists():
                        console.print(f"Loading full model object from: {model_pt}", style="blue")
                        self.model = torch.load(model_pt, map_location='cpu')
                        self.model.eval()
                        full_model_loaded = True
                        console.print("Full model loaded and set to eval()", style="green")
                    elif model_pt_alt.exists():
                        # Will attempt state_dict loading using model class from model_def.py
                        self.inference_ready_model_path = model_pt_alt
                    elif (model_dir / "data.pkl").exists():
                        # Support case where .pt was unzipped into a directory
                        console.print(f"Loading full model from unpacked directory: {model_dir}", style="blue")
                        self.model = torch.load(model_dir / "data.pkl", map_location='cpu')
                        self.model.eval()
                        full_model_loaded = True
                        console.print("Full model (unpacked) loaded and set to eval()", style="green")
                except Exception as e:
                    console.print(f"Full-model load failed, will try state_dict path: {e}", style="yellow")

                if not full_model_loaded:
                    # Determine final feature count after preprocessing pipeline
                    try:
                        # Prefer final post-preprocessing dimensionality
                        final_input_dim = None
                        if self.kbest_selector is not None:
                            k_val = getattr(self.kbest_selector, 'k', None)
                            if isinstance(k_val, int) and k_val > 0:
                                final_input_dim = k_val
                            else:
                                support = getattr(self.kbest_selector, 'get_support', None)
                                if callable(support):
                                    try:
                                        final_input_dim = int(support().sum())
                                    except Exception:
                                        final_input_dim = None
                        if final_input_dim is None and self.model_params is not None:
                            kbest_k = self.model_params.get('kbest_k')
                            if isinstance(kbest_k, int) and kbest_k > 0:
                                final_input_dim = kbest_k
                        if final_input_dim is None:
                            # Fallback to scaler-reported input
                            final_input_dim = int(getattr(self.scaler, 'n_features_in_', 0))
                        if not isinstance(final_input_dim, int) or final_input_dim <= 0:
                            # Last-chance transform simulation
                            dummy = np.random.randn(1, self.expected_raw_features).astype(np.float32)
                            x = dummy
                            if self.var_threshold is not None:
                                x = self.var_threshold.transform(x)
                            if self.scaler is not None:
                                try:
                                    x = self.scaler.transform(x)
                                except Exception:
                                    pass
                            if self.kbest_selector is not None:
                                x = self.kbest_selector.transform(x)
                            final_input_dim = int(x.shape[1])
                    except Exception:
                        # Fallback chain using selector metadata
                        final_input_dim = getattr(self.kbest_selector, 'k', None) or getattr(self.scaler, 'n_features_in_', None)
                        if final_input_dim is None or final_input_dim == 'all':
                            support = getattr(self.kbest_selector, 'get_support', None)
                            if callable(support):
                                try:
                                    final_input_dim = int(support().sum())
                                except Exception:
                                    final_input_dim = None
                        if not isinstance(final_input_dim, int) or final_input_dim <= 0:
                            final_input_dim = self.expected_raw_features

                    # Create model instance using provided model_def if available, else dynamic
                    try:
                        sd_path_to_use = self.inference_ready_model_path or (model_pt_alt if model_pt_alt.exists() else model_pt)
                        sd = torch.load(sd_path_to_use, map_location='cpu')

                        # If inference-ready includes model_def.py, import and instantiate
                        model_def_path = self.model_dir / "model" / "model_def.py"
                        if model_def_path.exists():
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("ir_model_def", str(model_def_path))
                            assert spec and spec.loader
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            ModelClass = getattr(mod, 'CustomResNet', None)
                            if ModelClass is None:
                                raise RuntimeError("CustomResNet not found in model_def.py")
                            # Load params if present
                            params_path = self.model_dir / "model_params.json"
                            hidden_dim = 128
                            num_blocks = 2
                            dropout = 0.1
                            residual_init = 1.0
                            if params_path.exists():
                                try:
                                    import json
                                    with open(params_path, 'r', encoding='utf-8') as f:
                                        params = json.load(f)
                                    hidden_dim = int(params.get('hidden_dim', hidden_dim))
                                    num_blocks = int(params.get('num_blocks', num_blocks))
                                    dropout = float(params.get('dropout', dropout))
                                    residual_init = float(params.get('residual_init', residual_init))
                                except Exception:
                                    pass
                            self.model = ModelClass(
                                input_dim=final_input_dim,
                                hidden_dim=hidden_dim,
                                num_blocks=num_blocks,
                                dropout=dropout,
                                residual_init=residual_init,
                            )
                            # Determine potential nesting in sd
                            from collections import OrderedDict
                            state_to_load = None
                            if isinstance(sd, OrderedDict):
                                state_to_load = sd
                            elif isinstance(sd, dict):
                                for key in ('state_dict', 'model_state_dict', 'model', 'net', 'weights'):
                                    if key in sd and isinstance(sd[key], (dict, OrderedDict)):
                                        state_to_load = sd[key]
                                        break
                            if state_to_load is None:
                                raise RuntimeError("Unsupported state dict format for inference-ready model")
                            self.model.load_state_dict(state_to_load, strict=True)
                            self.model.eval()
                            console.print("Loaded model from inference-ready model_def.py and resnet_model.pth", style="green")
                        else:
                            # Fallback to previously inferred architecture (legacy)
                            from collections import OrderedDict
                            if not isinstance(sd, OrderedDict):
                                raise RuntimeError("Legacy fallback requires flat OrderedDict state_dict")
                            linear_keys = [(k, v) for k, v in sd.items() if k.startswith('backbone.') and k.endswith('.weight') and hasattr(v, 'ndim') and int(getattr(v, 'ndim', 0)) == 2]
                            if not linear_keys:
                                raise RuntimeError("No backbone linear layers found in state_dict")
                            # Minimal fallback: reuse simple CustomResNet
                            self.model = CustomResNet(
                                input_dim=final_input_dim,
                                hidden_dim=128,
                                num_blocks=2,
                                dropout=0.1,
                            )
                            self.model.load_state_dict(sd, strict=False)
                            self.model.eval()
                            console.print("Loaded legacy state_dict with fallback CustomResNet (strict=False)", style="yellow")
                    except Exception as e:
                        console.print(f"Error constructing/loading model: {e}", style="red")
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
                # Adjusted frequency range and steps; steps are configurable for latency/accuracy
            cmd = [
                "java",
                "-Dpurejavacomm.log=false",
                "-Dpurejavacomm.debug=false",
                "-Dfstart=45000000",
                    "-Dfstop=60000000",  # Updated range per D5 spec
                    f"-Dfsteps={self.hw_points}",
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

            # Clean NaNs/Infs prior to sklearn transforms
            try:
                nan_count = int(np.isnan(x_sample).sum())
                inf_count = int(np.isinf(x_sample).sum())
                if nan_count or inf_count:
                    console.print(
                        f"Cleaning input: NaNs={nan_count}, Infs={inf_count}",
                        style="yellow",
                    )
                x_sample = np.nan_to_num(x_sample, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as _e:
                console.print(f"Failed to sanitize input array: {_e}", style="yellow")

            # Apply preprocessing pipeline (same as training): scaler -> kbest
            # Enforce order: first scaler (expects 20002 for D5), then KBest to final_dim
            x_pre = x_sample
            # Scaler
            if self.scaler is not None:
                try:
                    sc_in = getattr(self.scaler, 'n_features_in_', x_pre.shape[1])
                    if int(sc_in) == int(x_pre.shape[1]):
                        x_pre = self.scaler.transform(x_pre)
                    else:
                        console.print(f"Skipping Scaler: expects {sc_in} features but have {x_pre.shape[1]}", style="yellow")
                except Exception as _e:
                    console.print(f"Scaler transform failed ({_e}); skipping", style="yellow")
            # KBest
            if self.kbest_selector is not None:
                try:
                    x_pre = self.kbest_selector.transform(x_pre)
                except Exception as _e:
                    console.print(f"KBest transform failed ({_e}); proceeding without KBest", style="yellow")
            # Final matrix post-preprocessing
            x_kbest = x_pre
            try:
                _xk = np.asarray(x_kbest)
                console.print(
                    "x_kbest stats:",
                    {
                        "shape": tuple(_xk.shape),
                        "min": float(_xk.min()),
                        "max": float(_xk.max()),
                        "mean": float(_xk.mean()),
                        "std": float(_xk.std()),
                    },
                    style="cyan",
                )
            except Exception as _e:
                console.print(f"Failed to compute x_kbest stats: {_e}", style="yellow")
            
            # Scaler already applied above if present
            x_scaled = x_kbest
            try:
                _xs = np.asarray(x_scaled)
                console.print(
                    "x_scaled stats:",
                    {
                        "shape": tuple(_xs.shape),
                        "min": float(_xs.min()),
                        "max": float(_xs.max()),
                        "mean": float(_xs.mean()),
                        "std": float(_xs.std()),
                    },
                    style="cyan",
                )
            except Exception as _e:
                console.print(f"Failed to compute x_scaled stats: {_e}", style="yellow")
            
            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x_scaled)

            # Print tensor stats prior to model inference for debugging
            try:
                console.print(
                    "x_tensor stats:",
                    {
                        "shape": tuple(x_tensor.shape),
                        "min": float(x_tensor.min().item()),
                        "max": float(x_tensor.max().item()),
                        "mean": float(x_tensor.mean().item()),
                        "std": float(x_tensor.std().item()),
                    },
                    style="cyan",
                )
            except Exception as _e:
                console.print(f"Failed to compute x_tensor stats: {_e}", style="yellow")
            
            # Run prediction with PyTorch model
            with torch.no_grad():
                prediction = self.model(x_tensor).item()
            
            return prediction

    def read_arduino_temp(self, port: str | None = None, baud: int | None = None, timeout: float = 2.0):
        """Send 'TEMP' twice and return the second full line using a single session and read-until for stability."""
        if not self.read_arduino:
            return None
        if serial is None:
            console.print("pyserial not installed; skipping Arduino read", style="yellow")
            return None
        fixed_port = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_Due_Prog._Port_24336303633351406111-if00"
        fixed_baud = 115200
        ser = None
        try:
            ser = serial.Serial(fixed_port, fixed_baud, timeout=1, write_timeout=1)  # type: ignore
            try:
                ser.setDTR(False)
            except Exception:
                pass
            time.sleep(3)
            # First probe (discard)
            ser.write(b"TEMP\r\n")
            _ = ser.read_until(b"\n")
            # Second probe (use)
            ser.write(b"TEMP\r\n")
            raw = ser.read_until(b"\n")
            line = raw.decode('utf-8', 'ignore').strip()
            if not line:
                return None
            try:
                val = float(line)
                self.latest_arduino_temp = val
                return val
            except ValueError:
                return None
        except serial.SerialException as e:  # type: ignore
            console.print(f"[red]Arduino read error: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Arduino read error: {e}[/red]")
            return None
        finally:
            if ser is not None:
                try:
                    ser.close()  # type: ignore
                except Exception:
                    pass

    def start_arduino_reader(self):
        if serial is None:
            return
        if self._arduino_thread and self._arduino_thread.is_alive():
            return
        def _reader():
            while not self._arduino_stop.is_set():
                try:
                    if not os.path.exists(self.arduino_port):
                        detected = self.detect_arduino_port()
                        if detected:
                            self.arduino_port = detected
                    with serial.Serial(self.arduino_port, self.arduino_baud, timeout=1.0) as ser:  # type: ignore
                        ser.reset_input_buffer()
                        ser.reset_output_buffer()
                        line = ser.readline().decode(errors='ignore').strip()
                        if not line:
                            continue
                        m = self.temp_regex.search(line)
                        if m:
                            try:
                                val = float(m.group(1))
                                if -40.0 <= val <= 200.0:
                                    self.latest_arduino_temp = val
                            except Exception:
                                continue
                except Exception:
                    # Sleep briefly before retrying
                    time.sleep(0.5)
                    continue
        self._arduino_stop.clear()
        self._arduino_thread = threading.Thread(target=_reader, daemon=True)
        self._arduino_thread.start()

    def stop_arduino_reader(self):
        self._arduino_stop.set()
        if self._arduino_thread:
            self._arduino_thread.join(timeout=1.0)

    def detect_arduino_port(self) -> str | None:
        """Detect Arduino Due Programming Port by USB VID:PID 2341:003d via sysfs.
        Falls back to /dev/serial/by-id symlinks; avoids selecting VNA ttyUSB0.
        """
        try:
            target_vid = "2341"  # Arduino SA
            target_pid = "003d"  # Due Programming Port

            def match_sysfs(dev_path: str) -> bool:
                name = os.path.basename(dev_path)
                sys_dev = f"/sys/class/tty/{name}/device"
                if not os.path.exists(sys_dev):
                    return False
                # ascend up to 5 levels to find idVendor/idProduct
                cur = os.path.realpath(sys_dev)
                for _ in range(6):
                    vid_path = os.path.join(cur, "idVendor")
                    pid_path = os.path.join(cur, "idProduct")
                    if os.path.exists(vid_path) and os.path.exists(pid_path):
                        try:
                            with open(vid_path, 'r', encoding='utf-8') as f:
                                vid = f.read().strip().lower()
                            with open(pid_path, 'r', encoding='utf-8') as f:
                                pid = f.read().strip().lower()
                            return (vid == target_vid and pid == target_pid)
                        except Exception:
                            return False
                    cur = os.path.dirname(cur)
                return False

            # Prefer ACM devices first
            for glob_base in ("/dev/ttyACM", "/dev/ttyUSB"):
                for idx in range(0, 8):
                    dev = f"{glob_base}{idx}"
                    if os.path.exists(dev) and match_sysfs(dev):
                        return dev

            # Fallback to by-id symlinks
            by_id_dir = "/dev/serial/by-id"
            if os.path.isdir(by_id_dir):
                for entry in os.listdir(by_id_dir):
                    lower = entry.lower()
                    if "arduino" in lower and ("due" in lower or "prog" in lower):
                        link = os.path.join(by_id_dir, entry)
                        try:
                            resolved = os.path.realpath(link)
                            if os.path.exists(resolved):
                                return resolved
                        except Exception:
                            continue
        except Exception:
            pass
        return None

    def extract_vna_features(self, vna_df):
        """Extract features from VNA data.
        - Legacy: 4 channels (Return Loss, Phase, Xs, Rs) -> 40004 features
        - Inference-ready: 3 channels (Return Loss, Phase, |Z|) -> 30003 features
        """
        try:
            # Determine target channels based on scaler-reported expected input
            use_three_channels = False
            num_channels = 4
            try:
                n_in = int(getattr(self.scaler, 'n_features_in_', 0))
                if n_in > 0 and (n_in % self.model_points) == 0:
                    channels_expected = n_in // self.model_points
                    if channels_expected in (2, 3, 4):
                        num_channels = channels_expected
                        use_three_channels = (channels_expected == 3)
            except Exception:
                pass
            # Extract raw values from selected channels (NOT frequency)
            features = []
            target_len = self.model_points

            def resample_to_length(arr: np.ndarray, target: int) -> np.ndarray:
                if arr.size == target:
                    return arr.astype(np.float32, copy=False)
                # Robust linear interpolation on normalized index
                x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=True)
                x_new = np.linspace(0.0, 1.0, num=target, endpoint=True)
                return np.interp(x_new, x_old, arr.astype(float)).astype(np.float32)
            
            # Show all available columns for debugging
            console.print(f"Available columns: {list(vna_df.columns)}", style="cyan")
            
            # Try different possible column names for Return Loss (support: s11_db, db, etc.)
            return_loss_col = None
            preferred_aliases = {
                's11_db', 's11db', 'returnloss', 'return_loss', 'returnlosdb', 'returnlossdb', 'db'
            }
            for col in vna_df.columns:
                lower_col = col.lower()
                normalized = re.sub(r"[^a-z0-9]+", "", lower_col)
                if (
                    'return' in lower_col or
                    'loss' in lower_col or
                    's11' in lower_col or
                    normalized in preferred_aliases or
                    lower_col == 'db' or
                    lower_col.endswith('(db)')
                ):
                    return_loss_col = col
                    break
            
            rl = None
            if return_loss_col:
                return_loss = vna_df[return_loss_col].values
                rl = resample_to_length(return_loss, target_len)
                console.print(f"Prepared {len(rl)} Return Loss features from column '{return_loss_col}' (resampled)", style="green")
            else:
                console.print("Missing Return Loss column - tried: return, loss, s11", style="red")
                return None
            
            # Try different possible column names for Phase
            phase_col = None
            for col in vna_df.columns:
                if 'phase' in col.lower():
                    phase_col = col
                    break
            
            ph = None
            if phase_col:
                phase = vna_df[phase_col].values
                ph = resample_to_length(phase, target_len)
                console.print(f"Prepared {len(ph)} Phase features from column '{phase_col}' (resampled)", style="green")
            else:
                console.print("Missing Phase column", style="red")
                return None
            
            # Determine expected channels from scaler
            channels_expected = num_channels
            if channels_expected == 4:
                # Inference-ready 4-channel training order: [s11_db, db, phase, Xs]
                # db is a duplicate of s11_db when separate column isn't available
                # Xs required
                xs_col = None
                for col in vna_df.columns:
                    if col.lower() == 'xs' or 'reactance' in col.lower():
                        xs_col = col
                        break
                if xs_col is None:
                    console.print("Missing Xs column for 4-channel pipeline", style="red")
                    return None
                # Assemble in training order
                s11r = rl
                phr = ph
                features.extend(s11r.tolist())           # s11_db
                features.extend(s11r.tolist())           # db (duplicate of s11_db)
                features.extend(phr.tolist())            # phase
                xs = vna_df[xs_col].values
                xsr = resample_to_length(xs, target_len)
                features.extend(xsr.tolist())            # Xs
                console.print("Assembled 4-channel features in order: s11_db, db(dup), phase, Xs", style="green")
            elif channels_expected == 3:
                # Third channel preference: |Z| if present, else SWR, else sqrt(Rs^2 + Xs^2)
                mag_col = None
                for col in vna_df.columns:
                    lc = col.lower()
                    if '|z|' in lc or lc.strip() in ('|z|', 'mag', 'magnitude'):  # common name is '|Z|'
                        mag_col = col
                        break
                if mag_col is None:
                    for col in vna_df.columns:
                        if col.lower() == 'swr':
                            mag_col = col
                            break
                if mag_col is not None:
                    arr = vna_df[mag_col].values
                    mr = resample_to_length(arr, target_len)
                    features.extend(mr.tolist())
                    console.print(f"Added {len(mr)} third-channel features from column '{mag_col}' (resampled)", style="green")
                else:
                    # Fallback to sqrt(Rs^2 + Xs^2)
                    rs_col, xs_col = None, None
                    for col in vna_df.columns:
                        if rs_col is None and (col.lower() == 'rs' or 'resistance' in col.lower()):
                            rs_col = col
                        if xs_col is None and (col.lower() == 'xs' or 'reactance' in col.lower()):
                            xs_col = col
                    if rs_col is None or xs_col is None:
                        console.print("Missing both |Z|/SWR and Rs/Xs for fallback magnitude", style="red")
                        return None
                    rs = vna_df[rs_col].values.astype(float)
                    xs = vna_df[xs_col].values.astype(float)
                    mag = np.sqrt(np.square(rs) + np.square(xs))
                    mr = resample_to_length(mag, target_len)
                    features.extend(mr.tolist())
                    console.print(f"Added {len(mr)} magnitude sqrt(Rs^2+Xs^2) features (resampled)", style="green")
            elif channels_expected == 2:
                # Two-channel pipeline for D5: [phase, rs]. If Rs missing/degenerate, derive Rs = |Z| * cos(Theta).
                # Phase already prepared as 'ph'
                rs_vals_raw = None
                rs_col = next((c for c in vna_df.columns if c.lower() == 'rs' or 'resistance' in c.lower()), None)
                if rs_col is not None:
                    try:
                        rs_vals_raw = pd.to_numeric(vna_df[rs_col], errors='coerce').to_numpy()
                    except Exception:
                        rs_vals_raw = None
                # If Rs missing or near-constant, derive from |Z| and Theta when available
                derive_from_zt = False
                if rs_vals_raw is None or not np.isfinite(rs_vals_raw).any():
                    derive_from_zt = True
                else:
                    try:
                        rs_std = float(np.nanstd(rs_vals_raw))
                        if rs_std < 1e-6:
                            derive_from_zt = True
                    except Exception:
                        derive_from_zt = True
                if derive_from_zt:
                    z_col = next((c for c in vna_df.columns if c.strip().lower() in {'|z|', 'z', 'mag', 'magnitude'} or '|z|' in c.lower()), None)
                    th_col = next((c for c in vna_df.columns if c.strip().lower() == 'theta'), None)
                    if z_col is None or th_col is None:
                        console.print("Cannot derive Rs: missing |Z| or Theta columns", style="red")
                        return None
                    try:
                        z_vals = pd.to_numeric(vna_df[z_col], errors='coerce').to_numpy(dtype=float)
                        th_vals_deg = pd.to_numeric(vna_df[th_col], errors='coerce').to_numpy(dtype=float)
                        th_rad = np.deg2rad(th_vals_deg)
                        rs_vals_raw = z_vals * np.cos(th_rad)
                    except Exception:
                        console.print("Failed to derive Rs from |Z| and Theta", style="red")
                        return None
                if ph is None or rs_vals_raw is None:
                    console.print("Missing Phase or Rs for 2-channel D5 pipeline", style="red")
                    return None
                rsr = resample_to_length(rs_vals_raw, target_len)
                # Order: [phase, rs]
                features.extend(ph.tolist())
                features.extend(rsr.tolist())
                console.print("Assembled 2-channel D5 features: phase, rs(Xs fallback)", style="green")
            else:
                # Legacy 4-channel pipeline: Return Loss, Phase, Xs, Rs
                xs_col = None
                for col in vna_df.columns:
                    if col.lower() == 'xs' or 'reactance' in col.lower():
                        xs_col = col
                        break
                rs_col = None
                for col in vna_df.columns:
                    if col.lower() == 'rs' or 'resistance' in col.lower():
                        rs_col = col
                        break
                if xs_col is None or rs_col is None:
                    console.print("Missing Xs or Rs column for legacy 4-channel pipeline", style="red")
                    return None
                xs = vna_df[xs_col].values
                xsr = resample_to_length(xs, target_len)
                features.extend(xsr.tolist())
                rs = vna_df[rs_col].values
                rsr = resample_to_length(rs, target_len)
                features.extend(rsr.tolist())
                console.print("Assembled legacy 4-channel features: RL, phase, Xs, Rs", style="green")
            
            console.print(f"Total features extracted: {len(features)} (channels={channels_expected}, points={target_len})", style="green")
            return features
            
        except Exception as e:
            console.print(f"Error extracting features: {e}", style="red")
            return None

    def _is_valid_capture(self, vna_df: pd.DataFrame) -> bool:
        """Heuristic checks to avoid degenerate captures causing OOD predictions.
        - SWR non-finite or '?' across majority of rows -> invalid
        - Theta ~ 90 degrees across majority of rows -> likely open circuit
        - Rs present but near-constant/zero variance across rows -> fallback to Xs or invalid
        Returns True when capture is considered usable.
        """
        try:
            n = len(vna_df)
            if n <= 0:
                return False
            # SWR validity
            if 'SWR' in vna_df.columns:
                swr = pd.to_numeric(vna_df['SWR'], errors='coerce')
                bad_ratio = float(swr.isna().mean())
                if bad_ratio > 0.5:
                    return False
            # Theta near 90 degrees check
            th_col = next((c for c in vna_df.columns if c.lower().strip() == 'theta'), None)
            if th_col is not None:
                th = pd.to_numeric(vna_df[th_col], errors='coerce')
                mask = th.notna()
                if mask.any():
                    near_90 = np.isclose(th[mask].to_numpy(dtype=float), 90.0, atol=1.0).mean()
                    if float(near_90) > 0.7:
                        return False
            # Rs variance check (if present)
            rs_col = next((c for c in vna_df.columns if c.lower() == 'rs' or 'resistance' in c.lower()), None)
            if rs_col is not None:
                rs = pd.to_numeric(vna_df[rs_col], errors='coerce')
                rs_var = float(np.nanstd(rs.to_numpy(dtype=float)))
                if rs_var < 1e-6:
                    # try Xs variance instead
                    xs_col = next((c for c in vna_df.columns if c.lower() == 'xs' or 'reactance' in c.lower()), None)
                    if xs_col is not None:
                        xs = pd.to_numeric(vna_df[xs_col], errors='coerce')
                        xs_var = float(np.nanstd(xs.to_numpy(dtype=float)))
                        if xs_var < 1e-6:
                            return False
                    else:
                        return False
            return True
        except Exception:
            return True

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
            
            # Proceed with inference regardless of capture shape per D5 requirement
            
            # Extract features
            features = self.extract_vna_features(vna_df)
            if features is None:
                console.print("Failed to extract features", style="red")
                return
                
            console.print(f"Extracted {len(features)} features", style="green")
            
            # Check if we have the expected number of features
            if len(features) != self.expected_raw_features:
                console.print(
                    f"ERROR: Expected {self.expected_raw_features:,} features but got {len(features):,}",
                    style="red"
                )
                console.print("This will cause a shape mismatch with the model", style="red")
                return
            
            # Run inference
            prediction = self.run_inference(features)
            
            # Optionally read Arduino temperature
            measured_temp = self.read_arduino_temp(timeout=5.0)
            
            # Display results
            self.display_vna_results(file_path.name, prediction, measured_temp)
            
            # Save results
            self.save_vna_result(file_path.name, prediction, measured_temp)
            
            # Update counters
            self.last_prediction = prediction
            self.prediction_count += 1
            
        except Exception as e:
            console.print(f"Error processing VNA file: {e}", style="red")

    def display_vna_results(self, filename, prediction, measured_temp=None):
        """Display inference results."""
        # Create results panel
        extra = ""
        # Fallback to latest observed Arduino temp if explicit read returned None
        if measured_temp is None and getattr(self, 'latest_arduino_temp', None) is not None:
            measured_temp = float(self.latest_arduino_temp)
        if measured_temp is not None:
            diff = prediction - measured_temp
            abs_err = abs(diff)
            within_half = abs_err <= 0.5
            try:
                acc_pct = max(0.0, 100.0 * (1.0 - (abs_err / max(1e-6, abs(measured_temp)))))
            except Exception:
                acc_pct = 0.0
            extra = (
                f"\nReference (Arduino): {measured_temp:.2f}°C"
                f"\nError (pred - ref): {diff:+.2f}°C"
                f"\nAbs Error: {abs_err:.2f}°C (≤0.5°C: {within_half})"
                f"\nAccuracy: {acc_pct:.2f}%"
            )
        # Show trained model params if available
        model_desc = "Hold5 CustomResNet"
        try:
            if self.model_params is not None:
                nb = self.model_params.get('num_blocks', None)
                hd = self.model_params.get('hidden_dim', None)
                if nb is not None and hd is not None:
                    model_desc = f"CustomResNet ({nb} residual blocks, {hd} hidden dim)"
        except Exception:
            pass
        results_text = f"""
File: {filename}
Temperature Prediction: {prediction:.2f}°C
Model: {model_desc}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
{extra}
        """
        
        panel = Panel(
            results_text,
            title="[bold green]Hold5 Model Inference Results[/bold green]",
            border_style="green"
        )
        console.print(panel)

    def save_vna_result(self, filename, prediction, measured_temp=None):
        """Save inference results to file."""
        results_file = Path("vna_inference_results.txt")
        
        with open(results_file, "a") as f:
            if measured_temp is None:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {filename} | pred={prediction:.2f}°C\n")
            else:
                diff = prediction - measured_temp
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {filename} | pred={prediction:.2f}°C | ref={measured_temp:.2f}°C | err={diff:+.2f}°C\n")
        
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
            self.stop_arduino_reader()
            self.live_monitoring = False
            console.print("Live monitoring stopped", style="green")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Live VNA Monitoring with Hold5 Model")
    parser.add_argument("--points", type=int, default=10001, help="Hardware sweep points passed to VNAhl (-Dfsteps). Inference will resample to model's expected length.")
    parser.add_argument("--read-arduino", action="store_true", default=True, help="Read Arduino temperature over serial for accuracy comparison (default: on)")
    parser.add_argument("--arduino-port", type=str, default="/dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_Due_Prog._Port_24336303633351406111-if00", help="Arduino serial device path")
    parser.add_argument("--arduino-baud", type=int, default=115200, help="Arduino serial baud rate")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing hold5 artifacts (scaler, var_threshold, kbest, model)")
    args = parser.parse_args()
    
    inference_engine = LiveVnaInference(
        hw_points=args.points,
        read_arduino=args.read_arduino,
        arduino_port=args.arduino_port,
        arduino_baud=args.arduino_baud,
        model_dir=args.model_dir,
    )
    
    # Start live monitoring
    inference_engine.start_live_monitoring()

if __name__ == "__main__":
    main()

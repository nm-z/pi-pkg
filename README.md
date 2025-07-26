# Raspberry Pi 5 Temperature Prediction Deployment Package

This package contains everything needed to deploy the VNA-based temperature prediction model on a Raspberry Pi 5 using TinyML.

## 📦 Package Contents

```
PI_PKG/
├── README.md                    # This file
├── models/                      # Trained model artifacts
│   ├── hold3_final_model.pkl    # ExtraTreesRegressor model (25MB)
│   ├── hold3_scaler.pkl         # RobustScaler preprocessing
│   ├── hold3_var_threshold.pkl  # VarianceThreshold feature selection
│   └── model_metadata.json     # Model performance metrics
├── build_system/                # Firmware build tools
│   ├── setup_toolchain.sh      # Install dependencies and toolchain
│   ├── build_firmware.sh       # Build the temperature service
│   ├── Dockerfile              # Optional containerized build
│   └── cross_compile.cmake     # CMake cross-compilation settings
├── model_conversion/            # TensorFlow Lite conversion
│   ├── sklearn_to_tflite.py    # Model conversion script
│   ├── test_model_parity.py    # Validation script (±0.5°C accuracy)
│   ├── quantize_model.py       # INT8 quantization
│   └── model.tflm.cc           # Generated TFLite-Micro C++ code
├── firmware/                    # C++ temperature service
│   ├── src/
│   │   ├── main.cpp            # Main service entry point
│   │   ├── tempsvc.cpp         # Core temperature service
│   │   ├── vna_sensor.cpp      # VNA sensor interface
│   │   ├── feature_extractor.cpp # Feature engineering
│   │   ├── tflite_inference.cpp # TFLite model inference
│   │   └── json_output.cpp     # JSON response formatting
│   ├── include/                # Header files
│   ├── CMakeLists.txt          # Build configuration
│   └── config.h               # Service configuration
├── api/                        # REST API service
│   ├── rest_server.cpp         # HTTP server implementation
│   ├── endpoints.cpp           # API endpoint handlers
│   └── api_config.json        # API configuration
├── logging/                    # Logging system
│   ├── logger.cpp             # Rotating log implementation
│   ├── log_config.json        # Logging configuration
│   └── logrotate.conf         # System log rotation
├── weaviate/                   # Vector database integration
│   ├── docker-compose.yml     # Weaviate container setup
│   ├── ingest_service.py      # Data ingestion script
│   ├── weaviate_config.json   # Database configuration
│   └── schema.json            # Data schema definition
├── dashboard/                  # Web dashboard
│   ├── index.html             # Main dashboard page
│   ├── dashboard.js           # Frontend JavaScript
│   ├── styles.css             # Dashboard styling
│   └── server.py              # Simple Python web server
├── scripts/                    # Deployment and utility scripts
│   ├── install.sh             # Full system installation
│   ├── flash_firmware.sh      # Deploy to Pi 5
│   ├── test_system.sh         # System integration tests
│   └── demo.sh                # Quick demo script
└── docs/                      # Documentation
    ├── INSTALL.md             # Installation guide
    ├── API.md                 # API documentation
    ├── TROUBLESHOOTING.md     # Common issues and solutions
    └── ARCHITECTURE.md        # System architecture overview
```

## 🚀 Quick Start

### 1. System Requirements
- Raspberry Pi 5 (8GB RAM recommended)
- 64GB+ microSD card (Class 10/U3)
- Active cooling (fan-based case)
- 5V/5A power supply
- Ubuntu 22.04 LTS or Raspberry Pi OS

### 2. Installation
```bash
# Clone and install
cd PI_PKG
chmod +x scripts/install.sh
./scripts/install.sh

# Build firmware
cd build_system
./setup_toolchain.sh
./build_firmware.sh
```

### 3. Quick Demo
```bash
# Start the complete system
./scripts/demo.sh

# Access dashboard
http://localhost:8080

# Test API
curl http://localhost:5000/latest
```

## 📊 Performance Targets

- **Inference Time**: ≤50ms per reading
- **Accuracy**: ±0.5°C from desktop model
- **Throughput**: 100+ readings/second
- **Memory Usage**: <2GB RAM
- **Power Consumption**: ~13-15W

## 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Model Runtime** | TensorFlow Lite Micro | ExtraTreesRegressor inference |
| **Service** | C++ with REST API | Temperature readings + JSON output |
| **Database** | Weaviate (Vector DB) | Time-series data storage |
| **Dashboard** | HTML/JS + Python | Real-time visualization |
| **Logging** | Rotating file logs | Debug and monitoring |
| **Build System** | CMake + Ubuntu | Cross-compilation |

## 📈 Model Information

- **Algorithm**: ExtraTreesRegressor (optimized by Optuna)
- **Features**: 42 statistical features from 7 VNA measurements
- **Performance**: R² > 0.95 on holdout data
- **Size**: 25MB (original) → ~6MB (quantized)
- **Preprocessing**: RobustScaler + VarianceThreshold

## 🌐 API Endpoints

```bash
GET /latest          # Latest temperature reading
GET /history?n=100   # Historical readings
GET /health          # Service health check
GET /metrics         # Performance metrics
```

## 📱 Dashboard Features

- Real-time temperature display
- Historical trend graphs
- Model confidence indicators
- System health monitoring
- Performance metrics

## 🐳 Docker Support

```bash
# Start Weaviate database
cd weaviate
docker-compose up -d

# Optional: Containerized build
docker build -t pi5-tempsvc -f build_system/Dockerfile .
```

## 📝 Next Steps

1. **Review** the installation guide in `docs/INSTALL.md`
2. **Customize** the configuration in `firmware/config.h`
3. **Deploy** using the scripts in `scripts/`
4. **Monitor** via the dashboard and logs
5. **Scale** by adding more sensors or processing nodes

## 🆘 Support

- Check `docs/TROUBLESHOOTING.md` for common issues
- Review logs in `/var/log/tempsvc/`
- Use `./scripts/test_system.sh` for diagnostics

---

**Ready for deployment by first week of September 2025** ✅ 
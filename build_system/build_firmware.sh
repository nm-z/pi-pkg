#!/bin/bash
set -e

# Raspberry Pi 5 Temperature Service - Firmware Build Script
# Cross-compiles the complete temperature service for deployment

echo "🔨 Building Raspberry Pi 5 Temperature Service Firmware..."

# Check prerequisites
if [ ! -f "cross_compile.cmake" ]; then
    echo "❌ Cross-compilation toolchain not found. Run ./setup_toolchain.sh first"
    exit 1
fi

# Activate Python environment
if [ -d "venv_pi5" ]; then
    source venv_pi5/bin/activate
    echo "✅ Python environment activated"
else  
    echo "❌ Python environment not found. Run ./setup_toolchain.sh first"
    exit 1
fi

# Build TensorFlow Lite Micro library
echo "🧠 Building TensorFlow Lite Micro for ARM64..."
cd tensorflow_lite_micro

# Create build directory for TFLite Micro
mkdir -p build_arm64
cd build_arm64

# Configure TFLite Micro build
cmake -DCMAKE_TOOLCHAIN_FILE=../../cross_compile.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DTFLITE_ENABLE_MICRO=ON \
      -DTFLITE_ENABLE_NNAPI=OFF \
      -DTFLITE_ENABLE_GPU=OFF \
      ..

# Build TFLite Micro
make -j$(nproc)
cd ../..

# Convert sklearn model to TensorFlow Lite
echo "🔄 Converting sklearn model to TensorFlow Lite..."
cd ../model_conversion
python sklearn_to_tflite.py

# Validate model conversion accuracy
echo "🎯 Validating model conversion accuracy..."
python test_model_parity.py

# Generate quantized model
echo "📉 Generating INT8 quantized model..."
python quantize_model.py

# Move back to build directory
cd ../build_system

# Create build directory for main firmware
echo "🏗️ Building main temperature service..."
mkdir -p build
cd build

# Configure main firmware build
cmake -DCMAKE_TOOLCHAIN_FILE=../cross_compile.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DTFLITE_MICRO_ROOT=../tensorflow_lite_micro \
      ../../firmware

# Build the temperature service
make -j$(nproc)

# Create distribution directory
echo "📦 Packaging firmware for deployment..."
cd ..
mkdir -p dist/tempsvc

# Copy binaries
cp build/tempsvc dist/tempsvc/
cp build/tempsvc_test dist/tempsvc/ 2>/dev/null || true

# Copy configuration files
cp ../firmware/config.h dist/tempsvc/
cp ../api/api_config.json dist/tempsvc/
cp ../logging/log_config.json dist/tempsvc/

# Copy model files
cp -r ../models dist/tempsvc/
cp ../model_conversion/model.tflite dist/tempsvc/models/
cp ../model_conversion/model_quantized.tflite dist/tempsvc/models/

# Copy deployment scripts
cp ../scripts/*.sh dist/tempsvc/

# Create systemd service file
cat > dist/tempsvc/tempsvc.service << 'EOF'
[Unit]
Description=Raspberry Pi 5 Temperature Prediction Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/tempsvc
ExecStart=/opt/tempsvc/tempsvc
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Performance settings
Nice=-5
IOSchedulingClass=1
IOSchedulingPriority=4

# Security settings
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/log/tempsvc /tmp

[Install]
WantedBy=multi-user.target
EOF

# Create installation script
cat > dist/tempsvc/install.sh << 'EOF'
#!/bin/bash
set -e

echo "📥 Installing Temperature Service on Raspberry Pi 5..."

# Create service directories
sudo mkdir -p /opt/tempsvc
sudo mkdir -p /var/log/tempsvc

# Copy service files
sudo cp -r * /opt/tempsvc/
sudo chmod +x /opt/tempsvc/tempsvc
sudo chmod +x /opt/tempsvc/*.sh

# Install systemd service
sudo cp tempsvc.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tempsvc

# Create log rotation
sudo cp logrotate.conf /etc/logrotate.d/tempsvc

# Set permissions
sudo chown -R pi:pi /opt/tempsvc
sudo chown -R pi:pi /var/log/tempsvc

echo "✅ Installation complete!"
echo "🚀 Start service: sudo systemctl start tempsvc"
echo "📊 Check status: sudo systemctl status tempsvc"
echo "📝 View logs: sudo journalctl -u tempsvc -f"
EOF

chmod +x dist/tempsvc/install.sh

# Create deployment archive
echo "📦 Creating deployment archive..."
cd dist
tar -czf tempsvc_pi5_$(date +%Y%m%d_%H%M%S).tar.gz tempsvc/
cd ..

# Build summary
echo ""
echo "✅ Build complete!"
echo ""
echo "📊 Build Summary:"
echo "   Target: ARM64 (Raspberry Pi 5)"
echo "   Compiler: $(aarch64-linux-gnu-gcc --version | head -n1)"
echo "   Build type: Release"
echo "   TensorFlow Lite: Enabled"
echo "   Model quantization: INT8"
echo ""
echo "📁 Output files:"
echo "   Binary: dist/tempsvc/tempsvc"
echo "   Archive: dist/tempsvc_pi5_*.tar.gz"
echo "   Installer: dist/tempsvc/install.sh"
echo ""
echo "🚀 Next steps:"
echo "1. Copy dist/tempsvc_pi5_*.tar.gz to your Raspberry Pi 5"
echo "2. Extract and run ./install.sh on the Pi"
echo "3. Start service: sudo systemctl start tempsvc"
echo ""
echo "🔧 Cross-compilation complete for Raspberry Pi 5 deployment!"

# Performance estimates
echo ""
echo "📈 Expected Performance on Pi 5:"
echo "   Inference time: ~15-25ms per reading"
echo "   Memory usage: ~50-100MB"
echo "   Power consumption: ~13-15W"
echo "   Throughput: 100+ readings/second" 
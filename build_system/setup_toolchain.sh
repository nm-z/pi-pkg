#!/bin/bash
set -e

# Raspberry Pi 5 Temperature Service - Toolchain Setup
# Ubuntu SBC cross-compilation environment setup

echo "🚀 Setting up Raspberry Pi 5 TinyML toolchain..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install basic build tools
echo "📦 Installing build essentials..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    libjson-c-dev \
    libmicrohttpd-dev

# Install cross-compilation tools for ARM64
echo "🔧 Installing ARM64 cross-compilation tools..."
sudo apt install -y \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross

# Install TensorFlow Lite dependencies
echo "🧠 Installing TensorFlow Lite build dependencies..."
sudo apt install -y \
    flatbuffers-compiler \
    libflatbuffers-dev \
    libeigen3-dev \
    libgemmlowp-dev

# Create Python virtual environment for model conversion
echo "🐍 Setting up Python environment for model conversion..."
python3 -m venv venv_pi5
source venv_pi5/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install \
    tensorflow==2.15.0 \
    scikit-learn==1.3.0 \
    numpy==1.24.3 \
    joblib==1.3.2 \
    sklearn2tflite \
    tflite-runtime

# Download TensorFlow Lite Micro source
echo "📥 Downloading TensorFlow Lite Micro..."
if [ ! -d "tensorflow_lite_micro" ]; then
    git clone https://github.com/tensorflow/tflite-micro.git tensorflow_lite_micro
fi

# Create CMake toolchain file for cross-compilation
echo "⚙️ Creating CMake toolchain file..."
cat > cross_compile.cmake << 'EOF'
# CMake toolchain for Raspberry Pi 5 cross-compilation
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler paths
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Target environment paths
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Compiler flags for Raspberry Pi 5 (Cortex-A76)
set(CMAKE_C_FLAGS "-mcpu=cortex-a76 -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "-mcpu=cortex-a76 -O3 -DNDEBUG -std=c++17")
EOF

# Install Docker (optional containerized build)
echo "🐳 Installing Docker for containerized builds..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Create build directories
mkdir -p {build,dist,tmp}

# Set environment variables
echo "📝 Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# Raspberry Pi 5 TinyML Development Environment
export PI5_TOOLCHAIN_ROOT="$(pwd)"
export PI5_CROSS_COMPILE=aarch64-linux-gnu-
export PATH="$PATH:$PI5_TOOLCHAIN_ROOT/venv_pi5/bin"
export TFLITE_MICRO_ROOT="$PI5_TOOLCHAIN_ROOT/tensorflow_lite_micro"
EOF

echo "✅ Toolchain setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Source the environment: source ~/.bashrc"
echo "2. Activate Python venv: source venv_pi5/bin/activate"
echo "3. Run ./build_firmware.sh to compile the temperature service"
echo ""
echo "🔧 Toolchain info:"
echo "   Cross-compiler: $(aarch64-linux-gnu-gcc --version | head -n1)"
echo "   CMake: $(cmake --version | head -n1)"
echo "   Python: $(python3 --version)"
echo "   TensorFlow: $(python3 -c 'import tensorflow; print(f\"TensorFlow {tensorflow.__version__}\")')" 
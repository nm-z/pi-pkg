#!/bin/bash
set -e

# Raspberry Pi 5 Temperature Service - Complete Installation Script
# ================================================================
# 
# This script installs and configures the complete temperature prediction
# system on Raspberry Pi 5, including all dependencies and services.

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Installation configuration
INSTALL_DIR="/opt/tempsvc"
SERVICE_USER="pi"
LOG_DIR="/var/log/tempsvc"
DATA_DIR="/opt/weaviate/data"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Raspberry Pi 5 Temperature Service Installer        ║"
echo "║              TinyML Edge Deployment v1.0                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running on Raspberry Pi 5
check_hardware() {
    echo -e "${CYAN}🔍 Checking hardware compatibility...${NC}"
    
    # Check ARM64 architecture
    if [[ $(uname -m) != "aarch64" ]]; then
        echo -e "${RED}❌ Error: This installer requires ARM64 architecture${NC}"
        exit 1
    fi
    
    # Check for Raspberry Pi
    if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
        echo -e "${YELLOW}⚠️ Warning: Not detected as Raspberry Pi hardware${NC}"
        echo -e "${YELLOW}   Continuing installation anyway...${NC}"
    fi
    
    # Check available memory
    TOTAL_RAM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [[ $TOTAL_RAM -lt 4000 ]]; then
        echo -e "${YELLOW}⚠️ Warning: Recommended minimum 4GB RAM, detected ${TOTAL_RAM}MB${NC}"
    fi
    
    echo -e "${GREEN}✅ Hardware check completed${NC}"
}

# Update system packages
update_system() {
    echo -e "${CYAN}📦 Updating system packages...${NC}"
    
    sudo apt update -y
    sudo apt upgrade -y
    
    echo -e "${GREEN}✅ System updated${NC}"
}

# Install system dependencies
install_system_dependencies() {
    echo -e "${CYAN}🔧 Installing system dependencies...${NC}"
    
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip \
        python3 \
        python3-pip \
        python3-venv \
        pkg-config \
        libssl-dev \
        libcurl4-openssl-dev \
        libjson-c-dev \
        libmicrohttpd-dev \
        flatbuffers-compiler \
        libflatbuffers-dev \
        libeigen3-dev \
        systemd \
        logrotate \
        htop \
        iotop \
        tree
    
    echo -e "${GREEN}✅ System dependencies installed${NC}"
}

# Install Docker and Docker Compose
install_docker() {
    echo -e "${CYAN}🐳 Installing Docker and Docker Compose...${NC}"
    
    # Check if Docker is already installed
    if command -v docker &> /dev/null; then
        echo -e "${YELLOW}📋 Docker already installed, skipping...${NC}"
    else
        # Install Docker
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        rm get-docker.sh
        
        # Add user to docker group
        sudo usermod -aG docker $SERVICE_USER
        
        # Enable Docker service
        sudo systemctl enable docker
        sudo systemctl start docker
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        sudo pip3 install docker-compose
    fi
    
    echo -e "${GREEN}✅ Docker and Docker Compose installed${NC}"
}

# Create service directories
create_directories() {
    echo -e "${CYAN}📁 Creating service directories...${NC}"
    
    # Create main installation directory
    sudo mkdir -p $INSTALL_DIR
    sudo chown $SERVICE_USER:$SERVICE_USER $INSTALL_DIR
    
    # Create log directory
    sudo mkdir -p $LOG_DIR
    sudo chown $SERVICE_USER:$SERVICE_USER $LOG_DIR
    
    # Create Weaviate data directory
    sudo mkdir -p $DATA_DIR
    sudo chown $SERVICE_USER:$SERVICE_USER $DATA_DIR
    
    # Create run directory
    sudo mkdir -p /var/run/tempsvc
    sudo chown $SERVICE_USER:$SERVICE_USER /var/run/tempsvc
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Install Python dependencies
install_python_dependencies() {
    echo -e "${CYAN}🐍 Installing Python dependencies...${NC}"
    
    # Create Python virtual environment
    python3 -m venv $INSTALL_DIR/venv
    source $INSTALL_DIR/venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install required packages
    pip install \
        tensorflow==2.15.0 \
        tflite-runtime \
        numpy==1.24.3 \
        requests \
        flask \
        waitress \
        weaviate-client \
        schedule \
        psutil
    
    deactivate
    
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
}

# Copy service files
copy_service_files() {
    echo -e "${CYAN}📋 Copying service files...${NC}"
    
    # Copy all service files to installation directory
    sudo cp -r ../models $INSTALL_DIR/
    sudo cp -r ../firmware $INSTALL_DIR/
    sudo cp -r ../api $INSTALL_DIR/
    sudo cp -r ../logging $INSTALL_DIR/
    sudo cp -r ../weaviate $INSTALL_DIR/
    sudo cp -r ../dashboard $INSTALL_DIR/
    sudo cp -r ../scripts $INSTALL_DIR/
    
    # Set proper permissions
    sudo chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_DIR
    sudo chmod +x $INSTALL_DIR/scripts/*.sh
    
    # Copy configuration files
    sudo cp ../firmware/config.h $INSTALL_DIR/
    
    echo -e "${GREEN}✅ Service files copied${NC}"
}

# Build the temperature service
build_service() {
    echo -e "${CYAN}🔨 Building temperature service...${NC}"
    
    cd $INSTALL_DIR/firmware
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    
    # Build service
    make -j$(nproc)
    
    # Make binary executable
    chmod +x tempsvc
    
    echo -e "${GREEN}✅ Temperature service built${NC}"
}

# Install systemd service
install_systemd_service() {
    echo -e "${CYAN}⚙️ Installing systemd service...${NC}"
    
    # Create systemd service file
    cat > /tmp/tempsvc.service << EOF
[Unit]
Description=Raspberry Pi 5 Temperature Prediction Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/firmware/build/tempsvc
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
ReadWritePaths=$LOG_DIR /var/run/tempsvc /tmp

# Environment
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

    # Install service file
    sudo mv /tmp/tempsvc.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable tempsvc
    
    echo -e "${GREEN}✅ Systemd service installed${NC}"
}

# Setup log rotation
setup_log_rotation() {
    echo -e "${CYAN}📝 Setting up log rotation...${NC}"
    
    # Create logrotate configuration
    cat > /tmp/tempsvc << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload tempsvc 2>/dev/null || true
    endscript
}

$LOG_DIR/*.csv {
    weekly
    missingok
    rotate 4
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
}
EOF

    sudo mv /tmp/tempsvc /etc/logrotate.d/
    
    echo -e "${GREEN}✅ Log rotation configured${NC}"
}

# Setup Weaviate database
setup_weaviate() {
    echo -e "${CYAN}🗄️ Setting up Weaviate vector database...${NC}"
    
    cd $INSTALL_DIR/weaviate
    
    # Start Weaviate services
    docker-compose up -d weaviate
    
    # Wait for Weaviate to be ready
    echo -e "${YELLOW}⏳ Waiting for Weaviate to start...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; then
            break
        fi
        sleep 2
    done
    
    # Initialize schema
    python3 ../scripts/init_weaviate_schema.py
    
    echo -e "${GREEN}✅ Weaviate database configured${NC}"
}

# Setup dashboard
setup_dashboard() {
    echo -e "${CYAN}🖥️ Setting up web dashboard...${NC}"
    
    cd $INSTALL_DIR/dashboard
    
    # Create dashboard service
    cat > /tmp/dashboard.service << EOF
[Unit]
Description=Temperature Service Web Dashboard
After=tempsvc.service
Requires=tempsvc.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR/dashboard
ExecStart=$INSTALL_DIR/venv/bin/python server.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/dashboard.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable dashboard
    
    echo -e "${GREEN}✅ Dashboard service configured${NC}"
}

# Configure firewall
configure_firewall() {
    echo -e "${CYAN}🔥 Configuring firewall...${NC}"
    
    # Install UFW if not present
    if ! command -v ufw &> /dev/null; then
        sudo apt install -y ufw
    fi
    
    # Configure firewall rules
    sudo ufw --force enable
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH
    sudo ufw allow ssh
    
    # Allow service ports
    sudo ufw allow 5000  # Temperature service API
    sudo ufw allow 8080  # Weaviate
    sudo ufw allow 8081  # Dashboard
    sudo ufw allow 3000  # Grafana (optional)
    sudo ufw allow 9090  # Prometheus (optional)
    
    echo -e "${GREEN}✅ Firewall configured${NC}"
}

# Performance optimization
optimize_performance() {
    echo -e "${CYAN}⚡ Applying performance optimizations...${NC}"
    
    # CPU governor settings for performance
    echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
    
    # Memory optimization
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
    
    # Enable memory cgroup
    if ! grep -q "cgroup_memory=1" /boot/cmdline.txt; then
        sudo sed -i 's/$/ cgroup_memory=1 cgroup_enable=memory/' /boot/cmdline.txt
    fi
    
    echo -e "${GREEN}✅ Performance optimizations applied${NC}"
}

# Create helper scripts
create_helper_scripts() {
    echo -e "${CYAN}📋 Creating helper scripts...${NC}"
    
    # Create status check script
    cat > $INSTALL_DIR/check_status.sh << 'EOF'
#!/bin/bash
echo "=== Temperature Service Status ==="
systemctl status tempsvc --no-pager -l
echo
echo "=== Recent Logs ==="
journalctl -u tempsvc -n 10 --no-pager
echo
echo "=== Performance ==="
curl -s http://localhost:5000/metrics | head -10
EOF
    
    # Create restart script
    cat > $INSTALL_DIR/restart_services.sh << 'EOF'
#!/bin/bash
echo "Restarting all services..."
sudo systemctl restart tempsvc
sudo systemctl restart dashboard
cd /opt/tempsvc/weaviate && docker-compose restart
echo "Services restarted!"
EOF
    
    # Make scripts executable
    chmod +x $INSTALL_DIR/*.sh
    
    echo -e "${GREEN}✅ Helper scripts created${NC}"
}

# Final system check
final_system_check() {
    echo -e "${CYAN}🔍 Running final system check...${NC}"
    
    # Check service status
    if systemctl is-enabled tempsvc &>/dev/null; then
        echo -e "${GREEN}✅ Temperature service enabled${NC}"
    else
        echo -e "${RED}❌ Temperature service not enabled${NC}"
    fi
    
    # Check Docker status
    if systemctl is-active docker &>/dev/null; then
        echo -e "${GREEN}✅ Docker is running${NC}"
    else  
        echo -e "${RED}❌ Docker is not running${NC}"
    fi
    
    # Check available disk space
    DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $DISK_USAGE -lt 80 ]]; then
        echo -e "${GREEN}✅ Sufficient disk space (${DISK_USAGE}% used)${NC}"
    else
        echo -e "${YELLOW}⚠️ Warning: High disk usage (${DISK_USAGE}% used)${NC}"
    fi
    
    echo -e "${GREEN}✅ System check completed${NC}"
}

# Display completion message
display_completion() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          🎉 Installation Completed Successfully! 🎉      ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${CYAN}📋 Next Steps:${NC}"
    echo -e "${YELLOW}1. Reboot the system to apply all changes:${NC}"
    echo -e "   ${GREEN}sudo reboot${NC}"
    echo
    echo -e "${YELLOW}2. After reboot, start the services:${NC}"
    echo -e "   ${GREEN}sudo systemctl start tempsvc${NC}"
    echo -e "   ${GREEN}sudo systemctl start dashboard${NC}"
    echo
    echo -e "${YELLOW}3. Check service status:${NC}"
    echo -e "   ${GREEN}$INSTALL_DIR/check_status.sh${NC}"
    echo
    echo -e "${YELLOW}4. Access the dashboard:${NC}"
    echo -e "   ${GREEN}http://$(hostname -I | awk '{print $1}'):8081${NC}"
    echo
    echo -e "${YELLOW}5. API endpoints:${NC}"
    echo -e "   ${GREEN}curl http://localhost:5000/latest${NC}"
    echo -e "   ${GREEN}curl http://localhost:5000/health${NC}"
    echo
    echo -e "${CYAN}📚 Documentation: $INSTALL_DIR/docs/${NC}"
    echo -e "${CYAN}📝 Logs: $LOG_DIR/${NC}"
    echo -e "${CYAN}⚙️ Config: $INSTALL_DIR/config.h${NC}"
}

# Main installation workflow
main() {
    echo -e "${BLUE}🚀 Starting installation process...${NC}"
    
    # Check for root privileges for certain operations
    if [[ $EUID -eq 0 ]]; then
        echo -e "${RED}❌ Please run this script as a regular user, not root${NC}"
        echo -e "${YELLOW}   The script will use sudo when needed${NC}"
        exit 1
    fi
    
    # Run installation steps
    check_hardware
    update_system
    install_system_dependencies
    install_docker
    create_directories
    install_python_dependencies
    copy_service_files
    build_service
    install_systemd_service
    setup_log_rotation
    setup_weaviate
    setup_dashboard
    configure_firewall
    optimize_performance
    create_helper_scripts
    final_system_check
    display_completion
    
    echo -e "${GREEN}✅ Installation completed successfully!${NC}"
}

# Run main installation
main "$@" 
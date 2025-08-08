#!/bin/bash

# Raspberry Pi 5 Temperature Service - Quick Demo Script
# =====================================================
# 
# Demonstrates the complete temperature sensor-to-dashboard loop
# Ready for demo by first week of September 2025

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
INSTALL_DIR="/opt/tempsvc"
API_PORT=5000
DASHBOARD_PORT=8081
WEAVIATE_PORT=8080

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║    🎯 Raspberry Pi 5 Temperature Service Demo 🎯        ║"
echo "║         Temperature Sensor → Dashboard Loop             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if services are installed
check_installation() {
    echo -e "${CYAN}🔍 Checking installation...${NC}"
    
    if [ ! -d "$INSTALL_DIR" ]; then
        echo -e "${RED}❌ Service not installed. Run install.sh first.${NC}"
        exit 1
    fi
    
    if [ ! -f "$INSTALL_DIR/firmware/build/tempsvc" ]; then
        echo -e "${RED}❌ Temperature service binary not found. Run build process.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Installation verified${NC}"
}

# Start Weaviate database
start_weaviate() {
    echo -e "${CYAN}🗄️ Starting Weaviate vector database...${NC}"
    
    cd $INSTALL_DIR/weaviate
    
    # Start Weaviate in background
    docker-compose up -d weaviate
    
    # Wait for Weaviate to be ready
    echo -e "${YELLOW}⏳ Waiting for Weaviate to start...${NC}"
    
    for i in {1..30}; do
        if curl -s http://localhost:$WEAVIATE_PORT/v1/.well-known/ready > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Weaviate is ready${NC}"
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ Weaviate failed to start${NC}"
            exit 1
        fi
        
        sleep 2
        echo -n "."
    done
}

# Start temperature service
start_temperature_service() {
    echo -e "${CYAN}🌡️ Starting temperature prediction service...${NC}"
    
    cd $INSTALL_DIR
    
    # Kill any existing service
    pkill -f tempsvc || true
    
    # Start service in background
    ./firmware/build/tempsvc &
    SERVICE_PID=$!
    
    # Wait for service to be ready
    echo -e "${YELLOW}⏳ Waiting for temperature service...${NC}"
    
    for i in {1..20}; do
        if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Temperature service is ready (PID: $SERVICE_PID)${NC}"
            break
        fi
        
        if [ $i -eq 20 ]; then
            echo -e "${RED}❌ Temperature service failed to start${NC}"
            kill $SERVICE_PID 2>/dev/null || true
            exit 1
        fi
        
        sleep 1
        echo -n "."
    done
}

# Start dashboard
start_dashboard() {
    echo -e "${CYAN}🖥️ Starting web dashboard...${NC}"
    
    cd $INSTALL_DIR/dashboard
    
    # Kill any existing dashboard
    pkill -f "dashboard.*server.py" || true
    
    # Activate Python environment and start dashboard
    source $INSTALL_DIR/venv/bin/activate
    python server.py &
    DASHBOARD_PID=$!
    deactivate
    
    # Wait for dashboard to be ready
    echo -e "${YELLOW}⏳ Waiting for dashboard...${NC}"
    
    for i in {1..15}; do
        if curl -s http://localhost:$DASHBOARD_PORT > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Dashboard is ready (PID: $DASHBOARD_PID)${NC}"
            break
        fi
        
        if [ $i -eq 15 ]; then
            echo -e "${RED}❌ Dashboard failed to start${NC}"
            kill $DASHBOARD_PID 2>/dev/null || true
            exit 1
        fi
        
        sleep 1
        echo -n "."
    done
}

# Demonstrate the sensor-to-dashboard loop
demonstrate_system() {
    echo -e "${CYAN}🔄 Demonstrating Temperature Sensor → Dashboard Loop${NC}"
    echo
    
    # Get system information
    echo -e "${YELLOW}📊 System Information:${NC}"
    echo "   Host: $(hostname)"
    echo "   IP: $(hostname -I | awk '{print $1}')"
    echo "   Architecture: $(uname -m)"
    echo "   OS: $(lsb_release -d | cut -f2)"
    echo "   Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo
    
    # Test API endpoints
    echo -e "${YELLOW}🔌 Testing API Endpoints:${NC}"
    
    # Health check
    if curl -s http://localhost:$API_PORT/health | jq . > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Health endpoint: http://localhost:$API_PORT/health${NC}"
    else
        echo -e "${RED}   ❌ Health endpoint failed${NC}"
    fi
    
    # Latest reading
    if curl -s http://localhost:$API_PORT/latest | jq . > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Latest reading: http://localhost:$API_PORT/latest${NC}"
    else
        echo -e "${RED}   ❌ Latest reading failed${NC}"
    fi
    
    # Metrics
    if curl -s http://localhost:$API_PORT/metrics > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Metrics endpoint: http://localhost:$API_PORT/metrics${NC}"
    else
        echo -e "${RED}   ❌ Metrics endpoint failed${NC}"
    fi
    
    echo
    
    # Show live data
    echo -e "${YELLOW}🌡️ Live Temperature Readings (5 samples):${NC}"
    for i in {1..5}; do
        echo -n "   Sample $i: "
        
        RESPONSE=$(curl -s http://localhost:$API_PORT/latest 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$RESPONSE" ]; then
            TEMP=$(echo $RESPONSE | jq -r '.temp_c' 2>/dev/null)
            CONFIDENCE=$(echo $RESPONSE | jq -r '.confidence' 2>/dev/null)
            TIME=$(echo $RESPONSE | jq -r '.inference_time_ms' 2>/dev/null)
            
            if [ "$TEMP" != "null" ] && [ "$TEMP" != "" ]; then
                echo -e "${GREEN}${TEMP}°C (${CONFIDENCE}, ${TIME}ms)${NC}"
            else
                echo -e "${RED}Invalid response${NC}"
            fi
        else
            echo -e "${RED}No response${NC}"
        fi
        
        sleep 2
    done
    
    echo
    
    # Test Weaviate connection
    echo -e "${YELLOW}🗄️ Testing Weaviate Database:${NC}"
    if curl -s http://localhost:$WEAVIATE_PORT/v1/meta > /dev/null 2>&1; then
        echo -e "${GREEN}   ✅ Weaviate is accessible${NC}"
    else
        echo -e "${RED}   ❌ Weaviate connection failed${NC}"
    fi
    
    echo
}

# Display access information
display_access_info() {
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"  
    echo "║               🎉 Demo System Ready! 🎉                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${CYAN}🌐 Access Points:${NC}"
    echo
    echo -e "${YELLOW}📊 Web Dashboard:${NC}"
    echo -e "   ${GREEN}http://localhost:$DASHBOARD_PORT${NC}"
    echo -e "   ${GREEN}http://$LOCAL_IP:$DASHBOARD_PORT${NC}"
    echo
    echo -e "${YELLOW}🔌 API Endpoints:${NC}"
    echo -e "   ${GREEN}curl http://localhost:$API_PORT/latest${NC}"
    echo -e "   ${GREEN}curl http://localhost:$API_PORT/health${NC}"
    echo -e "   ${GREEN}curl http://localhost:$API_PORT/metrics${NC}"
    echo -e "   ${GREEN}curl http://localhost:$API_PORT/history?n=10${NC}"
    echo
    echo -e "${YELLOW}🗄️ Weaviate Database:${NC}"
    echo -e "   ${GREEN}http://localhost:$WEAVIATE_PORT/v1/meta${NC}"
    echo
    echo -e "${CYAN}📈 Performance Monitoring:${NC}"
    echo -e "   Temperature readings: Every 100ms (10Hz)"
    echo -e "   Inference time target: ≤50ms per reading"
    echo -e "   Model accuracy: ±0.5°C from desktop model"
    echo
    echo -e "${CYAN}🎯 Demo Features:${NC}"
    echo -e "   ✅ Real-time temperature prediction"
    echo -e "   ✅ TensorFlow Lite model inference"
    echo -e "   ✅ REST API endpoints"
    echo -e "   ✅ Vector database storage (Weaviate)"
    echo -e "   ✅ Live web dashboard"
    echo -e "   ✅ Performance monitoring"
    echo
}

# Interactive demo menu
interactive_demo() {
    while true; do
        echo -e "${CYAN}🎮 Interactive Demo Menu:${NC}"
        echo "   1) Show latest temperature reading"
        echo "   2) Show system metrics"
        echo "   3) Test API response time"
        echo "   4) Show recent history"
        echo "   5) Check service health"
        echo "   0) Exit demo"
        echo
        read -p "Select option (0-5): " choice
        
        case $choice in
            1)
                echo -e "${YELLOW}🌡️ Latest Temperature Reading:${NC}"
                curl -s http://localhost:$API_PORT/latest | jq .
                echo
                ;;
            2)
                echo -e "${YELLOW}📊 System Metrics:${NC}"
                curl -s http://localhost:$API_PORT/metrics
                echo
                ;;
            3)
                echo -e "${YELLOW}⚡ API Response Time Test:${NC}"
                for i in {1..3}; do
                    echo -n "   Test $i: "
                    START_TIME=$(date +%s%3N)
                    curl -s http://localhost:$API_PORT/latest > /dev/null
                    END_TIME=$(date +%s%3N)
                    RESPONSE_TIME=$((END_TIME - START_TIME))
                    echo "${RESPONSE_TIME}ms"
                done
                echo
                ;;
            4)
                echo -e "${YELLOW}📈 Recent History (last 5 readings):${NC}"
                curl -s "http://localhost:$API_PORT/history?n=5" | jq .
                echo
                ;;
            5)
                echo -e "${YELLOW}❤️ Service Health Check:${NC}"
                curl -s http://localhost:$API_PORT/health | jq .
                echo
                ;;
            0)
                echo -e "${GREEN}👋 Exiting demo...${NC}"
                break
                ;;
            *)
                echo -e "${RED}❌ Invalid option. Please select 0-5.${NC}"
                ;;
        esac
        
        echo "Press Enter to continue..."
        read
        echo
    done
}

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up demo processes...${NC}"
    
    # Stop services gracefully
    if [ ! -z "$SERVICE_PID" ]; then
        kill $SERVICE_PID 2>/dev/null && echo "   Temperature service stopped"
    fi
    
    if [ ! -z "$DASHBOARD_PID" ]; then
        kill $DASHBOARD_PID 2>/dev/null && echo "   Dashboard stopped"
    fi
    
    # Stop docker services
    cd $INSTALL_DIR/weaviate 2>/dev/null && docker-compose down 2>/dev/null && echo "   Weaviate stopped"
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Set up signal handlers for cleanup
trap cleanup EXIT INT TERM

# Main demo workflow
main() {
    echo -e "${BLUE}🚀 Starting demo system...${NC}"
    echo
    
    # Run demo steps
    check_installation
    start_weaviate
    start_temperature_service
    start_dashboard
    
    echo
    demonstrate_system
    display_access_info
    
    # Ask if user wants interactive demo
    echo -e "${CYAN}Would you like to try the interactive demo? (y/n): ${NC}"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        interactive_demo
    else
        echo -e "${YELLOW}Demo system is running. Press Ctrl+C to stop.${NC}"
        
        # Keep demo running until interrupted
        while true; do
            sleep 1
        done
    fi
}

# Run main demo
main "$@" 
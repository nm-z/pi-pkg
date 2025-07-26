/**
 * Raspberry Pi 5 Temperature Prediction Service - Main Entry Point
 * ================================================================
 * 
 * High-performance C++ service that reads VNA sensor data, extracts features,
 * runs TensorFlow Lite inference, and outputs JSON temperature predictions.
 * 
 * Target performance: ≤50ms per reading
 * Output format: {"timestamp", "temp_c", "confidence", "inference_time_ms"}
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <csignal>

#include "../include/tempsvc.h"
#include "../include/logger.h"
#include "../config.h"

// Global flag for graceful shutdown
volatile sig_atomic_t g_shutdown_requested = 0;

/**
 * Signal handler for graceful shutdown
 */
void signal_handler(int signal) {
    std::cout << "\n🛑 Shutdown signal received (" << signal << "), stopping service..." << std::endl;
    g_shutdown_requested = 1;
}

/**
 * Setup signal handlers for graceful shutdown
 */
void setup_signal_handlers() {
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // systemctl stop
    signal(SIGHUP, signal_handler);   // systemctl restart
}

/**
 * Create necessary directories for logging and operation
 */
bool create_directories() {
    const char* log_dir = "/var/log/tempsvc";
    const char* run_dir = "/var/run/tempsvc";
    
    // Create log directory
    if (mkdir(log_dir, 0755) != 0 && errno != EEXIST) {
        std::cerr << "❌ Failed to create log directory: " << log_dir << std::endl;
        return false;
    }
    
    // Create run directory for PID file
    if (mkdir(run_dir, 0755) != 0 && errno != EEXIST) {
        std::cerr << "❌ Failed to create run directory: " << run_dir << std::endl;
        return false;
    }
    
    return true;
}

/**
 * Write PID file for service management
 */
bool write_pid_file() {
    std::ofstream pid_file("/var/run/tempsvc/tempsvc.pid");
    if (!pid_file.is_open()) {
        std::cerr << "❌ Failed to write PID file" << std::endl;
        return false;
    }
    
    pid_file << getpid() << std::endl;
    pid_file.close();
    
    return true;
}

/**
 * Remove PID file on shutdown
 */
void cleanup_pid_file() {
    unlink("/var/run/tempsvc/tempsvc.pid");
}

/**
 * Display service banner and startup information
 */
void display_banner() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Raspberry Pi 5 Temperature Prediction           ║\n";
    std::cout << "║              TinyML Edge Service v1.0                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "🎯 Target performance: ≤50ms per reading\n";
    std::cout << "🧠 Model: ExtraTreesRegressor (TensorFlow Lite)\n";
    std::cout << "📊 Features: 42 VNA statistical features\n";
    std::cout << "📡 Output: JSON via REST API (port " << Config::REST_API_PORT << ")\n";
    std::cout << "📝 Logs: /var/log/tempsvc/\n";
    std::cout << "\n";
}

/**
 * Main service loop - core temperature prediction processing
 */
int run_service_loop() {
    // Initialize temperature service
    TemperatureService temp_service;
    
    if (!temp_service.initialize()) {
        Logger::error("Failed to initialize temperature service");
        return EXIT_FAILURE;
    }
    
    Logger::info("Temperature service initialized successfully");
    Logger::info("Starting main prediction loop...");
    
    // Performance tracking
    auto last_stats_time = std::chrono::steady_clock::now();
    int readings_count = 0;
    double total_inference_time = 0.0;
    
    // Main service loop
    while (!g_shutdown_requested) {
        auto loop_start = std::chrono::high_resolution_clock::now();
        
        try {
            // Read VNA sensor data and perform prediction
            auto result = temp_service.predict_temperature();
            
            if (result.success) {
                readings_count++;
                total_inference_time += result.inference_time_ms;
                
                // Log successful prediction (debug level)
                Logger::debug("Prediction successful: " + 
                             std::to_string(result.temperature) + "°C, " +
                             "confidence: " + result.confidence + ", " +
                             "time: " + std::to_string(result.inference_time_ms) + "ms");
                
                // Check performance target
                if (result.inference_time_ms > Config::MAX_INFERENCE_TIME_MS) {
                    Logger::warning("Inference time exceeded target: " + 
                                  std::to_string(result.inference_time_ms) + "ms > " +
                                  std::to_string(Config::MAX_INFERENCE_TIME_MS) + "ms");
                }
                
            } else {
                Logger::error("Prediction failed: " + result.error_message);
            }
            
        } catch (const std::exception& e) {
            Logger::error("Exception in service loop: " + std::string(e.what()));
        }
        
        // Performance statistics (every 60 seconds)
        auto now = std::chrono::steady_clock::now();
        auto stats_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time);
        
        if (stats_elapsed.count() >= 60) {
            double avg_inference_time = readings_count > 0 ? total_inference_time / readings_count : 0.0;
            double readings_per_second = readings_count / 60.0;
            
            Logger::info("Performance stats - Readings: " + std::to_string(readings_count) + 
                        ", Avg time: " + std::to_string(avg_inference_time) + "ms" +
                        ", Throughput: " + std::to_string(readings_per_second) + "/sec");
            
            // Reset counters
            readings_count = 0;
            total_inference_time = 0.0;
            last_stats_time = now;
        }
        
        // Control loop timing to achieve target frequency
        auto loop_end = std::chrono::high_resolution_clock::now();
        auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);
        
        // Sleep to maintain target reading frequency (default: 10Hz = 100ms interval)
        auto target_interval = std::chrono::milliseconds(Config::READING_INTERVAL_MS);
        if (loop_duration < target_interval) {
            std::this_thread::sleep_for(target_interval - loop_duration);
        }
    }
    
    Logger::info("Service loop terminated gracefully");
    return EXIT_SUCCESS;
}

/**
 * Main function - service initialization and management
 */
int main(int argc, char* argv[]) {
    try {
        // Display banner
        display_banner();
        
        // Setup signal handlers
        setup_signal_handlers();
        
        // Create necessary directories
        if (!create_directories()) {
            std::cerr << "❌ Failed to create service directories" << std::endl;
            return EXIT_FAILURE;
        }
        
        // Initialize logging system
        if (!Logger::initialize()) {
            std::cerr << "❌ Failed to initialize logging system" << std::endl;
            return EXIT_FAILURE;
        }
        
        Logger::info("=== Raspberry Pi 5 Temperature Service Starting ===");
        Logger::info("Process ID: " + std::to_string(getpid()));
        Logger::info("Build target: ARM64 (Raspberry Pi 5)");
        Logger::info("TensorFlow Lite: Enabled");
        
        // Write PID file
        if (!write_pid_file()) {
            Logger::error("Failed to write PID file");
            return EXIT_FAILURE;
        }
        
        std::cout << "✅ Service initialized successfully\n";
        std::cout << "🚀 Starting temperature prediction service...\n";
        std::cout << "📊 Press Ctrl+C to stop gracefully\n\n";
        
        // Run main service loop
        int result = run_service_loop();
        
        // Cleanup
        cleanup_pid_file();
        Logger::info("=== Temperature Service Shutdown Complete ===");
        
        std::cout << "\n✅ Service stopped gracefully\n";
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        Logger::error("Fatal error: " + std::string(e.what()));
        cleanup_pid_file();
        return EXIT_FAILURE;
    }
}

/**
 * Service Health Check Function
 * Can be called by external monitoring systems
 */
extern "C" int check_service_health() {
    try {
        // Check if PID file exists and process is running
        std::ifstream pid_file("/var/run/tempsvc/tempsvc.pid");
        if (!pid_file.is_open()) {
            return 1; // Service not running
        }
        
        // Additional health checks could be added here:
        // - Check model is loaded
        // - Check sensor connectivity  
        // - Check recent prediction success rate
        
        return 0; // Service healthy
        
    } catch (...) {
        return 2; // Health check failed
    }
} 
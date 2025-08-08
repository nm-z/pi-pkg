/**
 * Raspberry Pi 5 Temperature Service Configuration
 * ===============================================
 * 
 * Central configuration file for all service parameters.
 * Compile-time constants for optimal performance.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace Config {

    // === Performance Settings ===
    
    // Maximum allowed inference time (target: ≤50ms)
    constexpr int MAX_INFERENCE_TIME_MS = 50;
    
    // Reading interval for main service loop (100ms = 10Hz)
    constexpr int READING_INTERVAL_MS = 100;
    
    // Buffer sizes for data processing
    constexpr int VNA_DATA_BUFFER_SIZE = 1024;
    constexpr int FEATURE_BUFFER_SIZE = 42;
    
    // === Model Configuration ===
    
    // Model file paths (relative to service working directory)
    const std::string TFLITE_MODEL_PATH = "models/model_quantized.tflite";
    const std::string MODEL_METADATA_PATH = "models/model_metadata.json";
    
    // Model parameters
    constexpr int INPUT_FEATURES = 42;    // 7 VNA measurements × 6 statistics
    constexpr int OUTPUT_SIZE = 1;        // Single temperature value
    
    // Feature engineering parameters
    constexpr int VNA_MEASUREMENTS = 7;   // Number of VNA measurement types
    constexpr int STATISTICS_PER_MEASUREMENT = 6;  // Mean, std, min, max, 25th, 75th
    
    // === VNA Sensor Configuration ===
    
    // VNA measurement types (must match training data)
    const std::string VNA_MEASUREMENT_NAMES[7] = {
        "Return Loss (dB)",
        "Phase (degrees)",
        "Rs (ohms)", 
        "SWR (ratio)",
        "Xs (ohms)",
        "|Z| (impedance magnitude)",
        "Theta (impedance phase)"
    };
    
    // Statistical feature names
    const std::string STATISTICAL_FEATURE_NAMES[6] = {
        "Mean",
        "Standard Deviation", 
        "Minimum",
        "Maximum",
        "25th Percentile",
        "75th Percentile"
    };
    
    // VNA sensor interface settings
    const std::string VNA_DEVICE_PATH = "/dev/ttyUSB0";  // Adjust based on actual connection
    constexpr int VNA_BAUD_RATE = 115200;
    constexpr int VNA_TIMEOUT_MS = 1000;
    
    // === REST API Configuration ===
    
    // HTTP server settings
    constexpr int REST_API_PORT = 5000;
    constexpr int MAX_CONCURRENT_CONNECTIONS = 10;
    constexpr int HTTP_TIMEOUT_SECONDS = 30;
    
    // API endpoint paths
    const std::string ENDPOINT_LATEST = "/latest";
    const std::string ENDPOINT_HISTORY = "/history";
    const std::string ENDPOINT_HEALTH = "/health";
    const std::string ENDPOINT_METRICS = "/metrics";
    
    // === Logging Configuration ===
    
    // Log file settings
    const std::string LOG_DIRECTORY = "/var/log/tempsvc";
    const std::string LOG_FILE_PREFIX = "tempsvc";
    const std::string LOG_FILE_EXTENSION = ".log";
    
    // Log rotation settings
    constexpr long MAX_LOG_FILE_SIZE = 10 * 1024 * 1024;  // 10MB
    constexpr int MAX_LOG_FILES = 5;  // Keep 5 rotated files
    
    // Log levels (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
    constexpr int DEFAULT_LOG_LEVEL = 1;  // INFO level
    
    // === Data Storage Configuration ===
    
    // In-memory history buffer
    constexpr int HISTORY_BUFFER_SIZE = 1000;  // Keep last 1000 readings
    
    // CSV logging (optional)
    constexpr bool ENABLE_CSV_LOGGING = true;
    const std::string CSV_LOG_PATH = "/var/log/tempsvc/readings.csv";
    
    // === Performance Monitoring ===
    
    // Statistics reporting interval
    constexpr int STATS_REPORT_INTERVAL_SECONDS = 60;
    
    // Performance thresholds for alerts
    constexpr double MAX_ACCEPTABLE_ERROR_RATE = 0.05;  // 5% error rate
    constexpr int MIN_READINGS_PER_MINUTE = 300;        // Minimum throughput
    
    // === Model Confidence Thresholds ===
    
    // Confidence categories based on prediction variance
    constexpr double EXCELLENT_CONFIDENCE_THRESHOLD = 0.01;  // ≤1% error
    constexpr double GOOD_CONFIDENCE_THRESHOLD = 0.02;       // ≤2% error
    constexpr double FAIR_CONFIDENCE_THRESHOLD = 0.05;       // ≤5% error
    
    // Confidence level names
    const std::string CONFIDENCE_EXCELLENT = "EXCELLENT";
    const std::string CONFIDENCE_GOOD = "GOOD";
    const std::string CONFIDENCE_FAIR = "FAIR";
    const std::string CONFIDENCE_POOR = "POOR";
    
    // === Hardware-Specific Settings ===
    
    // Raspberry Pi 5 optimization flags
    constexpr bool ENABLE_ARM_NEON = true;      // Use ARM NEON SIMD instructions
    constexpr bool ENABLE_CPU_AFFINITY = true;  // Pin to performance cores
    constexpr int PREFERRED_CPU_CORE = 2;       // Use core 2 (performance core)
    
    // Memory settings
    constexpr size_t STACK_SIZE = 8 * 1024 * 1024;  // 8MB stack
    constexpr bool ENABLE_MEMORY_PREFETCH = true;    // Prefetch model data
    
    // === Development/Debug Settings ===
    
    #ifdef DEBUG
        constexpr bool ENABLE_DEBUG_OUTPUT = true;
        constexpr bool ENABLE_TIMING_LOGS = true;
        constexpr bool ENABLE_FEATURE_LOGGING = true;
    #else
        constexpr bool ENABLE_DEBUG_OUTPUT = false;
        constexpr bool ENABLE_TIMING_LOGS = false;
        constexpr bool ENABLE_FEATURE_LOGGING = false;
    #endif
    
    // === Service Health Monitoring ===
    
    // Health check parameters
    constexpr int HEALTH_CHECK_INTERVAL_SECONDS = 30;
    constexpr int MAX_CONSECUTIVE_FAILURES = 5;
    constexpr double MIN_SUCCESS_RATE = 0.95;  // 95% success rate required
    
    // === JSON Output Configuration ===
    
    // Precision for floating point values in JSON
    constexpr int TEMPERATURE_PRECISION = 3;    // 3 decimal places for temperature
    constexpr int TIMING_PRECISION = 1;         // 1 decimal place for timing
    
    // JSON field names (must match API specification)
    const std::string JSON_TIMESTAMP = "timestamp";
    const std::string JSON_TEMPERATURE = "temp_c";
    const std::string JSON_CONFIDENCE = "confidence";
    const std::string JSON_INFERENCE_TIME = "inference_time_ms";
    const std::string JSON_FEATURES_PROCESSED = "features_processed";
    const std::string JSON_VNA_MEASUREMENTS = "vna_measurements";
    
    // === Error Handling Configuration ===
    
    // Retry settings
    constexpr int MAX_SENSOR_READ_RETRIES = 3;
    constexpr int MAX_INFERENCE_RETRIES = 2;
    constexpr int RETRY_DELAY_MS = 100;
    
    // Error recovery settings
    constexpr int MODEL_RELOAD_THRESHOLD = 10;    // Reload model after 10 consecutive failures
    constexpr int SENSOR_RESET_THRESHOLD = 5;     // Reset sensor after 5 consecutive failures
    
    // === Weaviate Integration (Vector Database) ===
    
    // Weaviate connection settings
    const std::string WEAVIATE_HOST = "localhost";
    constexpr int WEAVIATE_PORT = 8080;
    const std::string WEAVIATE_SCHEME = "http";
    
    // Data ingestion settings
    constexpr bool ENABLE_WEAVIATE_INGESTION = true;
    constexpr int WEAVIATE_BATCH_SIZE = 100;
    constexpr int WEAVIATE_INGESTION_INTERVAL_SECONDS = 60;
    
    // === Build Information ===
    
    const std::string SERVICE_VERSION = "1.0.0";
    const std::string BUILD_TARGET = "ARM64-RaspberryPi5";
    const std::string BUILD_DATE = __DATE__ " " __TIME__;
    
    #ifdef __ARM_NEON
        const std::string SIMD_SUPPORT = "ARM NEON";
    #else
        const std::string SIMD_SUPPORT = "None";
    #endif
    
} // namespace Config

#endif // CONFIG_H 
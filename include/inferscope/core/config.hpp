#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace inferscope {
namespace config {

// ============================================================================
// Ring Buffer Configuration
// ============================================================================
constexpr size_t MAX_RING_BUFFER_SIZE = 1024;
constexpr size_t DEFAULT_RING_BUFFER_SIZE = 512;

// ============================================================================
// Model Limits
// ============================================================================
constexpr int MAX_ATTENTION_HEADS = 128;
constexpr int MAX_SEQ_LEN = 8192;
constexpr int MAX_LAYERS = 128;
constexpr int MAX_TOKENS_DISPLAY = 32;

// ============================================================================
// TUI Configuration
// ============================================================================
constexpr int REFRESH_RATE_MS = 50;        // 20 FPS
constexpr int SPARKLINE_HISTORY = 60;       // Number of data points in sparklines
constexpr int MAX_LIVE_ENTRIES = 200;       // Max visible forward pass entries
constexpr int ATTENTION_CELL_WIDTH = 7;     // Width of each cell in attention matrix
constexpr int MIN_TERMINAL_WIDTH = 120;
constexpr int MIN_TERMINAL_HEIGHT = 35;

// ============================================================================
// Anomaly Detection Thresholds
// ============================================================================
constexpr double ANOMALY_SPARSITY_THRESHOLD = 0.95;
constexpr double ANOMALY_ACTIVATION_MAX = 1e6;
constexpr double ANOMALY_ACTIVATION_MIN = -1e6;
constexpr double ANOMALY_DEAD_NEURON_THRESHOLD = 1e-7;
constexpr double LATENCY_WARNING_MS = 50.0;   // Yellow warning threshold
constexpr double LATENCY_CRITICAL_MS = 100.0;  // Red critical threshold

// ============================================================================
// Trace File Configuration
// ============================================================================
constexpr size_t TRACE_MAX_FILE_SIZE = 512 * 1024 * 1024;  // 512 MB max trace file
constexpr const char* TRACE_FILE_EXTENSION = ".nstrace";
constexpr const char* TRACE_DEFAULT_DIR = "./traces";

// ============================================================================
// Simulation Configuration  
// ============================================================================
constexpr double SIM_BASE_LATENCY_MS = 0.5;
constexpr double SIM_LATENCY_VARIANCE = 0.3;
constexpr double SIM_ATTN_LATENCY_MULTIPLIER = 2.5;
constexpr double SIM_MLP_LATENCY_MULTIPLIER = 1.8;
constexpr int SIM_DEFAULT_SEQ_LEN = 8;
constexpr int SIM_FORWARD_PASS_INTERVAL_MS = 500;  // Time between simulated passes

// ============================================================================
// Application Info
// ============================================================================
constexpr const char* APP_NAME = "inferscope";
constexpr const char* APP_VERSION = "1.0.0";
constexpr const char* APP_DESCRIPTION = "Local LLM Instrumentation, Tracing & Replay Platform";

} // namespace config
} // namespace inferscope

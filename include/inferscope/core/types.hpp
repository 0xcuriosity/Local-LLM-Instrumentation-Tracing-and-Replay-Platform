#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <cstdint>
#include <array>
#include <functional>
#include <memory>
#include <atomic>
#include <map>

namespace inferscope {

// ============================================================================
// Time Utilities
// ============================================================================
using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::duration<double, std::milli>;

// ============================================================================
// Enumerations
// ============================================================================

enum class LayerType : uint8_t {
    EMBEDDING = 0,
    ATTENTION,
    MLP,
    NORM,
    OUTPUT,
    LINEAR,
    ACTIVATION,
    RESIDUAL,
    ROPE,
    SOFTMAX,
    UNKNOWN
};

inline const char* layerTypeToString(LayerType t) {
    switch (t) {
        case LayerType::EMBEDDING:  return "Embedding";
        case LayerType::ATTENTION:  return "Attention";
        case LayerType::MLP:        return "MLP";
        case LayerType::NORM:       return "Norm";
        case LayerType::OUTPUT:     return "Output";
        case LayerType::LINEAR:     return "Linear";
        case LayerType::ACTIVATION: return "Activation";
        case LayerType::RESIDUAL:   return "Residual";
        case LayerType::ROPE:       return "RoPE";
        case LayerType::SOFTMAX:    return "Softmax";
        default:                    return "Unknown";
    }
}

inline const char* layerTypeIcon(LayerType t) {
    switch (t) {
        case LayerType::EMBEDDING:  return "◆";
        case LayerType::ATTENTION:  return "◉";
        case LayerType::MLP:        return "▣";
        case LayerType::NORM:       return "◇";
        case LayerType::OUTPUT:     return "▶";
        case LayerType::LINEAR:     return "─";
        case LayerType::ACTIVATION: return "⚡";
        case LayerType::RESIDUAL:   return "↻";
        case LayerType::ROPE:       return "⟳";
        case LayerType::SOFTMAX:    return "∑";
        default:                    return "?";
    }
}

enum class DataType : uint8_t {
    FLOAT32 = 0,
    FLOAT16,
    BFLOAT16,
    INT8,
    INT4
};

inline const char* dataTypeToString(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32:  return "float32";
        case DataType::FLOAT16:  return "float16";
        case DataType::BFLOAT16: return "bfloat16";
        case DataType::INT8:     return "int8";
        case DataType::INT4:     return "int4";
        default:                 return "unknown";
    }
}

inline size_t dataTypeSize(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32:  return 4;
        case DataType::FLOAT16:  return 2;
        case DataType::BFLOAT16: return 2;
        case DataType::INT8:     return 1;
        case DataType::INT4:     return 1; // packed
        default:                 return 0;
    }
}

enum class AnomalyFlag : uint16_t {
    NONE              = 0,
    CLIPPING_RISK     = 1 << 0,
    NAN_DETECTED      = 1 << 1,
    INF_DETECTED      = 1 << 2,
    HIGH_SPARSITY     = 1 << 3,
    OUTLIER_ACTIVATION = 1 << 4,
    DEAD_NEURONS      = 1 << 5,
    GRADIENT_EXPLODING = 1 << 6,
    GRADIENT_VANISHING = 1 << 7
};

inline AnomalyFlag operator|(AnomalyFlag a, AnomalyFlag b) {
    return static_cast<AnomalyFlag>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}

inline AnomalyFlag operator&(AnomalyFlag a, AnomalyFlag b) {
    return static_cast<AnomalyFlag>(static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
}

inline bool hasFlag(AnomalyFlag flags, AnomalyFlag flag) {
    return (static_cast<uint16_t>(flags) & static_cast<uint16_t>(flag)) != 0;
}

enum class CaptureState : uint8_t {
    IDLE = 0,
    CAPTURING,
    PAUSED,
    REPLAYING
};

// ============================================================================
// Data Structures
// ============================================================================

struct TensorShape {
    std::vector<int64_t> dims;
    
    int64_t numElements() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    std::string toString() const {
        std::string s = "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }
};

struct TensorMeta {
    TensorShape shape;
    DataType dtype = DataType::FLOAT32;
    size_t memoryBytes = 0;
    
    void computeMemory() {
        memoryBytes = shape.numElements() * dataTypeSize(dtype);
    }
};

struct ActivationStats {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double sparsity_rate = 0.0;  // fraction of zeros/near-zeros
    double l2_norm = 0.0;
    double abs_mean = 0.0;
    bool has_nan = false;
    bool has_inf = false;
    int64_t num_zeros = 0;
    int64_t num_elements = 0;
};

struct LayerLatency {
    int layer_id = -1;
    TimePoint start_time;
    TimePoint end_time;
    double delta_ms = 0.0;
    
    void compute() {
        delta_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

struct AttentionData {
    int head_id = 0;
    int num_heads = 0;
    int seq_len = 0;
    std::vector<float> weights;  // flattened [seq_len x seq_len] per head
    std::vector<std::string> tokens;  // token labels
};

struct LayerInfo {
    int id = -1;
    std::string name;
    std::string full_path;
    LayerType type = LayerType::UNKNOWN;
    int parent_id = -1;
    std::vector<int> children_ids;
    int64_t param_count = 0;
    bool expanded = false;
    bool is_capture_target = false;
    int depth = 0;
};

struct LayerSnapshot {
    int layer_id = -1;
    std::string layer_name;
    LayerType layer_type = LayerType::UNKNOWN;
    LayerLatency latency;
    ActivationStats activation_stats;
    TensorMeta tensor_meta;
    std::optional<AttentionData> attention_data;
    AnomalyFlag anomalies = AnomalyFlag::NONE;
    TimePoint timestamp;
};

struct ForwardPassRecord {
    uint64_t pass_id = 0;
    TimePoint timestamp;
    double total_latency_ms = 0.0;
    int token_count = 0;
    std::vector<LayerSnapshot> layer_snapshots;
    CaptureState state = CaptureState::IDLE;
    
    std::string timestampString() const {
        auto time_t = Clock::to_time_t(timestamp);
        auto tm = std::localtime(&time_t);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%H:%M:%S", tm);
        // Add milliseconds
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            timestamp.time_since_epoch()
        ).count() % 1000;
        std::string result(buf);
        result += ".";
        if (ms < 100) result += "0";
        if (ms < 10) result += "0";
        result += std::to_string(ms);
        return result;
    }
};

// ============================================================================
// Model Architecture Description
// ============================================================================

struct ModelConfig {
    std::string name = "llama-3-8b";
    int num_layers = 32;
    int hidden_dim = 4096;
    int num_heads = 32;
    int head_dim = 128;
    int intermediate_dim = 11008;
    int vocab_size = 128256;
    int max_seq_len = 2048;
    DataType dtype = DataType::FLOAT32;
    
    int64_t totalParams() const {
        int64_t params = 0;
        // Embedding
        params += static_cast<int64_t>(vocab_size) * hidden_dim;
        // Per layer: attention (Q,K,V,O) + MLP (gate, up, down) + 2x norm
        int64_t per_layer = 4LL * hidden_dim * hidden_dim           // Q,K,V,O
                          + 3LL * hidden_dim * intermediate_dim     // gate, up, down
                          + 2LL * hidden_dim;                        // 2x RMSNorm
        params += per_layer * num_layers;
        // Output norm + output projection
        params += hidden_dim + static_cast<int64_t>(vocab_size) * hidden_dim;
        return params;
    }
};

// ============================================================================
// Hook Callback Types
// ============================================================================

using PreForwardHook = std::function<void(int layer_id, const std::string& layer_name)>;
using PostForwardHook = std::function<void(int layer_id, const std::string& layer_name, 
                                            const ActivationStats& stats,
                                            const TensorMeta& meta,
                                            const LayerLatency& latency,
                                            const std::optional<AttentionData>& attn)>;

// ============================================================================
// Trace File Format
// ============================================================================

struct TraceFileHeader {
    char magic[8] = {'N','S','T','R','A','C','E','\0'};
    uint32_t version = 1;
    uint32_t num_records = 0;
    ModelConfig model_config;
    TimePoint start_time;
};

} // namespace inferscope

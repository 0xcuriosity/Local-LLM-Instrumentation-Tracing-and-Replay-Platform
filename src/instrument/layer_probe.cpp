#include "neuralscope/instrument/layer_probe.hpp"
#include <cmath>

namespace neuralscope {

LayerProbe::LayerProbe(int layer_id, std::string layer_name, LayerType layer_type)
    : layer_id_(layer_id), layer_name_(std::move(layer_name)), layer_type_(layer_type) {}

void LayerProbe::beginCapture() {
    latency_.layer_id = layer_id_;
    latency_.start_time = Clock::now();
}

void LayerProbe::endCapture(const float* data, size_t num_elements, const TensorShape& shape) {
    latency_.end_time = Clock::now();
    latency_.compute();
    
    meta_.shape = shape;
    meta_.dtype = DataType::FLOAT32;
    meta_.computeMemory();
    
    stats_ = computeActivationStats(data, num_elements);
    anomalies_ = detectAnomalies(stats_);
    captured_ = true;
}

LayerLatency LayerProbe::getLatency() const {
    return latency_;
}

ActivationStats LayerProbe::getStats() const {
    return stats_;
}

TensorMeta LayerProbe::getMeta() const {
    return meta_;
}

LayerSnapshot LayerProbe::getSnapshot() const {
    LayerSnapshot snap;
    snap.layer_id = layer_id_;
    snap.layer_name = layer_name_;
    snap.layer_type = layer_type_;
    snap.latency = latency_;
    snap.activation_stats = stats_;
    snap.tensor_meta = meta_;
    snap.anomalies = anomalies_;
    snap.timestamp = Clock::now();
    return snap;
}

ActivationStats LayerProbe::computeActivationStats(const float* data, size_t n) {
    ActivationStats stats;
    stats.num_elements = n;
    if (n == 0 || data == nullptr) return stats;
    
    double sum = 0.0;
    double sum_sq = 0.0;
    double abs_sum = 0.0;
    
    stats.min_val = data[0];
    stats.max_val = data[0];
    
    const double SPARSITY_THRESHOLD = 1e-6;
    
    for (size_t i = 0; i < n; ++i) {
        float val = data[i];
        
        if (std::isnan(val)) stats.has_nan = true;
        if (std::isinf(val)) stats.has_inf = true;
        
        sum += val;
        sum_sq += static_cast<double>(val) * val;
        abs_sum += std::abs(val);
        
        if (val < stats.min_val) stats.min_val = val;
        if (val > stats.max_val) stats.max_val = val;
        
        if (std::abs(val) < SPARSITY_THRESHOLD) {
            stats.num_zeros++;
        }
    }
    
    stats.mean = sum / n;
    double variance = (sum_sq / n) - (stats.mean * stats.mean);
    stats.std_dev = variance > 0 ? std::sqrt(variance) : 0.0;
    stats.sparsity_rate = static_cast<double>(stats.num_zeros) / n;
    stats.l2_norm = std::sqrt(sum_sq);
    stats.abs_mean = abs_sum / n;
    
    return stats;
}

double LayerProbe::computeSparsity(const float* data, size_t n, double threshold) {
    if (n == 0 || data == nullptr) return 0.0;
    int64_t zeros = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(data[i]) < threshold) {
            zeros++;
        }
    }
    return static_cast<double>(zeros) / n;
}

AnomalyFlag LayerProbe::detectAnomalies(const ActivationStats& stats) {
    AnomalyFlag flags = AnomalyFlag::NONE;
    
    if (stats.has_nan) flags = flags | AnomalyFlag::NAN_DETECTED;
    if (stats.has_inf) flags = flags | AnomalyFlag::INF_DETECTED;
    
    if (stats.sparsity_rate > config::ANOMALY_SPARSITY_THRESHOLD) {
        flags = flags | AnomalyFlag::HIGH_SPARSITY;
    }
    if (stats.max_val > config::ANOMALY_ACTIVATION_MAX || stats.min_val < config::ANOMALY_ACTIVATION_MIN) {
        flags = flags | AnomalyFlag::OUTLIER_ACTIVATION;
    }
    // Simple clipping risk heuristic (large chunk of values near max bounds usually indicates clipping in fp16/int8 before conversion)
    if (stats.max_val > 65000.0) { // Near fp16 max
        flags = flags | AnomalyFlag::CLIPPING_RISK;
    }
    
    return flags;
}

} // namespace neuralscope

#pragma once

#include <inferscope/core/types.hpp>
#include <inferscope/core/config.hpp>

#include <string>
#include <cstddef>
#include <cmath>

namespace inferscope {

/// Per-layer instrumentation probe.
///
/// Wraps a single layer identified by (layer_id, layer_name).  The caller
/// brackets the forward pass of that layer with beginCapture() / endCapture();
/// the probe records wall-clock timing, computes activation statistics from the
/// raw output tensor, and can flag anomalies.
class LayerProbe {
public:
    /// Construct a probe for a specific layer.
    explicit LayerProbe(int layer_id,
                        std::string layer_name,
                        LayerType layer_type = LayerType::UNKNOWN);

    ~LayerProbe() = default;

    // Copyable / movable – no mutex; one probe per layer per thread.
    LayerProbe(const LayerProbe&) = default;
    LayerProbe& operator=(const LayerProbe&) = default;
    LayerProbe(LayerProbe&&) = default;
    LayerProbe& operator=(LayerProbe&&) = default;

    // -----------------------------------------------------------------------
    // Capture lifecycle
    // -----------------------------------------------------------------------

    /// Record start timestamp.  Must be called before endCapture().
    void beginCapture();

    /// Record end timestamp and compute activation statistics from the raw
    /// output tensor.
    /// @param data         Pointer to contiguous float buffer (output activations).
    /// @param num_elements Number of floats in the buffer.
    /// @param shape        Logical shape of the tensor.
    void endCapture(const float* data,
                    size_t num_elements,
                    const TensorShape& shape);

    // -----------------------------------------------------------------------
    // Accessors (valid after endCapture)
    // -----------------------------------------------------------------------

    /// Return the latency recorded between begin/end.
    LayerLatency getLatency() const;

    /// Return the most-recently computed activation statistics.
    ActivationStats getStats() const;

    /// Return the tensor metadata captured in endCapture().
    TensorMeta getMeta() const;

    /// Build a full LayerSnapshot from the most recent capture.
    LayerSnapshot getSnapshot() const;

    // -----------------------------------------------------------------------
    // Anomaly detection
    // -----------------------------------------------------------------------

    /// Check an ActivationStats struct against the thresholds defined in
    /// config.hpp and return a bitmask of anomaly flags.
    static AnomalyFlag detectAnomalies(const ActivationStats& stats);

    // -----------------------------------------------------------------------
    // Static utilities
    // -----------------------------------------------------------------------

    /// Compute activation statistics from a raw float buffer.
    static ActivationStats computeActivationStats(const float* data, size_t n);

    /// Compute the fraction of elements whose absolute value is below
    /// @p threshold.
    static double computeSparsity(const float* data, size_t n,
                                  double threshold = 1e-7);

    // -----------------------------------------------------------------------
    // Identity
    // -----------------------------------------------------------------------

    int layerId() const { return layer_id_; }
    const std::string& layerName() const { return layer_name_; }
    LayerType layerType() const { return layer_type_; }

private:
    int layer_id_;
    std::string layer_name_;
    LayerType layer_type_;

    // Timing
    TimePoint start_time_;
    TimePoint end_time_;

    // Most-recently captured data
    ActivationStats stats_{};
    TensorMeta meta_{};
    LayerLatency latency_{};
    AnomalyFlag anomalies_ = AnomalyFlag::NONE;
    bool captured_ = false;
};

} // namespace inferscope

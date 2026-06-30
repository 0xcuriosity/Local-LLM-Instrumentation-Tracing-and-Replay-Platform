#pragma once

#include <inferscope/core/types.hpp>
#include <inferscope/core/config.hpp>

#include <string>
#include <fstream>
#include <cstdint>
#include <mutex>

namespace inferscope {

/// Binary trace file writer.
///
/// File format (all multi-byte integers are little-endian on x86):
///
///   ┌───────────────────────────────────────────────────────┐
///   │ Header                                                │
///   │   magic[8]    : "NSTRACE\0"                           │
///   │   version     : uint32                                │
///   │   num_records : uint32                                │
///   │   ModelConfig : (serialised inline – fixed fields)    │
///   │   start_time  : int64  (epoch nanoseconds)            │
///   ├───────────────────────────────────────────────────────┤
///   │ Record 0                                              │
///   │   record_size  : uint32  (bytes of payload that follow)│
///   │   pass_id      : uint64                               │
///   │   timestamp    : int64   (epoch ns)                   │
///   │   total_latency: double                               │
///   │   num_snapshots: uint32                               │
///   │   snapshot[0] … snapshot[N-1]                         │
///   ├───────────────────────────────────────────────────────┤
///   │ Record 1 …                                            │
///   └───────────────────────────────────────────────────────┘
///
/// Each snapshot:
///   layer_id    : int32
///   name_len    : uint16
///   name        : chars[name_len]
///   type        : uint8  (LayerType)
///   latency_ms  : double
///   stats       : ActivationStats (fixed-size)
///   shape_dims  : uint8  (number of dimensions)
///   dims        : int64[shape_dims]
///   has_attention: bool (uint8)
///   [if has_attention]
///     head_id   : int32
///     num_heads : int32
///     seq_len   : int32
///     weights   : float[seq_len*seq_len]
///     num_tokens: uint16
///     tokens    : (uint16 len + chars)[num_tokens]
///   anomalies   : uint16  (AnomalyFlag bitmask)
class TraceWriter {
public:
    /// Construct a writer; optionally open a file immediately.
    explicit TraceWriter() = default;
    explicit TraceWriter(const std::string& filename,
                         const ModelConfig& model_config);
    ~TraceWriter();

    // Non-copyable
    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;
    TraceWriter(TraceWriter&&) noexcept;
    TraceWriter& operator=(TraceWriter&&) noexcept;

    /// Open (or re-open) a file for writing.  Any previous file is closed.
    bool open(const std::string& filename);

    /// Write the header.  Must be called before writeRecord().
    bool writeHeader(const ModelConfig& model_config);

    /// Append a forward-pass record.
    bool writeRecord(const ForwardPassRecord& record);

    /// Flush and close the file.  Updates the num_records field in the header.
    void close();

    /// True if the underlying stream is open and good.
    bool isOpen() const;

    /// Number of records written so far.
    uint32_t numRecords() const;

    /// Current file size in bytes (approximate – based on stream position).
    size_t fileSize() const;

private:
    // Helpers
    void writeSnapshotBinary(const LayerSnapshot& snap);
    void writeActivationStatsBinary(const ActivationStats& stats);
    void updateRecordCount();

    std::ofstream stream_;
    std::string filename_;
    ModelConfig model_config_;
    uint32_t num_records_ = 0;
    bool header_written_ = false;
    mutable std::mutex mutex_;
};

} // namespace inferscope

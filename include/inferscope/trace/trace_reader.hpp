#pragma once

#include <inferscope/core/types.hpp>
#include <inferscope/core/config.hpp>

#include <string>
#include <fstream>
#include <vector>
#include <optional>
#include <cstdint>

namespace inferscope {

/// Binary trace file reader.
///
/// Reads files produced by TraceWriter.  On open(), the header is validated
/// (magic bytes and version) and a record offset index is built so that
/// seekToRecord() runs in O(1).
class TraceReader {
public:
    TraceReader() = default;
    explicit TraceReader(const std::string& filename);
    ~TraceReader();

    // Non-copyable
    TraceReader(const TraceReader&) = delete;
    TraceReader& operator=(const TraceReader&) = delete;
    TraceReader(TraceReader&&) noexcept;
    TraceReader& operator=(TraceReader&&) noexcept;

    /// Open a trace file and read + validate the header.
    /// Returns true on success.
    bool open(const std::string& filename);

    /// Return the header read from the file.
    TraceFileHeader readHeader();

    /// Read the next record sequentially.
    /// Returns std::nullopt when there are no more records or on error.
    std::optional<ForwardPassRecord> readRecord();

    /// Seek to a specific record index (0-based).
    /// The next call to readRecord() will return that record.
    bool seekToRecord(uint32_t index);

    /// Total number of records as declared in the header.
    uint32_t totalRecords() const;

    /// Index of the *next* record that readRecord() will return.
    uint32_t currentIndex() const;

    /// Convenience: read all records from the file.
    /// Resets the read position to the first record before reading.
    std::vector<ForwardPassRecord> readAll();

    /// Close the file.
    void close();

    /// True if the file is open and the header was validated.
    bool isOpen() const;

private:
    // Helpers
    bool buildRecordIndex();
    std::optional<LayerSnapshot> readSnapshotBinary();
    ActivationStats readActivationStatsBinary();

    std::ifstream stream_;
    std::string filename_;
    TraceFileHeader header_{};
    bool header_valid_ = false;

    // Stream offset of the first record (right after header)
    std::streampos data_start_ = 0;

    // Precomputed byte offsets for each record so seekToRecord is O(1)
    std::vector<std::streampos> record_offsets_;

    // Index of the next record to be returned by readRecord()
    uint32_t current_index_ = 0;
};

} // namespace inferscope

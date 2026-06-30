#include "inferscope/trace/trace_writer.hpp"
#include <iostream>

namespace inferscope {

TraceWriter::TraceWriter(const std::string& filename, const ModelConfig& model_config) 
    : model_config_(model_config) {
    open(filename);
}

TraceWriter::~TraceWriter() {
    close();
}

TraceWriter::TraceWriter(TraceWriter&& other) noexcept 
    : stream_(std::move(other.stream_)),
      filename_(std::move(other.filename_)),
      model_config_(std::move(other.model_config_)),
      num_records_(other.num_records_),
      header_written_(other.header_written_) {
}

TraceWriter& TraceWriter::operator=(TraceWriter&& other) noexcept {
    if (this != &other) {
        close();
        stream_ = std::move(other.stream_);
        filename_ = std::move(other.filename_);
        model_config_ = std::move(other.model_config_);
        num_records_ = other.num_records_;
        header_written_ = other.header_written_;
    }
    return *this;
}

bool TraceWriter::open(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_.is_open()) {
        stream_.close();
    }
    
    filename_ = filename;
    num_records_ = 0;
    header_written_ = false;
    
    stream_.open(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    return stream_.is_open();
}

bool TraceWriter::writeHeader(const ModelConfig& model_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open()) return false;
    
    model_config_ = model_config;
    
    TraceFileHeader header;
    header.model_config = model_config;
    header.start_time = Clock::now();
    
    stream_.write(header.magic, 8);
    stream_.write(reinterpret_cast<const char*>(&header.version), sizeof(header.version));
    
    // Will seek back here to update num_records
    stream_.write(reinterpret_cast<const char*>(&num_records_), sizeof(num_records_));
    
    uint16_t name_len = model_config.name.length();
    stream_.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    stream_.write(model_config.name.c_str(), name_len);
    
    stream_.write(reinterpret_cast<const char*>(&model_config.num_layers), sizeof(model_config.num_layers));
    stream_.write(reinterpret_cast<const char*>(&model_config.hidden_dim), sizeof(model_config.hidden_dim));
    stream_.write(reinterpret_cast<const char*>(&model_config.num_heads), sizeof(model_config.num_heads));
    stream_.write(reinterpret_cast<const char*>(&model_config.head_dim), sizeof(model_config.head_dim));
    stream_.write(reinterpret_cast<const char*>(&model_config.intermediate_dim), sizeof(model_config.intermediate_dim));
    stream_.write(reinterpret_cast<const char*>(&model_config.vocab_size), sizeof(model_config.vocab_size));
    stream_.write(reinterpret_cast<const char*>(&model_config.max_seq_len), sizeof(model_config.max_seq_len));
    
    auto ts = header.start_time.time_since_epoch().count();
    stream_.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
    
    header_written_ = true;
    return stream_.good();
}

bool TraceWriter::writeRecord(const ForwardPassRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open() || !header_written_) return false;
    
    std::string buffer;
    auto write_to_buf = [&](const void* ptr, size_t size) {
        buffer.append(reinterpret_cast<const char*>(ptr), size);
    };

    // 1. pass_id (uint64)
    write_to_buf(&record.pass_id, sizeof(record.pass_id));

    // 2. timestamp (int64 nanoseconds)
    int64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
        record.timestamp.time_since_epoch()
    ).count();
    write_to_buf(&ts, sizeof(ts));

    // 3. total_latency_ms (double)
    write_to_buf(&record.total_latency_ms, sizeof(record.total_latency_ms));

    // 4. token_count (int32)
    write_to_buf(&record.token_count, sizeof(record.token_count));

    // 5. num_snapshots (uint32)
    uint32_t num_snaps = record.layer_snapshots.size();
    write_to_buf(&num_snaps, sizeof(num_snaps));

    // 6. snapshots
    for (const auto& snap : record.layer_snapshots) {
        // layer_id (int32)
        write_to_buf(&snap.layer_id, sizeof(snap.layer_id));

        // name_len (uint16)
        uint16_t name_len = snap.layer_name.length();
        write_to_buf(&name_len, sizeof(name_len));

        // name (chars)
        write_to_buf(snap.layer_name.data(), name_len);

        // layer_type (uint8)
        uint8_t type_val = static_cast<uint8_t>(snap.layer_type);
        write_to_buf(&type_val, sizeof(type_val));

        // latency.delta_ms (double)
        write_to_buf(&snap.latency.delta_ms, sizeof(snap.latency.delta_ms));

        // activation_stats (ActivationStats struct)
        write_to_buf(&snap.activation_stats, sizeof(ActivationStats));

        // ndims (uint8)
        uint8_t ndims = snap.tensor_meta.shape.dims.size();
        write_to_buf(&ndims, sizeof(ndims));

        // dims (int64[])
        for (size_t j = 0; j < ndims; ++j) {
            write_to_buf(&snap.tensor_meta.shape.dims[j], sizeof(int64_t));
        }

        // has_attention (bool)
        bool has_attn = snap.attention_data.has_value();
        write_to_buf(&has_attn, sizeof(has_attn));

        if (has_attn) {
            const auto& attn = snap.attention_data.value();
            // head_id (int32)
            write_to_buf(&attn.head_id, sizeof(attn.head_id));

            // num_heads (int32)
            write_to_buf(&attn.num_heads, sizeof(attn.num_heads));

            // seq_len (int32)
            write_to_buf(&attn.seq_len, sizeof(attn.seq_len));

            // weights_size (uint32)
            uint32_t weights_size = attn.weights.size();
            write_to_buf(&weights_size, sizeof(weights_size));

            // weights (float[])
            write_to_buf(attn.weights.data(), weights_size * sizeof(float));

            // num_tokens (uint32)
            uint32_t num_tokens = attn.tokens.size();
            write_to_buf(&num_tokens, sizeof(num_tokens));

            // tokens
            for (const auto& tok : attn.tokens) {
                uint16_t tok_len = tok.length();
                write_to_buf(&tok_len, sizeof(tok_len));
                write_to_buf(tok.data(), tok_len);
            }
        }

        // anomalies (uint16)
        uint16_t flags = static_cast<uint16_t>(snap.anomalies);
        write_to_buf(&flags, sizeof(flags));
    }

    // Now write record_size (uint32) to the stream
    uint32_t record_size = buffer.size();
    stream_.write(reinterpret_cast<const char*>(&record_size), sizeof(record_size));

    // Then write the serialized buffer
    stream_.write(buffer.data(), buffer.size());

    num_records_++;
    return stream_.good();
}

void TraceWriter::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_.is_open()) {
        updateRecordCount();
        stream_.flush();
        stream_.close();
    }
}

bool TraceWriter::isOpen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stream_.is_open();
}

uint32_t TraceWriter::numRecords() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_records_;
}

size_t TraceWriter::fileSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_.is_open()) return 0;
    auto pos = const_cast<std::ofstream&>(stream_).tellp();
    return pos;
}

void TraceWriter::writeSnapshotBinary(const LayerSnapshot& snap) {
    // Helper implementation
}

void TraceWriter::writeActivationStatsBinary(const ActivationStats& stats) {
    // Helper implementation
}

void TraceWriter::updateRecordCount() {
    if (!stream_.is_open() || !header_written_) return;
    
    auto current_pos = stream_.tellp();
    stream_.seekp(8 + sizeof(uint32_t)); // magic + version
    stream_.write(reinterpret_cast<const char*>(&num_records_), sizeof(num_records_));
    stream_.seekp(current_pos);
}

} // namespace inferscope

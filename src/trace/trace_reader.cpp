#include "inferscope/trace/trace_reader.hpp"
#include <iostream>
#include <cstring>

namespace inferoscope {

TraceReader::TraceReader(const std::string& filename) {
    open(filename);
}

TraceReader::~TraceReader() {
    close();
}

TraceReader::TraceReader(TraceReader&& other) noexcept
    : stream_(std::move(other.stream_)),
      filename_(std::move(other.filename_)),
      header_(std::move(other.header_)),
      header_valid_(other.header_valid_),
      data_start_(other.data_start_),
      record_offsets_(std::move(other.record_offsets_)),
      current_index_(other.current_index_) {
}

TraceReader& TraceReader::operator=(TraceReader&& other) noexcept {
    if (this != &other) {
        close();
        stream_ = std::move(other.stream_);
        filename_ = std::move(other.filename_);
        header_ = std::move(other.header_);
        header_valid_ = other.header_valid_;
        data_start_ = other.data_start_;
        record_offsets_ = std::move(other.record_offsets_);
        current_index_ = other.current_index_;
    }
    return *this;
}

bool TraceReader::open(const std::string& filename) {
    close();
    filename_ = filename;
    stream_.open(filename, std::ios::binary | std::ios::in);
    if (!stream_.is_open()) return false;
    
    header_ = readHeader();
    return header_valid_;
}

TraceFileHeader TraceReader::readHeader() {
    TraceFileHeader header;
    if (!stream_.is_open()) return header;
    
    stream_.seekg(0);
    stream_.read(header.magic, 8);
    if (std::strncmp(header.magic, "NSTRACE", 7) != 0) {
        return header;
    }
    
    stream_.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    stream_.read(reinterpret_cast<char*>(&header.num_records), sizeof(header.num_records));
    
    uint16_t name_len;
    stream_.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    std::string name(name_len, '\0');
    stream_.read(&name[0], name_len);
    header.model_config.name = name;
    
    stream_.read(reinterpret_cast<char*>(&header.model_config.num_layers), sizeof(header.model_config.num_layers));
    stream_.read(reinterpret_cast<char*>(&header.model_config.hidden_dim), sizeof(header.model_config.hidden_dim));
    stream_.read(reinterpret_cast<char*>(&header.model_config.num_heads), sizeof(header.model_config.num_heads));
    stream_.read(reinterpret_cast<char*>(&header.model_config.head_dim), sizeof(header.model_config.head_dim));
    stream_.read(reinterpret_cast<char*>(&header.model_config.intermediate_dim), sizeof(header.model_config.intermediate_dim));
    stream_.read(reinterpret_cast<char*>(&header.model_config.vocab_size), sizeof(header.model_config.vocab_size));
    stream_.read(reinterpret_cast<char*>(&header.model_config.max_seq_len), sizeof(header.model_config.max_seq_len));
    
    int64_t ts;
    stream_.read(reinterpret_cast<char*>(&ts), sizeof(ts));
    header.start_time = TimePoint(std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(ts)));
    
    data_start_ = stream_.tellg();
    header_valid_ = true;
    
    buildRecordIndex();
    
    return header;
}

bool TraceReader::buildRecordIndex() {
    if (!header_valid_) return false;
    
    record_offsets_.clear();
    record_offsets_.push_back(data_start_);
    
    stream_.seekg(data_start_);
    for (uint32_t i = 0; i < header_.num_records; ++i) {
        uint32_t record_size;
        if (!stream_.read(reinterpret_cast<char*>(&record_size), sizeof(record_size))) break;
        stream_.seekg(record_size, std::ios::cur);
        record_offsets_.push_back(stream_.tellg());
    }
    
    stream_.seekg(data_start_);
    current_index_ = 0;
    return true;
}

std::optional<ForwardPassRecord> TraceReader::readRecord() {
    if (!header_valid_ || current_index_ >= header_.num_records) {
        return std::nullopt;
    }
    
    ForwardPassRecord record;
    
    uint32_t record_size;
    stream_.read(reinterpret_cast<char*>(&record_size), sizeof(record_size));
    
    stream_.read(reinterpret_cast<char*>(&record.pass_id), sizeof(record.pass_id));
    int64_t ts;
    stream_.read(reinterpret_cast<char*>(&ts), sizeof(ts));
    record.timestamp = TimePoint(std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(ts)));
    stream_.read(reinterpret_cast<char*>(&record.total_latency_ms), sizeof(record.total_latency_ms));
    stream_.read(reinterpret_cast<char*>(&record.token_count), sizeof(record.token_count));
    
    uint32_t num_snaps;
    stream_.read(reinterpret_cast<char*>(&num_snaps), sizeof(num_snaps));
    
    for (uint32_t i = 0; i < num_snaps; ++i) {
        LayerSnapshot snap;
        stream_.read(reinterpret_cast<char*>(&snap.layer_id), sizeof(snap.layer_id));
        
        uint16_t name_len;
        stream_.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        stream_.read(&name[0], name_len);
        snap.layer_name = name;
        
        uint8_t type_val;
        stream_.read(reinterpret_cast<char*>(&type_val), sizeof(type_val));
        snap.layer_type = static_cast<LayerType>(type_val);
        
        stream_.read(reinterpret_cast<char*>(&snap.latency.delta_ms), sizeof(snap.latency.delta_ms));
        stream_.read(reinterpret_cast<char*>(&snap.activation_stats), sizeof(ActivationStats));
        
        uint8_t ndims;
        stream_.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
        snap.tensor_meta.shape.dims.resize(ndims);
        for (int j = 0; j < ndims; ++j) {
            stream_.read(reinterpret_cast<char*>(&snap.tensor_meta.shape.dims[j]), sizeof(int64_t));
        }
        
        bool has_attn;
        stream_.read(reinterpret_cast<char*>(&has_attn), sizeof(has_attn));
        if (has_attn) {
            AttentionData attn;
            stream_.read(reinterpret_cast<char*>(&attn.head_id), sizeof(attn.head_id));
            stream_.read(reinterpret_cast<char*>(&attn.num_heads), sizeof(attn.num_heads));
            stream_.read(reinterpret_cast<char*>(&attn.seq_len), sizeof(attn.seq_len));
            
            uint32_t weights_size;
            stream_.read(reinterpret_cast<char*>(&weights_size), sizeof(weights_size));
            attn.weights.resize(weights_size);
            stream_.read(reinterpret_cast<char*>(attn.weights.data()), weights_size * sizeof(float));
            
            uint32_t num_tokens;
            stream_.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
            for (uint32_t k = 0; k < num_tokens; ++k) {
                uint16_t tok_len;
                stream_.read(reinterpret_cast<char*>(&tok_len), sizeof(tok_len));
                std::string tok(tok_len, '\0');
                stream_.read(&tok[0], tok_len);
                attn.tokens.push_back(tok);
            }
            snap.attention_data = std::move(attn);
        }
        
        uint16_t flags;
        stream_.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        snap.anomalies = static_cast<AnomalyFlag>(flags);
        
        record.layer_snapshots.push_back(std::move(snap));
    }
    
    current_index_++;
    return record;
}

bool TraceReader::seekToRecord(uint32_t index) {
    if (!header_valid_ || index >= header_.num_records) return false;
    stream_.seekg(record_offsets_[index]);
    current_index_ = index;
    return true;
}

std::vector<ForwardPassRecord> TraceReader::readAll() {
    std::vector<ForwardPassRecord> records;
    if (!header_valid_) return records;
    
    seekToRecord(0);
    while (auto rec = readRecord()) {
        records.push_back(std::move(*rec));
    }
    return records;
}

void TraceReader::close() {
    if (stream_.is_open()) {
        stream_.close();
    }
    header_valid_ = false;
}

bool TraceReader::isOpen() const {
    return stream_.is_open() && header_valid_;
}

uint32_t TraceReader::totalRecords() const {
    return header_valid_ ? header_.num_records : 0;
}

uint32_t TraceReader::currentIndex() const {
    return current_index_;
}

} 

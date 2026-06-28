#include "neuralscope/model/transformer.hpp"
#include <chrono>
#include <thread>
#include <random>

namespace neuralscope {

TransformerModel::TransformerModel(const ModelConfig& config) : config_(config) {}

int TransformerModel::addLayer(const std::string& name, const std::string& full_path, LayerType type, int parent_id, int64_t params) {
    LayerInfo info;
    info.id = next_layer_id_++;
    info.name = name;
    info.full_path = full_path;
    info.type = type;
    info.parent_id = parent_id;
    info.param_count = params;
    
    if (parent_id >= 0) {
        info.depth = layers_[parent_id].depth + 1;
        layers_[parent_id].children_ids.push_back(info.id);
    } else {
        info.depth = 0;
    }
    
    layers_.push_back(info);
    return info.id;
}

std::vector<LayerInfo> TransformerModel::buildTopology() {
    layers_.clear();
    next_layer_id_ = 0;
    
    // Root model layer
    int root_id = addLayer(config_.name, "model", LayerType::UNKNOWN, -1, config_.totalParams());
    layers_[root_id].expanded = true; // Root expanded by default
    
    // Embeddings
    int64_t embed_params = static_cast<int64_t>(config_.vocab_size) * config_.hidden_dim;
    embed_tokens_id_ = addLayer("embed_tokens", "model.embed_tokens", LayerType::EMBEDDING, root_id, embed_params);
    
    // Layers container
    int layers_container_id = addLayer("layers", "model.layers", LayerType::UNKNOWN, root_id, 0);
    layers_[layers_container_id].expanded = true;
    
    // Calculate param sizes
    int64_t qkvo_params = static_cast<int64_t>(config_.hidden_dim) * config_.hidden_dim;
    int64_t mlp_gate_up = static_cast<int64_t>(config_.hidden_dim) * config_.intermediate_dim;
    int64_t mlp_down = static_cast<int64_t>(config_.intermediate_dim) * config_.hidden_dim;
    int64_t norm_params = config_.hidden_dim;
    
    blocks_.clear();
    for (int i = 0; i < config_.num_layers; ++i) {
        std::string layer_name = "layers." + std::to_string(i);
        std::string layer_path = "model." + layer_name;
        
        int64_t block_params = 4*qkvo_params + 2*mlp_gate_up + mlp_down + 2*norm_params;
        int block_id = addLayer(std::to_string(i), layer_path, LayerType::UNKNOWN, layers_container_id, block_params);
        
        TransformerBlock block;
        block.id = block_id;
        block.prefix = layer_path;
        
        block.norm1_id = addLayer("input_layernorm", layer_path + ".input_layernorm", LayerType::NORM, block_id, norm_params);
        
        int self_attn_id = addLayer("self_attn", layer_path + ".self_attn", LayerType::ATTENTION, block_id, 4*qkvo_params);
        block.attn_q_id = addLayer("q_proj", layer_path + ".self_attn.q_proj", LayerType::LINEAR, self_attn_id, qkvo_params);
        block.attn_k_id = addLayer("k_proj", layer_path + ".self_attn.k_proj", LayerType::LINEAR, self_attn_id, qkvo_params);
        block.attn_v_id = addLayer("v_proj", layer_path + ".self_attn.v_proj", LayerType::LINEAR, self_attn_id, qkvo_params);
        block.attn_o_id = addLayer("o_proj", layer_path + ".self_attn.o_proj", LayerType::LINEAR, self_attn_id, qkvo_params);
        
        block.norm2_id = addLayer("post_attention_layernorm", layer_path + ".post_attention_layernorm", LayerType::NORM, block_id, norm_params);
        
        int mlp_id = addLayer("mlp", layer_path + ".mlp", LayerType::MLP, block_id, 2*mlp_gate_up + mlp_down);
        block.mlp_gate_id = addLayer("gate_proj", layer_path + ".mlp.gate_proj", LayerType::LINEAR, mlp_id, mlp_gate_up);
        block.mlp_up_id = addLayer("up_proj", layer_path + ".mlp.up_proj", LayerType::LINEAR, mlp_id, mlp_gate_up);
        block.mlp_down_id = addLayer("down_proj", layer_path + ".mlp.down_proj", LayerType::LINEAR, mlp_id, mlp_down);
        
        blocks_.push_back(block);
    }
    
    // Output Norm & LM Head
    output_norm_id_ = addLayer("norm", "model.norm", LayerType::NORM, root_id, norm_params);
    int64_t lm_head_params = static_cast<int64_t>(config_.hidden_dim) * config_.vocab_size;
    lm_head_id_ = addLayer("lm_head", "model.lm_head", LayerType::OUTPUT, root_id, lm_head_params);
    
    return layers_;
}

void TransformerModel::simulateLayer(int layer_id, const std::string& name, const TensorShape& shape, DataType dtype, 
                                     double base_latency_ms, HookManager& hooks, bool is_attention) {
    auto start_time = Clock::now();
    hooks.firePreHooks(layer_id, name);
    
    // Simulate work latency
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> lat_dist(base_latency_ms, base_latency_ms * config::SIM_LATENCY_VARIANCE);
    double actual_latency = std::max(0.1, lat_dist(gen));
    
    // Generate simulated tensor data
    double mean = 0.0;
    double stddev = 1.0;
    
    // Adjust stats based on layer type
    if (name.find("norm") != std::string::npos) {
        mean = 0.0;
        stddev = 1.0; // Normalized
    } else if (name.find("mlp") != std::string::npos) {
        mean = 0.5; // Relu-like activations
        stddev = 2.0;
    } else {
        mean = 0.1;
        stddev = 1.5;
    }
    
    Tensor t = Tensor::random(shape, mean, stddev);
    
    // Small chance for anomalies for demonstration
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    if (prob(gen) < 0.005) {
        // High sparsity anomaly
        for (size_t i = 0; i < t.shape().numElements() * 0.98; ++i) {
            t.data()[i] = 0.0f;
        }
    } else if (prob(gen) < 0.001) {
        // Exploding activation anomaly
        t.data()[0] = 5e6f;
    }
    
    ActivationStats stats = t.computeStats();
    
    std::optional<AttentionData> attn;
    if (is_attention) {
        AttentionData ad;
        ad.head_id = 0; // Just simulate one head for the overall hook
        ad.num_heads = config_.num_heads;
        ad.seq_len = shape.dims[1]; // batch, seq_len, hidden
        
        ad.weights.resize(ad.seq_len * ad.seq_len);
        
        // Generate a diagonal-heavy attention pattern with random variations
        std::uniform_real_distribution<double> noise_dist(-0.15, 0.15);
        for (int i = 0; i < ad.seq_len; ++i) {
            double sum = 0;
            for (int j = 0; j <= i; ++j) { // causal masking
                double val = std::exp(-std::abs(i - j) * 0.5) + std::abs(noise_dist(gen));
                ad.weights[i * ad.seq_len + j] = static_cast<float>(val);
                sum += val;
            }
            // Normalize row
            if (sum > 0) {
                for (int j = 0; j <= i; ++j) {
                    ad.weights[i * ad.seq_len + j] /= static_cast<float>(sum);
                }
            }
        }
        
        ad.tokens = {"<bos>", "Neural", "Scope", "is", "a", "TUI", "tool", "<eos>"};
        // Truncate or pad tokens to seq_len
        if (ad.tokens.size() > static_cast<size_t>(ad.seq_len)) {
            ad.tokens.resize(ad.seq_len);
        } else {
            while (ad.tokens.size() < static_cast<size_t>(ad.seq_len)) {
                ad.tokens.push_back("pad");
            }
        }
        attn = ad;
    }
    
    auto end_time = Clock::now();
    
    LayerLatency latency;
    latency.layer_id = layer_id;
    latency.start_time = start_time;
    latency.end_time = end_time;
    // Override computed latency with our simulated latency for realistic numbers without actually sleeping
    latency.delta_ms = actual_latency; 
    
    TensorMeta meta;
    meta.shape = shape;
    meta.dtype = dtype;
    meta.computeMemory();
    hooks.firePostHooks(layer_id, name, stats, meta, latency, attn);
}

void TransformerModel::simulateForwardPass(int seq_len, HookManager& hooks) {
    if (layers_.empty()) return;
    
    TensorShape token_shape = {{1, seq_len, config_.hidden_dim}}; // [batch, seq_len, hidden_dim]
    
    // Embeddings
    simulateLayer(embed_tokens_id_, layers_[embed_tokens_id_].full_path, token_shape, config_.dtype, 
                  config::SIM_BASE_LATENCY_MS, hooks);
                  
    // Blocks
    for (const auto& block : blocks_) {
        // Norm 1
        simulateLayer(block.norm1_id, layers_[block.norm1_id].full_path, token_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS, hooks);
                      
        // Attention
        simulateLayer(block.attn_q_id, layers_[block.attn_q_id].full_path, token_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS * config::SIM_ATTN_LATENCY_MULTIPLIER, hooks, true);
        simulateLayer(block.attn_k_id, layers_[block.attn_k_id].full_path, token_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS * config::SIM_ATTN_LATENCY_MULTIPLIER, hooks, true);
        simulateLayer(block.attn_v_id, layers_[block.attn_v_id].full_path, token_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS * config::SIM_ATTN_LATENCY_MULTIPLIER, hooks, true);
        simulateLayer(block.attn_o_id, layers_[block.attn_o_id].full_path, token_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS * config::SIM_ATTN_LATENCY_MULTIPLIER, hooks);
                      
        // Norm 2
        simulateLayer(block.norm2_id, layers_[block.norm2_id].full_path, token_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS, hooks);
                      
        // MLP
        TensorShape mlp_shape = {{1, seq_len, config_.intermediate_dim}};
        simulateLayer(block.mlp_gate_id, layers_[block.mlp_gate_id].full_path, mlp_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS * config::SIM_MLP_LATENCY_MULTIPLIER, hooks);
        simulateLayer(block.mlp_up_id, layers_[block.mlp_up_id].full_path, mlp_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS * config::SIM_MLP_LATENCY_MULTIPLIER, hooks);
        simulateLayer(block.mlp_down_id, layers_[block.mlp_down_id].full_path, token_shape, config_.dtype, 
                      config::SIM_BASE_LATENCY_MS * config::SIM_MLP_LATENCY_MULTIPLIER, hooks);
    }
    
    // Output Norm
    simulateLayer(output_norm_id_, layers_[output_norm_id_].full_path, token_shape, config_.dtype, 
                  config::SIM_BASE_LATENCY_MS, hooks);
                  
    // LM Head
    TensorShape vocab_shape = {{1, seq_len, config_.vocab_size}};
    simulateLayer(lm_head_id_, layers_[lm_head_id_].full_path, vocab_shape, config_.dtype, 
                  config::SIM_BASE_LATENCY_MS * 3.0, hooks); // usually a big matmul
}

} // namespace neuralscope

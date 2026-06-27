#pragma once

#include "inferscope/core/types.hpp"
#include "inferscope/core/config.hpp"
#include "inferscope/model/tensor.hpp"
#include "inferscope/instrument/hook_manager.hpp"

#include <vector>
#include <string>
#include <memory>

namespace inferscope {

// Simulated Transformer Block
struct TransformerBlock {
    int id;
    
    // Submodule layer IDs for hooking
    int norm1_id;
    int attn_q_id;
    int attn_k_id;
    int attn_v_id;
    int attn_o_id;
    int norm2_id;
    int mlp_gate_id;
    int mlp_up_id;
    int mlp_down_id;
    
    std::string prefix;
};

class TransformerModel {
public:
    explicit TransformerModel(const ModelConfig& config);
    
    std::vector<LayerInfo> buildTopology();
    
    // Simulates a forward pass through the model and triggers hooks
    void simulateForwardPass(int seq_len, HookManager& hooks);

private:
    ModelConfig config_;
    
    // Layer tracking for topology
    int next_layer_id_ = 0;
    std::vector<LayerInfo> layers_;
    
    // Submodules IDs
    int embed_tokens_id_;
    std::vector<TransformerBlock> blocks_;
    int output_norm_id_;
    int lm_head_id_;
    
    // Helper to register layers
    int addLayer(const std::string& name, const std::string& full_path, LayerType type, int parent_id, int64_t params);
    
    // Simulation helpers
    void simulateLayer(int layer_id, const std::string& name, const TensorShape& shape, DataType dtype, 
                       double base_latency_ms, HookManager& hooks, bool is_attention = false);
};

} // namespace inferscope

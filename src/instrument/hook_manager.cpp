#include "neuralscope/instrument/hook_manager.hpp"
#include <mutex>
#include <shared_mutex>

namespace neuralscope {

void HookManager::registerPreHook(int layer_id, PreForwardHook callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    pre_hooks_[layer_id].push_back(std::move(callback));
}

void HookManager::registerPostHook(int layer_id, PostForwardHook callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    post_hooks_[layer_id].push_back(std::move(callback));
}

void HookManager::registerGlobalPreHook(PreForwardHook callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    global_pre_hooks_.push_back(std::move(callback));
}

void HookManager::registerGlobalPostHook(PostForwardHook callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    global_post_hooks_.push_back(std::move(callback));
}

void HookManager::firePreHooks(int layer_id, const std::string& name) {
    if (!enabled_.load(std::memory_order_acquire)) return;
    
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    for (const auto& hook : global_pre_hooks_) {
        hook(layer_id, name);
    }
    
    auto it = pre_hooks_.find(layer_id);
    if (it != pre_hooks_.end()) {
        for (const auto& hook : it->second) {
            hook(layer_id, name);
        }
    }
}

void HookManager::firePostHooks(int layer_id, const std::string& name, 
                               const ActivationStats& stats, const TensorMeta& meta,
                               const LayerLatency& latency, const std::optional<AttentionData>& attn) {
    if (!enabled_.load(std::memory_order_acquire)) return;
    
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    for (const auto& hook : global_post_hooks_) {
        hook(layer_id, name, stats, meta, latency, attn);
    }
    
    auto it = post_hooks_.find(layer_id);
    if (it != post_hooks_.end()) {
        for (const auto& hook : it->second) {
            hook(layer_id, name, stats, meta, latency, attn);
        }
    }
}

void HookManager::clearHooks() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    pre_hooks_.clear();
    post_hooks_.clear();
    global_pre_hooks_.clear();
    global_post_hooks_.clear();
}

void HookManager::removeHook(int layer_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    pre_hooks_.erase(layer_id);
    post_hooks_.erase(layer_id);
}

void HookManager::setEnabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_release);
}

bool HookManager::isEnabled() const {
    return enabled_.load(std::memory_order_acquire);
}

} // namespace neuralscope

#pragma once

#include <inferscope/core/types.hpp>
#include <inferscope/core/config.hpp>

#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <atomic>
#include <string>
#include <functional>
#include <optional>

namespace inferscope {

/// Central instrumentation orchestrator.
///
/// Manages pre-forward and post-forward hooks that fire during model inference.
/// Hooks can be registered globally (applying to every layer) or per-layer.
/// All public methods are thread-safe: concurrent reads use a shared lock and
/// mutations take an exclusive lock.
class HookManager {
public:
    HookManager() = default;
    ~HookManager() = default;

    // Non-copyable, non-movable (owns mutex state)
    HookManager(const HookManager&) = delete;
    HookManager& operator=(const HookManager&) = delete;
    HookManager(HookManager&&) = delete;
    HookManager& operator=(HookManager&&) = delete;

    // -----------------------------------------------------------------------
    // Per-layer hook registration
    // -----------------------------------------------------------------------

    /// Register a pre-forward callback for a specific layer.
    void registerPreHook(int layer_id, PreForwardHook callback);

    /// Register a post-forward callback for a specific layer.
    void registerPostHook(int layer_id, PostForwardHook callback);

    // -----------------------------------------------------------------------
    // Global hook registration (applies to all layers)
    // -----------------------------------------------------------------------

    /// Register a global pre-forward callback (fires for every layer).
    void registerGlobalPreHook(PreForwardHook callback);

    /// Register a global post-forward callback (fires for every layer).
    void registerGlobalPostHook(PostForwardHook callback);

    // -----------------------------------------------------------------------
    // Hook invocation
    // -----------------------------------------------------------------------

    /// Fire all pre-hooks for the given layer: global hooks first, then
    /// per-layer hooks.
    void firePreHooks(int layer_id, const std::string& layer_name);

    /// Fire all post-hooks for the given layer with the collected data.
    void firePostHooks(int layer_id,
                       const std::string& layer_name,
                       const ActivationStats& stats,
                       const TensorMeta& meta,
                       const LayerLatency& latency,
                       const std::optional<AttentionData>& attn);

    // -----------------------------------------------------------------------
    // Management
    // -----------------------------------------------------------------------

    /// Remove all hooks (global and per-layer).
    void clearHooks();

    /// Remove hooks for a specific layer (does not affect global hooks).
    void removeHook(int layer_id);

    /// Enable or disable hook firing. When disabled, fire*Hooks() are no-ops.
    void setEnabled(bool enabled);

    /// Query whether the hook manager is enabled.
    bool isEnabled() const;

    /// Return the total number of registered hooks (global + per-layer).
    size_t hookCount() const;

private:
    mutable std::shared_mutex mutex_;
    std::atomic<bool> enabled_{true};

    // Per-layer hooks keyed by layer_id
    std::unordered_map<int, std::vector<PreForwardHook>> pre_hooks_;
    std::unordered_map<int, std::vector<PostForwardHook>> post_hooks_;

    // Global hooks that fire for every layer
    std::vector<PreForwardHook> global_pre_hooks_;
    std::vector<PostForwardHook> global_post_hooks_;
};

} // namespace inferscope

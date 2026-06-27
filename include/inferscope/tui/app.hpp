#pragma once

#include <memory>
#include <atomic>
#include <mutex>

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"

#include "inferscope/core/types.hpp"
#include "inferscope/core/config.hpp"
#include "inferscope/trace/trace_writer.hpp"
#include "inferscope/tui/theme.hpp"
#include "inferscope/tui/keybindings.hpp"
#include "inferscope/tui/panels/topology_panel.hpp"
#include "inferscope/tui/panels/live_panel.hpp"
#include "inferscope/tui/panels/attention_panel.hpp"
#include "inferscope/tui/panels/metrics_panel.hpp"
#include "inferscope/tui/panels/stats_panel.hpp"

namespace inferscope {

class App {
public:
    /// Construct the TUI application.
    /// @param model_config  Configuration of the model being inspected.
    /// @param layers        Layer topology (will be mutated for expand/collapse).
    /// @param buffer_size   Max trace buffer size.
    App(const ModelConfig& model_config, std::vector<LayerInfo> layers, int buffer_size = 512);

    /// Start the TUI main loop (blocking).
    void run();

    /// Thread-safe: push a new forward-pass record into all panels.
    void updateWithRecord(const ForwardPassRecord& record);

    /// Switch between live and replay modes.
    void setReplayMode(bool replay);

    /// Set trace writer for toggling recording from TUI.
    void setTraceWriter(std::shared_ptr<TraceWriter> writer);

    /// Set initial theme by index.
    void setInitialTheme(int theme_idx);

    /// Load trace record data for replay.
    void setReplayData(const std::vector<ForwardPassRecord>& records);

    /// Get the screen reference (for PostEvent from other threads).
    ftxui::ScreenInteractive& screen();

private:
    // ── Selection update helper ─────────────────────────────────────────
    void updateSelectedData();

    // ── Setup helpers ───────────────────────────────────────────────────
    void setupKeybindings();
    ftxui::Component buildUI();

    // ── Rendering helpers ───────────────────────────────────────────────
    ftxui::Element renderStatusBar() const;
    ftxui::Element renderCommandBar() const;
    ftxui::Element renderHelpOverlay() const;
    ftxui::Element renderMainLayout() const;

    // ── Panel focus management ──────────────────────────────────────────
    void cycleFocus();
    void updatePanelFocus();

    // ── State ───────────────────────────────────────────────────────────
    ModelConfig model_config_;
    std::vector<LayerInfo> layers_;
    int buffer_size_;

    ThemeEngine theme_engine_;
    KeybindingManager keybindings_;

    TopologyPanel topology_panel_;
    LivePanel live_panel_;
    AttentionPanel attention_panel_;
    MetricsPanel metrics_panel_;
    StatsPanel stats_panel_;

    ftxui::ScreenInteractive screen_;

    int active_panel_ = 0;
    static constexpr int NUM_PANELS = 5;

    std::atomic<uint64_t> pass_count_{0};
    std::atomic<bool> show_help_{false};
    std::atomic<bool> replay_mode_{false};
    CaptureState capture_state_ = CaptureState::IDLE;

    std::shared_ptr<TraceWriter> trace_writer_;

    mutable std::mutex data_mutex_;
};

} // namespace inferscope

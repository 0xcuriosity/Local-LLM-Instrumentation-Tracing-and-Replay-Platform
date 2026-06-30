#pragma once

#include <optional>

#include "ftxui/dom/elements.hpp"

#include "inferscope/core/types.hpp"
#include "inferscope/core/config.hpp"
#include "inferscope/tui/theme.hpp"

namespace inferscope {

class MetricsPanel {
public:
    MetricsPanel();

    /// Set the layer snapshot to display.
    void setSnapshot(const LayerSnapshot& snapshot);

    /// Clear displayed data.
    void clearData();

    /// Render the panel.
    ftxui::Element render(const ThemeEngine& theme) const;

    bool handleKey(const std::string& key);

    /// Set focused state.
    void setFocused(bool focused);

private:
    /// Format bytes to human-readable string (B / KB / MB / GB).
    static std::string formatBytes(size_t bytes);

    /// Render sparsity bar using block characters.
    ftxui::Element renderSparsityBar(const ThemeEngine& theme,
                                     double sparsity) const;

    /// Render latency status with colour coding.
    ftxui::Element renderLatencyStatus(const ThemeEngine& theme,
                                       double latency_ms) const;

    std::optional<LayerSnapshot> snapshot_;
    bool focused_ = false;
};

} // namespace inferscope

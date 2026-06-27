#pragma once

#include <vector>
#include <deque>

#include "ftxui/dom/elements.hpp"

#include "inferscope/core/types.hpp"
#include "inferscope/core/config.hpp"
#include "inferscope/tui/theme.hpp"

namespace inferscope {

class StatsPanel {
public:
    StatsPanel();

    /// Add a new stats entry with its anomaly flag.
    void addStats(const ActivationStats& stats, AnomalyFlag flags);

    /// Clear all stored history.
    void clearHistory();

    /// Render the panel.
    ftxui::Element render(const ThemeEngine& theme) const;

    bool handleKey(const std::string& key);

    /// Set focused state.
    void setFocused(bool focused);

private:
    /// Internal ring buffer entry.
    struct StatsEntry {
        ActivationStats stats;
        AnomalyFlag flags = AnomalyFlag::NONE;
    };

    /// Render a sparkline from a series of values.
    static ftxui::Element renderSparkline(const std::vector<double>& values,
                                          const ftxui::Color& color);

    /// Map a normalized value [0,1] → sparkline character.
    static const char* sparkChar(double normalized);

    /// Render anomaly indicator dot.
    static ftxui::Element anomalyDot(AnomalyFlag flags, const ThemeEngine& theme);

    /// Render active anomalies list.
    ftxui::Element renderAnomalyList(const ThemeEngine& theme) const;

    std::deque<StatsEntry> history_;
    static constexpr size_t MAX_HISTORY = config::SPARKLINE_HISTORY;
    bool focused_ = false;
};

} // namespace inferscope

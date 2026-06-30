#include "inferscope/tui/panels/stats_panel.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace inferscope {

StatsPanel::StatsPanel() : focused_(false) {}

void StatsPanel::addStats(const ActivationStats& stats, AnomalyFlag flags) {
    StatsEntry entry;
    entry.stats = stats;
    entry.flags = flags;
    history_.push_back(entry);
    if (history_.size() > MAX_HISTORY) {
        history_.pop_front();
    }
}

void StatsPanel::clearHistory() {
    history_.clear();
}

const char* StatsPanel::sparkChar(double normalized) {
    const char* blocks[] = {" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    int idx = static_cast<int>(normalized * 7.99);
    idx = std::max(0, std::min(7, idx));
    return blocks[idx];
}

ftxui::Element StatsPanel::renderSparkline(const std::vector<double>& values, const ftxui::Color& color) {
    using namespace ftxui;
    if (values.empty()) return text("");
    
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    
    if (max_val - min_val < 1e-6) {
        std::string result = "";
        for (size_t i = 0; i < values.size(); ++i) result += "▄";
        return text(result) | ftxui::color(color);
    }
    
    std::string result = "";
    for (double v : values) {
        double normalized = (v - min_val) / (max_val - min_val);
        result += sparkChar(normalized);
    }
    return text(result) | ftxui::color(color);
}

ftxui::Element StatsPanel::anomalyDot(AnomalyFlag flags, const ThemeEngine& theme) {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    
    if (flags == AnomalyFlag::NONE) {
        return text("● OK") | color(t.success);
    }
    
    bool has_crit = hasFlag(flags, AnomalyFlag::NAN_DETECTED) || 
                    hasFlag(flags, AnomalyFlag::INF_DETECTED);
    
    return text(has_crit ? "● CRITICAL " : "● WARNING ") | color(has_crit ? t.error : t.warning);
}

ftxui::Element StatsPanel::renderAnomalyList(const ThemeEngine& theme) const {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    
    if (history_.empty()) return text("");
    
    AnomalyFlag flags = history_.back().flags;
    if (flags == AnomalyFlag::NONE) return text("No anomalies") | color(t.dim);
    
    std::string desc = "";
    if (hasFlag(flags, AnomalyFlag::NAN_DETECTED)) desc += "NaN ";
    if (hasFlag(flags, AnomalyFlag::INF_DETECTED)) desc += "Inf ";
    if (hasFlag(flags, AnomalyFlag::CLIPPING_RISK)) desc += "ClipRisk ";
    if (hasFlag(flags, AnomalyFlag::HIGH_SPARSITY)) desc += "HighSparsity ";
    if (hasFlag(flags, AnomalyFlag::OUTLIER_ACTIVATION)) desc += "Outliers ";
    
    return text(desc) | color(t.fg);
}

ftxui::Element StatsPanel::render(const ThemeEngine& theme) const {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    
    if (history_.empty()) {
        auto content = text("No statistics data available") | center | color(t.border_color);
        auto title = text("5. NUMERICAL STATISTICS") | color(focused_ ? t.accent1 : t.fg) | bold;
        return window(title, content) 
               | (focused_ ? color(t.accent1) : color(t.border_color))
               | bgcolor(t.bg);
    }
    
    std::vector<double> means, stds, mins, maxs;
    for (const auto& s : history_) {
        means.push_back(s.stats.mean);
        stds.push_back(s.stats.std_dev);
        mins.push_back(s.stats.min_val);
        maxs.push_back(s.stats.max_val);
    }
    
    const auto& latest = history_.back().stats;
    
    Elements rows;
    
    auto sparkline_row = [&](const std::string& label, const std::vector<double>& vals, ftxui::Color c, double current) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << current;
        return hbox({
            text(label + " ") | color(t.fg) | size(WIDTH, EQUAL, 5),
            renderSparkline(vals, c) | flex,
            text(" " + ss.str()) | color(c) | size(WIDTH, EQUAL, 10)
        });
    };
    
    rows.push_back(sparkline_row("μ ", means, t.accent1, latest.mean));
    rows.push_back(sparkline_row("σ ", stds, t.accent2, latest.std_dev));
    rows.push_back(sparkline_row("↓ ", mins, t.accent3, latest.min_val));
    rows.push_back(sparkline_row("↑ ", maxs, t.error, latest.max_val));
    
    rows.push_back(separator() | color(t.border_color));
    
    Elements flags_row;
    flags_row.push_back(text("Status: ") | color(t.fg));
    flags_row.push_back(anomalyDot(history_.back().flags, theme));
    flags_row.push_back(renderAnomalyList(theme));
    
    rows.push_back(hbox(std::move(flags_row)));
    
    std::stringstream l2_ss;
    l2_ss << std::scientific << std::setprecision(2) << latest.l2_norm;
    rows.push_back(hbox({
        text("L2 Norm: ") | color(t.fg),
        text(l2_ss.str()) | color(t.accent1)
    }));
    
    auto content = vbox(std::move(rows)) | yframe;
    auto title = text("5. NUMERICAL STATISTICS") | color(focused_ ? t.accent1 : t.fg) | bold;
    
    return window(title, content)
           | (focused_ ? color(t.accent1) : color(t.border_color))
           | bgcolor(t.bg);
}

bool StatsPanel::handleKey(const std::string& key) {
    (void)key;
    return false;
}

void StatsPanel::setFocused(bool focused) {
    focused_ = focused;
}

} // namespace inferscope

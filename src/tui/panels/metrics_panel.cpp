#include "inferscope/tui/panels/metrics_panel.hpp"
#include <iomanip>
#include <sstream>

namespace inferscope {

MetricsPanel::MetricsPanel() : focused_(false) {}

void MetricsPanel::setSnapshot(const LayerSnapshot& snapshot) {
    snapshot_ = snapshot;
}

void MetricsPanel::clearData() {
    snapshot_ = std::nullopt;
}

ftxui::Element MetricsPanel::render(const ThemeEngine& theme) const {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    
    if (!snapshot_.has_value()) {
        auto content = text("No metrics data available") | center | color(t.border_color);
        auto title = text("4. RUNTIME METRICS INSPECTOR") | color(focused_ ? t.accent1 : t.fg) | bold;
        return window(title, content) 
               | (focused_ ? color(t.accent1) : color(t.border_color))
               | bgcolor(t.bg);
    }
    
    const auto& snap = snapshot_.value();
    
    Elements rows;
    
    // Shape & Dtype
    rows.push_back(hbox({
        text("Tensor Shape : ") | color(t.fg),
        text(snap.tensor_meta.shape.toString()) | color(t.accent1),
        text("   Dtype: ") | color(t.fg),
        text(dataTypeToString(snap.tensor_meta.dtype)) | color(t.accent2)
    }));
    
    // Sparsity Rate Bar
    std::stringstream sp_ss;
    sp_ss << std::fixed << std::setprecision(1) << (snap.activation_stats.sparsity_rate * 100) << "%";
    
    auto sparsity_color = t.success;
    if (snap.activation_stats.sparsity_rate > 0.9) sparsity_color = t.error;
    else if (snap.activation_stats.sparsity_rate > 0.7) sparsity_color = t.warning;
    
    rows.push_back(hbox({
        text("Sparsity Rate: ") | color(t.fg),
        gauge(snap.activation_stats.sparsity_rate) | color(sparsity_color) | size(WIDTH, EQUAL, 20),
        text(" " + sp_ss.str()) | color(t.fg)
    }));
    
    // Latency
    std::stringstream lat_ss;
    lat_ss << std::fixed << std::setprecision(3) << snap.latency.delta_ms << " ms";
    
    auto lat_color = t.success;
    std::string lat_desc = "(Within Normal Bounds)";
    if (snap.latency.delta_ms > config::LATENCY_CRITICAL_MS) {
        lat_color = t.error;
        lat_desc = "(Critical)";
    } else if (snap.latency.delta_ms > config::LATENCY_WARNING_MS) {
        lat_color = t.warning;
        lat_desc = "(Elevated)";
    }
    
    rows.push_back(hbox({
        text("Latency Delta: ") | color(t.fg),
        text(lat_ss.str()) | color(lat_color),
        text(" " + lat_desc) | color(t.border_color)
    }));
    
    // Memory footprint
    double mb = static_cast<double>(snap.tensor_meta.memoryBytes) / (1024 * 1024);
    std::stringstream mem_ss;
    mem_ss << std::fixed << std::setprecision(2) << mb << " MB";
    
    rows.push_back(hbox({
        text("Memory Size  : ") | color(t.fg),
        text(mem_ss.str()) | color(t.accent3)
    }));
    
    auto content = vbox(std::move(rows)) | yframe;
    auto title = text("4. RUNTIME METRICS INSPECTOR") | color(focused_ ? t.accent1 : t.fg) | bold;
    
    return window(title, content)
           | (focused_ ? color(t.accent1) : color(t.border_color))
           | bgcolor(t.bg);
}

bool MetricsPanel::handleKey(const std::string& key) {
    (void)key;
    return false;
}

void MetricsPanel::setFocused(bool focused) {
    focused_ = focused;
}

} // namespace inferscope

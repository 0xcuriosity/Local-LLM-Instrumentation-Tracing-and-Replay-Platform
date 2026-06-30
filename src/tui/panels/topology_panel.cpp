#include "inferscope/tui/panels/topology_panel.hpp"
#include <sstream>

namespace inferscope {

TopologyPanel::TopologyPanel(std::vector<LayerInfo>& layers) : layers_(layers) {
    rebuildDisplayList();
}

std::string TopologyPanel::formatParams(int64_t count) {
    if (count >= 1000000000) return std::to_string(count / 1000000000) + "B";
    if (count >= 1000000) return std::to_string(count / 1000000) + "M";
    if (count >= 1000) return std::to_string(count / 1000) + "K";
    return std::to_string(count);
}

void TopologyPanel::rebuildDisplayList() const {
    display_list_.clear();
    for (size_t i = 0; i < layers_.size(); ++i) {
        display_list_.push_back(i);
    }
    display_dirty_ = false;
}

ftxui::Element TopologyPanel::render(const ThemeEngine& theme) const {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    
    if (display_dirty_) {
        rebuildDisplayList();
    }
    
    Elements rows;
    for (int i = 0; i < static_cast<int>(display_list_.size()); ++i) {
        if (i < scroll_offset_ || i >= scroll_offset_ + visible_rows_) {
            continue;
        }
        
        int layer_idx = display_list_[i];
        const auto& layer = layers_[layer_idx];
        
        bool is_selected = (i == cursor_);
        
        std::string indent(layer.depth * 2, ' ');
        std::string icon = !layer.children_ids.empty() ? (layer.expanded ? "▼ " : "▶ ") : "  ";
        
        std::string text_str = indent + icon + layer.name;
        if (layer.param_count > 0) {
            text_str += " [" + formatParams(layer.param_count) + "]";
        }
        
        auto el = text(text_str) | color(t.fg);
        
        if (is_selected) {
            el = el | bgcolor(t.border_color) | bold;
        } else {
            el = el | bgcolor(t.bg);
        }
        
        rows.push_back(el);
    }
    
    auto content = vbox(std::move(rows)) | yframe;
    auto title = text("1. TOPOLOGY") | color(focused_ ? t.accent1 : t.fg) | bold;
                 
    return window(title, content) 
           | (focused_ ? color(t.accent1) : color(t.border_color))
           | bgcolor(t.bg);
}

bool TopologyPanel::handleKey(const std::string& key) {
    if (!focused_ || display_list_.empty()) return false;
    
    if (key == "j") {
        if (cursor_ + 1 < static_cast<int>(display_list_.size())) {
            cursor_++;
            if (cursor_ >= scroll_offset_ + visible_rows_) scroll_offset_++;
            return true;
        }
    } else if (key == "k") {
        if (cursor_ > 0) {
            cursor_--;
            if (cursor_ < scroll_offset_) scroll_offset_--;
            return true;
        }
    } else if (key == " " || key == "\n") {
        int layer_idx = display_list_[cursor_];
        layers_[layer_idx].expanded = !layers_[layer_idx].expanded;
        display_dirty_ = true;
        
        if (onCaptureTargetChanged) {
            onCaptureTargetChanged(layers_[layer_idx].id);
        }
        return true;
    }
    return false;
}

void TopologyPanel::setFocused(bool focused) {
    focused_ = focused;
}

int TopologyPanel::getSelectedLayerId() const {
    if (display_list_.empty() || cursor_ < 0 || cursor_ >= static_cast<int>(display_list_.size())) return -1;
    return layers_[display_list_[cursor_]].id;
}

} // namespace inferscope

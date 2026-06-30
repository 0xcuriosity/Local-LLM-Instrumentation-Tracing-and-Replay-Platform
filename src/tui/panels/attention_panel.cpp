#include "inferscope/tui/panels/attention_panel.hpp"
#include <cmath>

namespace inferscope {

AttentionPanel::AttentionPanel() : focused_(false) {}

void AttentionPanel::setAttentionData(const AttentionData& data) {
    data_ = data;
}

void AttentionPanel::clearData() {
    data_ = std::nullopt;
}

ftxui::Color AttentionPanel::weightColor(float val, const ThemeEngine& theme) {
    // Blue (0) -> Yellow (0.5) -> Red (1.0)
    float t = std::max(0.0f, std::min(1.0f, val));
    
    int r, g, b;
    if (t < 0.5f) {
        float s = t * 2.0f;
        r = static_cast<int>(s * 255);
        g = static_cast<int>(s * 255);
        b = static_cast<int>((1.0f - s) * 255);
    } else {
        float s = (t - 0.5f) * 2.0f;
        r = 255;
        g = static_cast<int>((1.0f - s) * 255);
        b = 0;
    }
    
    // In hacker theme, we just use monochrome green with varying intensity
    if (theme.themeName() == "Hacker") {
        int g_val = static_cast<int>(t * 255);
        return ftxui::Color::RGB(0, g_val, 0);
    }
    
    return ftxui::Color::RGB(r, g, b);
}

const char* AttentionPanel::intensityBlock(float val) {
    if (val < 0.05f) return " ";
    if (val < 0.25f) return "░";
    if (val < 0.50f) return "▒";
    if (val < 0.75f) return "▓";
    return "█";
}

ftxui::Element AttentionPanel::render(const ThemeEngine& theme) const {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    
    if (!data_.has_value()) {
        auto content = text("No attention data available") | center | color(t.border_color);
        auto title = text("3. ATTENTION MATRIX VISUALIZER") | color(focused_ ? t.accent1 : t.fg) | bold;
        return window(title, content) 
               | (focused_ ? color(t.accent1) : color(t.border_color))
               | bgcolor(t.bg);
    }
    
    const auto& d = data_.value();
    
    std::string title_str = "3. ATTENTION MATRIX (HEAD " + std::to_string(d.head_id) + "/" + std::to_string(d.num_heads) + ")";
    
    Elements rows;
    
    // Token Header
    Elements header_row;
    header_row.push_back(text("Tokens: ") | color(t.fg) | size(WIDTH, EQUAL, 10));
    for (int j = 0; j < d.seq_len; ++j) {
        std::string tok = "[" + d.tokens[j] + "]";
        if (tok.length() > static_cast<size_t>(config::ATTENTION_CELL_WIDTH)) {
            tok = tok.substr(0, config::ATTENTION_CELL_WIDTH - 2) + "..";
        }
        header_row.push_back(text(tok) | color(t.accent3) | size(WIDTH, EQUAL, config::ATTENTION_CELL_WIDTH));
    }
    rows.push_back(hbox(std::move(header_row)));
    
    // Matrix Rows
    for (int i = 0; i < d.seq_len; ++i) {
        Elements matrix_row;
        
        // Row token label
        std::string tok = "[" + d.tokens[i] + "]";
        if (tok.length() > 9) tok = tok.substr(0, 7) + "..";
        matrix_row.push_back(text(tok) | color(t.accent3) | size(WIDTH, EQUAL, 10));
        
        // Cells
        for (int j = 0; j < d.seq_len; ++j) {
            float val = d.weights[i * d.seq_len + j];
            auto cell_color = weightColor(val, theme);
            std::string block = intensityBlock(val);
            
            // Pad block to cell width
            std::string padded;
            for(int k=0; k<config::ATTENTION_CELL_WIDTH-1; k++) padded += block;
            
            matrix_row.push_back(text(padded + " ") | color(cell_color) | size(WIDTH, EQUAL, config::ATTENTION_CELL_WIDTH));
        }
        rows.push_back(hbox(std::move(matrix_row)));
    }
    
    auto content = vbox(std::move(rows)) | yframe;
    
    return window(text(title_str) | color(focused_ ? t.accent1 : t.fg) | bold, content)
           | (focused_ ? color(t.accent1) : color(t.border_color))
           | bgcolor(t.bg);
}

bool AttentionPanel::handleKey(const std::string& key) {
    if (!focused_ || !data_.has_value()) return false;
    
    if (key == "h") {
        // We can't actually change the head data here, the data is pushed from the simulation.
        // In a real implementation with random access to the trace, we would trigger a data reload.
        // For now, we consume the key.
        return true;
    } else if (key == "l") {
        return true;
    }
    return false;
}

void AttentionPanel::setFocused(bool focused) {
    focused_ = focused;
}

} // namespace inferscope

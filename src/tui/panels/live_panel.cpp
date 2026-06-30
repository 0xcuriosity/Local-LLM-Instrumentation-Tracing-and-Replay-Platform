#include "inferscope/tui/panels/live_panel.hpp"
#include <iomanip>
#include <sstream>

namespace inferscope {

LivePanel::LivePanel() : cursor_(-1), focused_(false) {}

void LivePanel::addRecord(const ForwardPassRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back(record);
    if (records_.size() > config::MAX_LIVE_ENTRIES) {
        records_.erase(records_.begin());
    }
    if (cursor_ == -1 && !records_.empty()) {
        cursor_ = records_.size() - 1;
    } else if (cursor_ > 0 && records_.size() >= config::MAX_LIVE_ENTRIES) {
        cursor_--;
    }
}

void LivePanel::setRecords(const std::vector<ForwardPassRecord>& records) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_ = records;
    if (!records_.empty()) {
        cursor_ = records_.size() - 1;
    } else {
        cursor_ = -1;
    }
}

ftxui::Element LivePanel::render(const ThemeEngine& theme) const {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (records_.empty()) {
        auto content = text("Waiting for forward passes...") | center | color(t.border_color);
        auto title = text("2. LIVE PASSES") | color(focused_ ? t.accent1 : t.fg) | bold;
        return window(title, content) 
               | (focused_ ? color(t.accent1) : color(t.border_color))
               | bgcolor(t.bg);
    }
    
    Elements items;
    
    items.push_back(hbox({
        text(" ID  ") | bold | color(t.accent1) | size(WIDTH, EQUAL, 5),
        text("│ Timestamp    ") | bold | color(t.accent1) | size(WIDTH, EQUAL, 15),
        text("│ ms   ") | bold | color(t.accent1) | size(WIDTH, EQUAL, 7),
        text("│ Tok") | bold | color(t.accent1)
    }));
    items.push_back(separator() | color(t.border_color));
    
    for (size_t i = 0; i < records_.size(); ++i) {
        if (static_cast<int>(i) < scroll_offset_ || static_cast<int>(i) >= scroll_offset_ + visible_rows_) {
            continue;
        }
        
        const auto& rec = records_[i];
        bool is_selected = (static_cast<int>(i) == cursor_);
        
        std::stringstream id_ss;
        id_ss << std::setw(3) << std::setfill('0') << (rec.pass_id % 1000);
        
        std::stringstream ts_ss;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(rec.timestamp.time_since_epoch()).count() % 100000;
        ts_ss << std::setw(8) << std::setfill(' ') << ms << "ms";
        
        std::stringstream lat_ss;
        lat_ss << std::setw(4) << std::setfill(' ') << std::fixed << std::setprecision(0) << rec.total_latency_ms;
        
        ftxui::Color lat_color = t.success;
        if (rec.total_latency_ms > 100.0) lat_color = t.error;
        else if (rec.total_latency_ms > 50.0) lat_color = t.warning;
        
        auto row = hbox({
            text(" " + id_ss.str() + " ") | color(t.fg) | size(WIDTH, EQUAL, 5),
            text("│ " + ts_ss.str() + " ") | color(t.fg) | size(WIDTH, EQUAL, 15),
            text("│ " + lat_ss.str() + " ") | color(lat_color) | size(WIDTH, EQUAL, 7),
            text("│ " + std::to_string(rec.token_count)) | color(t.fg)
        });
        
        if (is_selected) {
            row = row | bgcolor(t.border_color) | bold;
        } else {
            row = row | bgcolor(t.bg);
        }
        
        items.push_back(row);
    }
    
    auto content = vbox(std::move(items)) | yframe;
    auto title = text("2. LIVE PASSES") | color(focused_ ? t.accent1 : t.fg) | bold;
    return window(title, content) 
           | (focused_ ? color(t.accent1) : color(t.border_color))
           | bgcolor(t.bg);
}

bool LivePanel::handleKey(const std::string& key) {
    if (!focused_ || records_.empty()) return false;
    
    if (key == "j") {
        if (cursor_ + 1 < static_cast<int>(records_.size())) {
            cursor_++;
            if (cursor_ >= scroll_offset_ + visible_rows_) {
                scroll_offset_++;
            }
            return true;
        }
    } else if (key == "k") {
        if (cursor_ > 0) {
            cursor_--;
            if (cursor_ < scroll_offset_) {
                scroll_offset_--;
            }
            return true;
        }
    } else if (key == "\n" || key == " ") {
        return true;
    }
    return false;
}

void LivePanel::setFocused(bool focused) {
    focused_ = focused;
}

std::optional<ForwardPassRecord> LivePanel::getSelectedRecord() const {
    if (cursor_ >= 0 && cursor_ < static_cast<int>(records_.size())) {
        return records_[cursor_];
    }
    return std::nullopt;
}

} // namespace inferscope

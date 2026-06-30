#pragma once

#include <vector>
#include <optional>
#include <mutex>

#include "ftxui/dom/elements.hpp"

#include "inferscope/core/types.hpp"
#include "inferscope/core/config.hpp"
#include "inferscope/tui/theme.hpp"

namespace inferscope {

class LivePanel {
public:
    LivePanel();

    /// Append a new forward-pass record (thread-safe).
    void addRecord(const ForwardPassRecord& record);

    /// Set all records (for replay mode).
    void setRecords(const std::vector<ForwardPassRecord>& records);

    /// Render the panel.
    ftxui::Element render(const ThemeEngine& theme) const;

    /// Handle a key press. Returns true if consumed.
    bool handleKey(const std::string& key);

    /// Currently selected record, if any.
    std::optional<ForwardPassRecord> getSelectedRecord() const;

    /// Set focused state.
    void setFocused(bool focused);

private:
    mutable std::mutex mutex_;
    std::vector<ForwardPassRecord> records_;
    int cursor_ = 0;
    int scroll_offset_ = 0;
    bool focused_ = false;
    bool auto_scroll_ = true;
    int visible_rows_ = 18;
};

} // namespace inferscope

#pragma once

#include <vector>
#include <string>
#include <functional>

#include "ftxui/dom/elements.hpp"

#include "inferscope/core/types.hpp"
#include "inferscope/tui/theme.hpp"

namespace inferscope {

class TopologyPanel {
public:
    /// Construct with a reference to the layer topology vector.
    explicit TopologyPanel(std::vector<LayerInfo>& layers);

    /// Render the panel using the current theme.
    ftxui::Element render(const ThemeEngine& theme) const;

    /// Handle a key press. Returns true if consumed.
    bool handleKey(const std::string& key);

    /// Set whether this panel is the focused panel.
    void setFocused(bool focused);

    /// Get the layer id of the currently selected row.
    int getSelectedLayerId() const;

    /// Callback fired when a capture target changes.
    std::function<void(int layer_id)> onCaptureTargetChanged;

private:
    /// Build a flat display list honouring expand/collapse state.
    void rebuildDisplayList() const;

    /// Format a parameter count to a human-readable string.
    static std::string formatParams(int64_t count);

    std::vector<LayerInfo>& layers_;
    bool focused_ = false;
    int cursor_ = 0;               // index into display_list_
    mutable std::vector<int> display_list_; // layer ids in display order
    mutable bool display_dirty_ = true;
    int scroll_offset_ = 0;
    int visible_rows_ = 20;        // sensible default, adjusted dynamically
};

} // namespace inferscope

#pragma once

#include <optional>

#include "ftxui/dom/elements.hpp"

#include "inferscope/core/types.hpp"
#include "inferscope/core/config.hpp"
#include "inferscope/tui/theme.hpp"

namespace inferscope {

class AttentionPanel {
public:
    AttentionPanel();

    /// Set the attention data to visualize.
    void setAttentionData(const AttentionData& data);

    /// Clear displayed data.
    void clearData();

    /// Render the panel.
    ftxui::Element render(const ThemeEngine& theme) const;

    /// Handle a key press. Returns true if consumed.
    bool handleKey(const std::string& key);

    /// Set focused state.
    void setFocused(bool focused);

private:
    /// Map weight [0,1] → intensity block character.
    static const char* intensityBlock(float w);

    /// Map weight [0,1] → gradient colour (blue → yellow → red).
    static ftxui::Color weightColor(float w, const ThemeEngine& theme);

    /// Truncate a token string to a maximum length.
    static std::string truncToken(const std::string& tok, int max_len = 6);

    std::optional<AttentionData> data_;
    int current_head_ = 0;
    bool focused_ = false;
};

} // namespace inferscope

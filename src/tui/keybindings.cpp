#include "inferscope/tui/keybindings.hpp"

namespace inferscope {

KeybindingManager::KeybindingManager() {}

void KeybindingManager::registerBinding(const std::string& key_label,
                                         const std::string& action,
                                         const std::string& description,
                                         std::function<bool()> handler) {
    Binding b = {{key_label, action, description}, std::move(handler)};
    bindings_.push_back(std::move(b));
    rebuildCache();
}

bool KeybindingManager::handleEvent(const ftxui::Event& event) const {
    if (event.is_character() || event == ftxui::Event::Tab || event == ftxui::Event::Return) {
        std::string key_str = event.character();
        if (event == ftxui::Event::Tab) key_str = "Tab";
        else if (event == ftxui::Event::Return) key_str = "Enter";
        else if (event.character() == " ") key_str = "Space";

        for (const auto& b : bindings_) {
            if (b.info.key_label == key_str) {
                if (b.handler()) {
                    return true;
                }
            }
        }
    }
    return false;
}

const std::vector<BindingInfo>& KeybindingManager::getBindings() const {
    return info_cache_;
}

void KeybindingManager::rebuildCache() {
    info_cache_.clear();
    for (const auto& b : bindings_) {
        info_cache_.push_back(b.info);
    }
}

ftxui::Element KeybindingManager::renderHelpOverlay(const ThemeEngine& theme) const {
    using namespace ftxui;
    const auto& t = theme.currentTheme();
    
    Elements rows;
    rows.push_back(hbox({
        text("Key") | bold | color(t.accent1) | size(WIDTH, EQUAL, 10),
        text("Action") | bold | color(t.fg) | size(WIDTH, EQUAL, 20),
        text("Description") | bold | color(t.fg)
    }));
    rows.push_back(separator() | color(t.border_color));

    for (const auto& b : bindings_) {
        rows.push_back(hbox({
            text(b.info.key_label) | color(t.highlight) | size(WIDTH, EQUAL, 10),
            text(b.info.action) | color(t.fg) | size(WIDTH, EQUAL, 20),
            text(b.info.description) | color(t.fg)
        }));
    }

    return vbox(std::move(rows))
        | border
        | color(t.border_color)
        | bgcolor(t.bg)
        | clear_under
        | center;
}

} // namespace inferscope

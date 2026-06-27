#pragma once

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"

#include "inferscope/tui/theme.hpp"

namespace inferscope {

// ============================================================================
// Binding descriptor visible to the help overlay
// ============================================================================

struct BindingInfo {
    std::string key_label;   // e.g. "Tab", "j", "?"
    std::string action;      // e.g. "Cycle Panels"
    std::string description; // e.g. "Move focus to the next panel"
};

// ============================================================================
// KeybindingManager
// ============================================================================

class KeybindingManager {
public:
    KeybindingManager();

    /// Register a key binding.
    /// @param key_label  Display label for the key (e.g. "Tab", "q").
    /// @param action     Short action name.
    /// @param description  Longer description for help overlay.
    /// @param handler    Callback returning true if event was consumed.
    void registerBinding(const std::string& key_label,
                         const std::string& action,
                         const std::string& description,
                         std::function<bool()> handler);

    /// Try to match an FTXUI Event and execute the handler.
    /// @return true if a matching binding was found and consumed the event.
    bool handleEvent(const ftxui::Event& event) const;

    /// All registered bindings (for help overlay rendering).
    const std::vector<BindingInfo>& getBindings() const;

    /// Render the help overlay as an FTXUI Element.
    ftxui::Element renderHelpOverlay(const ThemeEngine& theme) const;

private:
    struct Binding {
        BindingInfo info;
        std::function<bool()> handler;
    };

    /// Map from FTXUI event string representation → Binding.
    std::vector<Binding> bindings_;

    /// Cached info vector for external queries.
    std::vector<BindingInfo> info_cache_;

    /// Rebuild info_cache_ from bindings_.
    void rebuildCache();

    /// Convert a key label to the FTXUI character string used for matching.
    static std::string labelToEventChar(const std::string& label);
};

} // namespace inferscope

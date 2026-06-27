#pragma once

#include <string>
#include <array>
#include <cstdint>

#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"

namespace inferscope {

// ============================================================================
// Theme Definition
// ============================================================================

struct Theme {
    std::string name;

    ftxui::Color bg;
    ftxui::Color fg;
    ftxui::Color accent1;    // Primary accent (blue-ish)
    ftxui::Color accent2;    // Secondary accent (green-ish)
    ftxui::Color accent3;    // Tertiary accent (amber-ish)
    ftxui::Color border_color;
    ftxui::Color highlight;
    ftxui::Color warning;
    ftxui::Color error;
    ftxui::Color success;
    ftxui::Color dim;        // Dimmed / muted text
};

// ============================================================================
// Theme Indices
// ============================================================================

enum class ThemeId : uint8_t {
    DARK = 0,
    LIGHT,
    HACKER,
    COUNT_  // sentinel
};

// ============================================================================
// ThemeEngine
// ============================================================================

class ThemeEngine {
public:
    ThemeEngine();

    /// Returns the currently active theme.
    const Theme& currentTheme() const;

    /// Cycle to the next theme (Dark → Light → Hacker → Dark).
    void nextTheme();

    /// Set a specific theme by id.
    void setTheme(ThemeId id);

    /// Human-readable name of the current theme.
    const std::string& themeName() const;

    // ── Styled element helpers ──────────────────────────────────────────

    /// Wrap an element with the current theme's border color.
    ftxui::Element styledBorder(ftxui::Element inner) const;

    /// Wrap an element with a focused (bright accent) border.
    ftxui::Element focusedBorder(ftxui::Element inner) const;

    /// Text coloured with the theme foreground.
    ftxui::Element styledText(const std::string& s) const;

    /// Text coloured with the primary accent.
    ftxui::Element accentText(const std::string& s) const;

    /// Text coloured with the warning colour.
    ftxui::Element warningText(const std::string& s) const;

    /// Text coloured with the error colour.
    ftxui::Element errorText(const std::string& s) const;

    /// Text coloured with the success colour.
    ftxui::Element successText(const std::string& s) const;

    /// Text coloured with the dim colour.
    ftxui::Element dimText(const std::string& s) const;

    /// Highlighted row background.
    ftxui::Decorator highlightBg() const;

    /// Apply theme background to an element.
    ftxui::Element withBackground(ftxui::Element inner) const;

private:
    void buildThemes();

    ThemeId active_id_ = ThemeId::DARK;
    std::array<Theme, static_cast<size_t>(ThemeId::COUNT_)> themes_;
};

} // namespace inferscope

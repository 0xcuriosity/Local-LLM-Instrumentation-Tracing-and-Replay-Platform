#include "inferscope/tui/theme.hpp"

namespace inferscope {

ThemeEngine::ThemeEngine() {
    buildThemes();
}

void ThemeEngine::buildThemes() {
    Theme dark;
    dark.name = "Dark";
    dark.bg = ftxui::Color::RGB(26, 27, 38);
    dark.fg = ftxui::Color::RGB(192, 202, 245);
    dark.accent1 = ftxui::Color::RGB(122, 162, 247);
    dark.accent2 = ftxui::Color::RGB(158, 206, 106);
    dark.accent3 = ftxui::Color::RGB(224, 175, 104);
    dark.border_color = ftxui::Color::RGB(59, 66, 97);
    dark.highlight = ftxui::Color::RGB(255, 158, 100);
    dark.warning = ftxui::Color::RGB(224, 175, 104);
    dark.error = ftxui::Color::RGB(247, 118, 142);
    dark.success = ftxui::Color::RGB(158, 206, 106);
    dark.dim = ftxui::Color::RGB(86, 95, 137);
    themes_[static_cast<size_t>(ThemeId::DARK)] = dark;

    Theme light;
    light.name = "Light";
    light.bg = ftxui::Color::RGB(213, 214, 219);
    light.fg = ftxui::Color::RGB(52, 59, 88);
    light.accent1 = ftxui::Color::RGB(52, 84, 138);
    light.accent2 = ftxui::Color::RGB(51, 99, 92);
    light.accent3 = ftxui::Color::RGB(143, 94, 21);
    light.border_color = ftxui::Color::RGB(150, 153, 163);
    light.highlight = ftxui::Color::RGB(150, 80, 39);
    light.warning = ftxui::Color::RGB(143, 94, 21);
    light.error = ftxui::Color::RGB(140, 67, 81);
    light.success = ftxui::Color::RGB(51, 99, 92);
    light.dim = ftxui::Color::RGB(150, 153, 163);
    themes_[static_cast<size_t>(ThemeId::LIGHT)] = light;

    Theme hacker;
    hacker.name = "Hacker";
    hacker.bg = ftxui::Color::RGB(10, 10, 10);
    hacker.fg = ftxui::Color::RGB(0, 255, 65);
    hacker.accent1 = ftxui::Color::RGB(0, 255, 65);
    hacker.accent2 = ftxui::Color::RGB(57, 255, 20);
    hacker.accent3 = ftxui::Color::RGB(0, 230, 118);
    hacker.border_color = ftxui::Color::RGB(13, 59, 13);
    hacker.highlight = ftxui::Color::RGB(118, 255, 3);
    hacker.warning = ftxui::Color::RGB(255, 234, 0);
    hacker.error = ftxui::Color::RGB(255, 23, 68);
    hacker.success = ftxui::Color::RGB(0, 230, 118);
    hacker.dim = ftxui::Color::RGB(0, 100, 0);
    themes_[static_cast<size_t>(ThemeId::HACKER)] = hacker;
}

const Theme& ThemeEngine::currentTheme() const {
    return themes_[static_cast<size_t>(active_id_)];
}

void ThemeEngine::nextTheme() {
    int next = (static_cast<int>(active_id_) + 1) % static_cast<int>(ThemeId::COUNT_);
    active_id_ = static_cast<ThemeId>(next);
}

void ThemeEngine::setTheme(ThemeId id) {
    if (id < ThemeId::COUNT_) {
        active_id_ = id;
    }
}

const std::string& ThemeEngine::themeName() const {
    return currentTheme().name;
}

ftxui::Element ThemeEngine::styledBorder(ftxui::Element inner) const {
    return ftxui::borderLight(std::move(inner)) | ftxui::color(currentTheme().border_color) | ftxui::bgcolor(currentTheme().bg);
}

ftxui::Element ThemeEngine::focusedBorder(ftxui::Element inner) const {
    return ftxui::border(std::move(inner)) | ftxui::color(currentTheme().accent1) | ftxui::bgcolor(currentTheme().bg);
}

ftxui::Element ThemeEngine::styledText(const std::string& s) const {
    return ftxui::text(s) | ftxui::color(currentTheme().fg);
}

ftxui::Element ThemeEngine::accentText(const std::string& s) const {
    return ftxui::text(s) | ftxui::color(currentTheme().accent1);
}

ftxui::Element ThemeEngine::warningText(const std::string& s) const {
    return ftxui::text(s) | ftxui::color(currentTheme().warning);
}

ftxui::Element ThemeEngine::errorText(const std::string& s) const {
    return ftxui::text(s) | ftxui::color(currentTheme().error);
}

ftxui::Element ThemeEngine::successText(const std::string& s) const {
    return ftxui::text(s) | ftxui::color(currentTheme().success);
}

ftxui::Element ThemeEngine::dimText(const std::string& s) const {
    return ftxui::text(s) | ftxui::color(currentTheme().dim);
}

ftxui::Decorator ThemeEngine::highlightBg() const {
    return ftxui::bgcolor(currentTheme().border_color); // Using border color as a highlight bg
}

ftxui::Element ThemeEngine::withBackground(ftxui::Element inner) const {
    return std::move(inner) | ftxui::bgcolor(currentTheme().bg);
}

} // namespace inferscope

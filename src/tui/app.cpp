#include "inferscope/tui/app.hpp"
#include <iostream>
#include <chrono>

namespace inferscope {

App::App(const ModelConfig& model_config, std::vector<LayerInfo> layers, int buffer_size)
    : model_config_(model_config),
      layers_(std::move(layers)),
      buffer_size_(buffer_size),
      topology_panel_(layers_),
      screen_(ftxui::ScreenInteractive::Fullscreen()) {
    setupKeybindings();
    updatePanelFocus();
}

void App::run() {
    auto component = ftxui::Renderer([&] { return renderMainLayout(); });
    
    component |= ftxui::CatchEvent([&](ftxui::Event event) {
        if (show_help_ && event.is_character() && event.character() == "?") {
            show_help_ = false;
            return true;
        }
        
        if (keybindings_.handleEvent(event)) {
            return true;
        }
        
        std::string key;
        if (event.is_character()) {
            key = event.character();
        } else if (event == ftxui::Event::Return) {
            key = "\n";
        } else if (event == ftxui::Event::ArrowDown) {
            key = "j";
        } else if (event == ftxui::Event::ArrowUp) {
            key = "k";
        } else if (event == ftxui::Event::ArrowLeft) {
            key = "h";
        } else if (event == ftxui::Event::ArrowRight) {
            key = "l";
        }
        
        if (!key.empty()) {
            bool handled = false;
            if (active_panel_ == 0) handled = topology_panel_.handleKey(key);
            else if (active_panel_ == 1) handled = live_panel_.handleKey(key);
            else if (active_panel_ == 2) handled = attention_panel_.handleKey(key);
            else if (active_panel_ == 3) handled = metrics_panel_.handleKey(key);
            else if (active_panel_ == 4) handled = stats_panel_.handleKey(key);
            
            if (handled) {
                updateSelectedData();
                return true;
            }
        }
        
        return false;
    });
    
    screen_.Loop(component);
}

void App::updateWithRecord(const ForwardPassRecord& record) {
    auto task = [&, record]() {
        pass_count_ = record.pass_id;
        capture_state_ = record.state;
        
        if (!replay_mode_) {
            live_panel_.addRecord(record);
        }
        
        int target_id = topology_panel_.getSelectedLayerId();
        if (target_id != -1) {
            for (const auto& snap : record.layer_snapshots) {
                if (snap.layer_id == target_id) {
                    metrics_panel_.setSnapshot(snap);
                    stats_panel_.addStats(snap.activation_stats, snap.anomalies);
                    if (snap.attention_data.has_value()) {
                        attention_panel_.setAttentionData(snap.attention_data.value());
                    }
                    break;
                }
            }
        }
    };
    
    screen_.Post(task);
}

void App::setReplayMode(bool replay) {
    replay_mode_ = replay;
    if (replay) {
        capture_state_ = CaptureState::REPLAYING;
    } else {
        capture_state_ = CaptureState::IDLE;
    }
}

void App::setTraceWriter(std::shared_ptr<TraceWriter> writer) {
    trace_writer_ = writer;
}

void App::setInitialTheme(int theme_idx) {
    theme_engine_.setTheme(static_cast<ThemeId>(theme_idx));
}

void App::setReplayData(const std::vector<ForwardPassRecord>& records) {
    replay_mode_ = true;
    capture_state_ = CaptureState::REPLAYING;
    live_panel_.setRecords(records);
    pass_count_ = records.size();
    updateSelectedData();
}

void App::updateSelectedData() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    auto opt_record = live_panel_.getSelectedRecord();
    if (!opt_record.has_value()) return;
    const auto& record = opt_record.value();
    
    int target_id = topology_panel_.getSelectedLayerId();
    if (target_id != -1) {
        for (const auto& snap : record.layer_snapshots) {
            if (snap.layer_id == target_id) {
                metrics_panel_.setSnapshot(snap);
                stats_panel_.addStats(snap.activation_stats, snap.anomalies);
                if (snap.attention_data.has_value()) {
                    attention_panel_.setAttentionData(snap.attention_data.value());
                }
                break;
            }
        }
    }
}

ftxui::ScreenInteractive& App::screen() {
    return screen_;
}

void App::setupKeybindings() {
    keybindings_.registerBinding("Tab", "Cycle Focus", "Switch active panel", [&]() {
        cycleFocus();
        return true;
    });
    
    keybindings_.registerBinding("t", "Theme", "Cycle color theme", [&]() {
        theme_engine_.nextTheme();
        return true;
    });
    
    keybindings_.registerBinding("r", "Record", "Toggle trace recording", [&]() {
        if (replay_mode_) return true;
        if (trace_writer_) {
            if (trace_writer_->isOpen()) {
                trace_writer_->close();
            } else {
                std::string fname = "traces/capture_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".nstrace";
                if (trace_writer_->open(fname)) {
                    trace_writer_->writeHeader(model_config_);
                }
            }
        }
        return true;
    });
    
    keybindings_.registerBinding("?", "Help", "Toggle keybindings overlay", [&]() {
        show_help_ = !show_help_;
        return true;
    });
    
    keybindings_.registerBinding("q", "Quit", "Exit application", [&]() {
        screen_.Exit();
        return true;
    });
}

void App::cycleFocus() {
    active_panel_ = (active_panel_ + 1) % NUM_PANELS;
    updatePanelFocus();
}

void App::updatePanelFocus() {
    topology_panel_.setFocused(active_panel_ == 0);
    live_panel_.setFocused(active_panel_ == 1);
    attention_panel_.setFocused(active_panel_ == 2);
    metrics_panel_.setFocused(active_panel_ == 3);
    stats_panel_.setFocused(active_panel_ == 4);
}

ftxui::Element App::renderStatusBar() const {
    using namespace ftxui;
    const auto& t = theme_engine_.currentTheme();
    
    std::string state_str = "● IDLE";
    if (capture_state_ == CaptureState::CAPTURING) state_str = "● CAPTURING";
    else if (capture_state_ == CaptureState::PAUSED) state_str = "⏸ PAUSED";
    else if (capture_state_ == CaptureState::REPLAYING) state_str = "▶ REPLAYING";
    
    std::string rec_str = "";
    if (trace_writer_ && trace_writer_->isOpen()) {
        rec_str = " [REC: " + std::to_string(trace_writer_->numRecords()) + " passes]";
    }
    
    return hbox({
        text(" inferscope ") | bold | color(t.bg) | bgcolor(t.accent1),
        text(" │ ") | color(t.border_color),
        text(model_config_.name) | color(t.fg),
        text(" │ Pass: ") | color(t.border_color),
        text(std::to_string(pass_count_)) | color(t.fg),
        text(" │ ") | color(t.border_color),
        text(state_str) | color(capture_state_ == CaptureState::CAPTURING ? t.error : t.success) | bold,
        text(rec_str) | color(t.error) | bold,
        filler(),
        text("Theme: " + t.name + " ") | color(t.fg)
    }) | bgcolor(t.border_color);
}

ftxui::Element App::renderCommandBar() const {
    using namespace ftxui;
    const auto& t = theme_engine_.currentTheme();
    return hbox({
        text(" [Tab] ") | color(t.accent1) | bold, text("Cycle Focus  "),
        text("[j/k] ") | color(t.accent1) | bold, text("Navigate  "),
        text("[Space/Enter] ") | color(t.accent1) | bold, text("Select  "),
        text("[t] ") | color(t.accent1) | bold, text("Theme  "),
        text("[r] ") | color(t.accent1) | bold, text("Record  "),
        text("[?] ") | color(t.accent1) | bold, text("Help  "),
        text("[q] ") | color(t.accent1) | bold, text("Quit")
    }) | center | bgcolor(t.bg) | color(t.fg);
}

ftxui::Element App::renderHelpOverlay() const {
    return keybindings_.renderHelpOverlay(theme_engine_);
}

ftxui::Element App::renderMainLayout() const {
    using namespace ftxui;
    
    auto top_row = hbox({
        topology_panel_.render(theme_engine_) | size(WIDTH, EQUAL, 45),
        live_panel_.render(theme_engine_) | flex
    }) | flex;
    
    auto bottom_row = hbox({
        attention_panel_.render(theme_engine_) | size(WIDTH, EQUAL, 45),
        metrics_panel_.render(theme_engine_) | size(WIDTH, EQUAL, 35),
        stats_panel_.render(theme_engine_) | flex
    }) | size(HEIGHT, EQUAL, 15);
    
    auto main_layout = vbox({
        renderStatusBar(),
        top_row,
        bottom_row,
        renderCommandBar()
    }) | bgcolor(theme_engine_.currentTheme().bg);
    
    if (show_help_) {
        return dbox({
            main_layout,
            renderHelpOverlay()
        });
    }
    
    return main_layout;
}

} // namespace inferscope

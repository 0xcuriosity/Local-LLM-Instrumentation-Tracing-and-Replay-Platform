// ==========================================================================
// inferscope — Local LLM Instrumentation, Tracing & Replay Platform
// ==========================================================================
// Main entry point. Initializes the simulated transformer model, sets up
// instrumentation hooks, and launches the interactive TUI.
// ==========================================================================

#include "inferscope/core/types.hpp"
#include "inferscope/core/config.hpp"
#include "inferscope/core/ring_buffer.hpp"
#include "inferscope/model/transformer.hpp"
#include "inferscope/instrument/hook_manager.hpp"
#include "inferscope/instrument/layer_probe.hpp"
#include "inferscope/trace/trace_writer.hpp"
#include "inferscope/trace/trace_reader.hpp"
#include "inferscope/tui/app.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// ==========================================================================
// Command-line argument parsing
// ==========================================================================

struct CLIArgs {
    std::string model_name = "llama-3-8b";
    int num_layers = 32;
    int hidden_dim = 4096;
    int num_heads = 32;
    int seq_len = inferscope::config::SIM_DEFAULT_SEQ_LEN;
    int buffer_size = inferscope::config::DEFAULT_RING_BUFFER_SIZE;
    std::string trace_file;       // Replay mode if set
    std::string theme = "dark";   // Initial theme
    bool help = false;
    int sim_interval_ms = inferscope::config::SIM_FORWARD_PASS_INTERVAL_MS;
};

void printUsage(const char* prog) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  inferscope v" << inferscope::config::APP_VERSION << "                                         ║\n";
    std::cout << "║  Local LLM Instrumentation, Tracing & Replay Platform      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model NAME        Model name (default: llama-3-8b)\n";
    std::cout << "  --layers N          Number of transformer layers (default: 32)\n";
    std::cout << "  --hidden N          Hidden dimension (default: 4096)\n";
    std::cout << "  --heads N           Number of attention heads (default: 32)\n";
    std::cout << "  --seq-len N         Sequence length for simulation (default: 8)\n";
    std::cout << "  --buffer-size N     Ring buffer capacity (default: 512)\n";
    std::cout << "  --trace FILE        Replay a trace file (.nstrace)\n";
    std::cout << "  --theme THEME       Initial theme: dark, light, hacker (default: dark)\n";
    std::cout << "  --interval MS       Simulation interval in ms (default: 500)\n";
    std::cout << "  --help              Show this help message\n";
    std::cout << "\n";
    std::cout << "Keybindings:\n";
    std::cout << "  Tab        Cycle focus between panels\n";
    std::cout << "  j/k        Navigate up/down in focused panel\n";
    std::cout << "  h/l        Navigate left/right (attention heads)\n";
    std::cout << "  Space      Expand/collapse tree nodes\n";
    std::cout << "  Enter      Set active capture target\n";
    std::cout << "  t          Cycle theme (Dark → Light → Hacker)\n";
    std::cout << "  r          Start/stop recording trace\n";
    std::cout << "  p          Toggle replay mode\n";
    std::cout << "  s          Save current trace\n";
    std::cout << "  ?          Toggle help overlay\n";
    std::cout << "  q          Quit\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << "                             # Run with defaults\n";
    std::cout << "  " << prog << " --layers 12 --heads 12      # Smaller model\n";
    std::cout << "  " << prog << " --trace traces/run1.nstrace  # Replay a trace\n";
    std::cout << "  " << prog << " --theme hacker --interval 200 # Fast hacker mode\n";
    std::cout << "\n";
}

CLIArgs parseArgs(int argc, char* argv[]) {
    CLIArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--model" && i + 1 < argc) {
            args.model_name = argv[++i];
        } else if (arg == "--layers" && i + 1 < argc) {
            args.num_layers = std::stoi(argv[++i]);
        } else if (arg == "--hidden" && i + 1 < argc) {
            args.hidden_dim = std::stoi(argv[++i]);
        } else if (arg == "--heads" && i + 1 < argc) {
            args.num_heads = std::stoi(argv[++i]);
        } else if (arg == "--seq-len" && i + 1 < argc) {
            args.seq_len = std::stoi(argv[++i]);
        } else if (arg == "--buffer-size" && i + 1 < argc) {
            args.buffer_size = std::stoi(argv[++i]);
        } else if (arg == "--trace" && i + 1 < argc) {
            args.trace_file = argv[++i];
        } else if (arg == "--theme" && i + 1 < argc) {
            args.theme = argv[++i];
        } else if (arg == "--interval" && i + 1 < argc) {
            args.sim_interval_ms = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }
    return args;
}

// ==========================================================================
// Signal handling
// ==========================================================================
static std::atomic<bool> g_running{true};

void signalHandler(int sig) {
    (void)sig;
    g_running.store(false);
}

// ==========================================================================
// Main
// ==========================================================================
int main(int argc, char* argv[]) {
    // Parse arguments
    CLIArgs args = parseArgs(argc, argv);
    if (args.help) {
        printUsage(argv[0]);
        return 0;
    }

    // Setup signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Create traces directory
    fs::create_directories("traces");

    // ======================================================================
    // Configure model
    // ======================================================================
    inferscope::ModelConfig model_config;
    model_config.name = args.model_name;
    model_config.num_layers = args.num_layers;
    model_config.hidden_dim = args.hidden_dim;
    model_config.num_heads = args.num_heads;
    model_config.head_dim = args.hidden_dim / args.num_heads;
    model_config.intermediate_dim = static_cast<int>(args.hidden_dim * 2.6875); // llama ratio
    model_config.max_seq_len = 2048;
    model_config.vocab_size = 128256;

    // ======================================================================
    // Replay mode
    // ======================================================================
    if (!args.trace_file.empty()) {
        if (!fs::exists(args.trace_file)) {
            std::cerr << "Error: Trace file not found: " << args.trace_file << "\n";
            return 1;
        }

        inferscope::TraceReader reader;
        if (!reader.open(args.trace_file)) {
            std::cerr << "Error: Failed to open trace file: " << args.trace_file << "\n";
            return 1;
        }

        auto header = reader.readHeader();
        model_config = header.model_config;
        auto records = reader.readAll();
        reader.close();

        // Build topology from config
        inferscope::TransformerModel model(model_config);
        auto topology = model.buildTopology();

        // Launch TUI in replay mode
        inferscope::App app(model_config, topology, args.buffer_size);
        
        // Set initial theme
        if (args.theme == "light") { app.setInitialTheme(1); }
        else if (args.theme == "hacker") { app.setInitialTheme(2); }
        
        app.setReplayData(records);
        app.run();
        return 0;
    }

    // ======================================================================
    // Live capture mode
    // ======================================================================
    inferscope::TransformerModel model(model_config);
    auto topology = model.buildTopology();
    inferscope::HookManager hook_manager;
    hook_manager.setEnabled(true);

    // Create ring buffer for forward pass records
    auto pass_buffer = std::make_shared<inferscope::RingBuffer<inferscope::ForwardPassRecord>>(args.buffer_size);

    // ======================================================================
    // Setup instrumentation hooks
    // ======================================================================
    // Global post-hook: collects all layer data into current pass record
    std::mutex record_mutex;
    auto current_record = std::make_shared<inferscope::ForwardPassRecord>();
    std::atomic<uint64_t> pass_counter{0};

    hook_manager.registerGlobalPostHook(
        [&](int layer_id, const std::string& layer_name,
            const inferscope::ActivationStats& stats,
            const inferscope::TensorMeta& meta,
            const inferscope::LayerLatency& latency,
            const std::optional<inferscope::AttentionData>& attn) {
            
            inferscope::LayerSnapshot snapshot;
            snapshot.layer_id = layer_id;
            snapshot.layer_name = layer_name;
            snapshot.latency = latency;
            snapshot.activation_stats = stats;
            snapshot.tensor_meta = meta;
            snapshot.attention_data = attn;
            snapshot.timestamp = inferscope::Clock::now();

            // Detect anomalies
            snapshot.anomalies = inferscope::LayerProbe::detectAnomalies(stats);

            std::lock_guard<std::mutex> lock(record_mutex);
            current_record->layer_snapshots.push_back(std::move(snapshot));
        }
    );

    // ======================================================================
    // Launch TUI
    // ======================================================================
    inferscope::App app(model_config, topology, args.buffer_size);
    
    // Set initial theme
    if (args.theme == "light") { app.setInitialTheme(1); }
    else if (args.theme == "hacker") { app.setInitialTheme(2); }

    // ======================================================================
    // Background simulation thread
    // ======================================================================
    std::atomic<bool> sim_running{true};
    
    // Trace writer (optional recording)
    auto trace_writer = std::make_shared<inferscope::TraceWriter>();
    app.setTraceWriter(trace_writer);

    std::thread sim_thread([&]() {
        while (sim_running.load() && g_running.load()) {
            // Simulate a forward pass
            {
                std::lock_guard<std::mutex> lock(record_mutex);
                current_record = std::make_shared<inferscope::ForwardPassRecord>();
                current_record->pass_id = pass_counter.fetch_add(1);
                current_record->timestamp = inferscope::Clock::now();
                current_record->token_count = args.seq_len;
                current_record->state = inferscope::CaptureState::CAPTURING;
            }

            // Run simulated forward pass (this fires hooks)
            model.simulateForwardPass(args.seq_len, hook_manager);

            // Finalize record
            inferscope::ForwardPassRecord completed_record;
            {
                std::lock_guard<std::mutex> lock(record_mutex);
                current_record->state = inferscope::CaptureState::IDLE;
                // Compute total latency
                double total = 0;
                for (const auto& snap : current_record->layer_snapshots) {
                    total += snap.latency.delta_ms;
                }
                current_record->total_latency_ms = total;
                completed_record = *current_record;
            }

            // Push to ring buffer and update TUI
            pass_buffer->push(completed_record);
            app.updateWithRecord(completed_record);

            // Write to trace if recording
            if (trace_writer->isOpen()) {
                trace_writer->writeRecord(completed_record);
            }

            // Sleep between passes
            std::this_thread::sleep_for(
                std::chrono::milliseconds(args.sim_interval_ms)
            );
        }
    });

    // Run the TUI (blocking)
    app.run();

    // Cleanup
    sim_running.store(false);
    g_running.store(false);
    if (sim_thread.joinable()) {
        sim_thread.join();
    }

    // Close any open trace
    if (trace_writer->isOpen()) {
        trace_writer->close();
    }

    std::cout << "\ninferscope terminated. " 
              << pass_counter.load() << " forward passes captured.\n";
    return 0;
}

# InferScope

<div align="center">

```text
╔══════════════════════════════════════════════════════════════╗
║            InferScope v1.0.0                                ║
║    Local LLM Instrumentation, Tracing & Replay Platform     ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Building from Source](#building-from-source)
- [Usage](#usage)
- [TUI Interface](#tui-interface)
- [Keybindings](#keybindings)
- [Theme System](#theme-system)
- [Data Storage & Ring Buffers](#data-storage--ring-buffers)
- [Trace File Format](#trace-file-format)
- [Non-Invasive Hooking](#non-invasive-hooking)
- [Integration Guide](#integration-guide)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Ever wanted to inspect what happens inside a Transformer during inference? **InferScope** is a lightweight instrumentation, tracing, and diagnostics platform for local LLMs.

The system:
1. **Hooks non-invasively** into a model's execution pipeline without modifying its source code.
2. **Captures runtime information** such as tensor shapes, activation statistics, and layer latencies as tokens propagate through the network.
3. **Visualizes the collected data** through an interactive terminal interface with Vim-inspired navigation.

### What It Does

- Intercepts forward-pass execution to collect metadata and activation statistics from embeddings, attention blocks, and MLP layers without requiring changes to the model implementation.
- Measures per-layer execution latency to identify computational bottlenecks.
- Displays attention matrices as real-time Unicode heatmaps.
- Monitors runtime metrics including tensor shapes, sparsity, timestamps, and memory usage.
- Detects numerical anomalies such as NaNs, Infs, clipping risks, excessive sparsity, and dead neurons.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **Non-Invasive Hooking** | Callback-based + LD_PRELOAD interception — zero model modification |
| 🌳 **Model Topology Explorer** | Interactive tree view with expand/collapse navigation |
| ⚡ **Live Forward Pass Tracking** | Real-time timestamped feed with color-coded latency indicators |
| 🔥 **Attention Heatmap** | Unicode block-character heatmap with per-head navigation |
| 📊 **Runtime Metrics Inspector** | Tensor shapes, sparsity bars, latency bounds, memory footprint |
| 📈 **Numerical Statistics** | Sparkline time-series graphs for mean, std, min, max, L2 norm |
| 🚨 **Anomaly Detection** | Automatic flagging of NaN, Inf, clipping risks, outlier activations |
| 💾 **Trace Recording & Replay** | Binary `.istrace` format for offline analysis with playback controls |
| 🎨 **Theme Engine** | Dark, Light, and Hacker themes — cycle with a single keypress |
| ⌨️ **Vim-like Navigation** | j/k, h/l, Tab, Space, Enter — keyboard-driven workflow |
| 🔄 **Ring Buffer Storage** | Fixed-size memory pools — RAM never grows unbounded |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     InferScope Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Instrumentation Layer                   │       │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐   │       │
│  │  │ Hook Manager │  │ Callback Reg │  │Layer Probes│  │       │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘   │       │
│  │         │                │                │          │       │
│  │  ┌──────┴────────────────┴────────────────┘          │       │
│  │  │ LD_PRELOAD Interceptor (libis_preload.so)         │       │
│  │  └───────────────────────────────────────────────────┘       │
│  └──────────────────────────┬────────────────────────────┘       │
│                             │                                    │
│  ┌──────────────────────────┴────────────────────────────┐       │
│  │              Data Collection Layer                    │       │
│  │  ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐    │       │
│  │  │Ring Buffer│ │Act. Stats│ │ Latency │ │Attn Mat│    │       │
│  │  │   Pool    │ │Collector │ │Profiler │ │Capture │    │       │
│  │  └─────┬─────┘ └────┬─────┘ └───┬─────┘ └───┬────┘    │       │
│  └────────┼─────────────┼───────────┼───────────┼─────────┘       │
│           │             │           │           │                 │
│  ┌────────┴─────────────┴───────────┴───────────┴─────────┐       │
│  │               TUI Presentation Layer (FTXUI)           │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐      │
│  │                Storage & Replay                        │      │
│  │ Trace Writer ───────────────▶ .istrace File            │      │
│  │ Trace Reader ◀───────────────┘                         │      │
│  └────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```


## Tech Stack

| Component           | Technology                    | Purpose                                                                            |
| ------------------- | ----------------------------- | ---------------------------------------------------------------------------------- |
| **Language**        | C++17                         | Primary implementation language                                                    |
| **Build System**    | CMake 3.16+                   | Cross-platform build configuration and dependency management                       |
| **TUI Framework**   | FTXUI v5.0                    | Modern C++ terminal UI with a declarative API and no external runtime dependencies |
| **Concurrency**     | `std::thread`, `std::mutex`   | Thread-safe data collection and rendering                                          |
| **Storage**         | In-memory Ring Buffers        | Fixed-capacity storage with bounded memory usage                                   |
| **Instrumentation** | Callback Hooks + `LD_PRELOAD` | Non-invasive runtime interception                                                  |
| **Serialization**   | Custom Binary Format          | Efficient storage and replay of captured traces                                    |

> **Why no database?**
> InferScope is designed around fixed-size ring buffers to guarantee predictable memory usage during execution. Runtime data is retained entirely in memory while tracing is active, eliminating disk I/O from the capture path. Sessions can be persisted as compact binary `.istrace` files for later replay and analysis.

---

## Install Prerequisites

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install build-essential cmake git
```

### Fedora

```bash
sudo dnf install gcc-c++ cmake git
```

### Arch Linux

```bash
sudo pacman -S base-devel cmake git
```

---

## Building from Source

```bash
# Clone the repository or navigate to the project directory
cd InferScope

# Create an out-of-source build directory
mkdir -p build && cd build

# Configure the project (FTXUI is fetched automatically)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile using all available CPU cores
make -j$(nproc)
```

The build generates the following artifacts:

* `inferscope` — Main terminal application
* `libis_preload.so` — Shared library for `LD_PRELOAD`-based instrumentation

---

## Usage

### Run the Built-in Simulator

```bash
./inferscope
```

Starts InferScope with a simulated **Llama-3-8B** configuration (32 transformer layers, 4096 hidden dimension, 32 attention heads).

---

### Launch with a Custom Model Configuration

```bash
./inferscope --model "my-model" --layers 12 --hidden 768 --heads 12
```

---

### Replay a Recorded Trace

```bash
./inferscope --trace traces/captured_run.istrace
```

---

### Command-Line Options

```text
Usage: inferscope [OPTIONS]

Options:
  --model NAME        Model name (default: llama-3-8b)
  --layers N          Number of transformer layers (default: 32)
  --hidden N          Hidden dimension (default: 4096)
  --heads N           Number of attention heads (default: 32)
  --seq-len N         Sequence length for simulation (default: 8)
  --buffer-size N     Ring buffer capacity (default: 512)
  --trace FILE        Replay a trace file (.istrace)
  --theme THEME       Initial theme: dark, light, hacker (default: dark)
  --interval MS       Simulation interval in milliseconds (default: 500)
  --help              Display the help message
```

---

### Instrument External Model Binaries

```bash
# Inject InferScope into any C/C++ model executable
LD_PRELOAD=./libis_preload.so ./your_model_binary
```

Example using **llama.cpp**:

```bash
LD_PRELOAD=./libis_preload.so ./llama-cli -m model.gguf -p "Hello"
```

The preload library transparently intercepts supported runtime functions and streams execution data into InferScope without requiring modifications to the target application's source code.

---

## TUI Interface

InferScope presents runtime information through a five-panel terminal interface arranged in a responsive grid layout.

<img width="1600" height="856" alt="tui interface" src="https://github.com/user-attachments/assets/7e68a6df-f66a-43e9-910a-c97dad67e676" />

### Panel Overview

| #     | Panel                           | Description                                                                                                                                                     |
| ----- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Model Topology**              | Interactive tree view of the Transformer architecture. Navigate with `j`/`k`, expand or collapse nodes with `Space`, and select a capture target using `Enter`. |
| **2** | **Live Forward Passes**         | Continuously updated timeline of forward-pass events with latency-based color coding (green < 50 ms, yellow 50–100 ms, red > 100 ms).                           |
| **3** | **Attention Matrix Visualizer** | Unicode heatmap displaying attention weights. Switch between attention heads using `h` and `l` to inspect token-to-token interactions.                          |
| **4** | **Runtime Metrics Inspector**   | Displays tensor shape, data type, sparsity visualization, latency measurements, and estimated memory usage for the selected layer.                              |
| **5** | **Numerical Statistics**        | Real-time sparklines for mean, standard deviation, minimum, maximum, and L2 norm, along with highlighted anomaly indicators.                                    |

---

## Keybindings

Every keybinding in InferScope performs a functional action, enabling a completely keyboard-driven workflow.

### Navigation

| Key     | Action                                | Scope                 |
| ------- | ------------------------------------- | --------------------- |
| `Tab`   | Move focus to the next panel          | Global                |
| `j`     | Move selection down                   | Active panel          |
| `k`     | Move selection up                     | Active panel          |
| `h`     | Previous attention head               | Attention panel       |
| `l`     | Next attention head                   | Attention panel       |
| `Space` | Expand or collapse the selected node  | Topology panel        |
| `Enter` | Select capture target or forward pass | Topology / Live panel |

### Actions

| Key | Action           | Description                                       |
| --- | ---------------- | ------------------------------------------------- |
| `t` | Cycle theme      | Dark → Light → Hacker → Dark                      |
| `r` | Toggle recording | Start or stop trace recording                     |
| `p` | Toggle replay    | Switch between live capture and replay mode       |
| `s` | Save trace       | Write the current recording to an `.istrace` file |
| `?` | Toggle help      | Show or hide the keybinding reference             |
| `q` | Quit             | Exit InferScope                                   |

---

## Theme System

InferScope ships with three built-in themes tailored for different working environments.

### 🌙 Dark Theme (Default)

* Deep charcoal background with blue, green, and amber accents
* Optimized for extended use in low-light environments
* Inspired by the Tokyo Night color palette

### ☀️ Light Theme

* Clean light background with navy, teal, and coral highlights
* High-contrast design for bright environments
* Reduced eye strain during daytime usage

### 💚 Hacker Theme

* Pure black background with monochrome matrix-green styling
* Maximum contrast with minimal visual distractions
* Inspired by classic terminal aesthetics

Press **`t`** at any time to cycle between available themes. Theme changes are applied instantly across all panels, borders, status indicators, and text elements.

---

## Data Storage & Ring Buffers

### Designed Without a Database

InferScope intentionally avoids using a database. Instead, runtime information is maintained entirely in memory using **fixed-size ring buffers (circular buffers)**. This approach provides predictable resource usage while keeping data capture lightweight.

Key benefits include:

1. **Bounded Memory Usage** – Ring buffers have a fixed capacity (512 entries by default). Once full, new records overwrite the oldest ones, preventing unbounded memory growth.
2. **Low-Latency Capture** – All operations occur in memory with **O(1)** insertion and retrieval, eliminating disk I/O during tracing.
3. **Thread Safety** – Mutex-protected access allows the simulation thread and the TUI rendering thread to safely share data.

### Ring Buffer Configuration

| Parameter            | Default | Description                                     |
| -------------------- | ------- | ----------------------------------------------- |
| Capacity             | 512     | Maximum number of forward-pass records retained |
| Memory per Record    | ~2–8 KB | Depends on model size and capture detail        |
| Maximum Memory Usage | ~4 MB   | Upper bound for ring buffer storage             |

```cpp
// Ring buffer template class
RingBuffer<ForwardPassRecord> buffer(512);  // Stores the latest 512 forward passes

buffer.push(record);   // O(1) insertion (overwrites oldest when full)
buffer.at(0);          // Access the oldest record
buffer.back();         // Access the most recent record
```

### Trace Persistence

For long-term storage, InferScope can record execution sessions as binary **`.istrace`** files (see the **Trace File Format** section). Recording is toggled with the **`r`** key, while the current session can be written to disk using **`s`**.

---

## Trace File Format

### `.istrace` Binary Format

InferScope stores recorded sessions using a compact binary format designed for efficient storage and high-speed sequential replay.

```text
┌─────────────────────────────────────────────┐
│                 FILE HEADER                  │
├─────────────────────────────────────────────┤
│ Magic:    "NSTRACE\0"  (8 bytes)            │
│ Version:  uint32       (4 bytes)            │
│ Records:  uint32       (4 bytes)            │
│ Model Config:          (variable)           │
│   - name_len: uint16                        │
│   - name: chars                             │
│   - num_layers: int32                       │
│   - hidden_dim: int32                       │
│   - num_heads: int32                        │
│   - vocab_size: int32                       │
├─────────────────────────────────────────────┤
│               RECORD 0                      │
├─────────────────────────────────────────────┤
│ record_size:    uint32                      │
│ pass_id:        uint64                      │
│ timestamp:      int64 (epoch ms)            │
│ total_latency:  double                      │
│ token_count:    int32                       │
│ num_snapshots:  uint32                      │
│ ┌─────────────────────────────────────────┐ │
│ │           SNAPSHOT 0                    │ │
│ │ layer_id:     int32                     │ │
│ │ name_len:     uint16                    │ │
│ │ name:         chars                     │ │
│ │ type:         uint8                     │ │
│ │ latency_ms:   double                    │ │
│ │ stats: {mean, std, min, max,            │ │
│ │         sparsity, l2_norm, ...}         │ │
│ │ shape_ndims:  uint8                     │ │
│ │ dims:         int64[]                   │ │
│ │ has_attention: bool                     │ │
│ │ [attention_data if present]             │ │
│ │ anomalies:    uint16                    │ │
│ └─────────────────────────────────────────┘ │
│ │           SNAPSHOT 1 ...                │ │
├─────────────────────────────────────────────┤
│               RECORD 1 ...                  │
└─────────────────────────────────────────────┘
```

### Trace File Size Limits

* **Maximum trace size:** 512 MB
* **Typical storage usage:** 10–100 MB per hour of capture
* **Automatic file rotation** when the configured size limit is reached

---

# Non-Invasive Instrumentation

InferScope offers two approaches for instrumenting Transformer models without requiring modifications to their source code.

## 1. Callback-Based Hooks (Recommended)

The `HookManager` exposes a lightweight callback interface for observing model execution.

```cpp
#include "inferscope/instrument/hook_manager.hpp"

inferscope::HookManager hooks;

// Invoked before a layer executes
hooks.registerPreHook(layer_id, [](int id, const std::string& name) {
    std::cout << "Entering layer: " << name << std::endl;
});

// Invoked after execution with captured runtime data
hooks.registerPostHook(layer_id,
[](int id,
   const std::string& name,
   const ActivationStats& stats,
   const TensorMeta& meta,
   const LayerLatency& latency,
   const std::optional<AttentionData>& attn) {

    std::cout << "Layer " << name
              << " latency: "
              << latency.delta_ms << " ms\n";

    std::cout << "  Mean activation: "
              << stats.mean << "\n";

    std::cout << "  Sparsity: "
              << stats.sparsity_rate * 100 << "%\n";
});
```

---

## 2. `LD_PRELOAD` Instrumentation

When source code isn't available, InferScope can instrument existing binaries through an `LD_PRELOAD` shared library.

```bash
# Build the preload library
cmake --build build --target is_preload

# Instrument an external model executable
LD_PRELOAD=./build/libis_preload.so ./your_model_binary
```

The preload library intercepts supported tensor operations at runtime and forwards the collected telemetry to InferScope.

---

# Integration Guide

## Integrating with llama.cpp

InferScope integrates with `llama.cpp` using its evaluation callback interface.

```cpp
// Configure llama.cpp callbacks
struct llama_context_params params = llama_context_default_params();

params.cb_eval = [](struct ggml_tensor* t,
                    bool ask,
                    void* user_data) -> bool {

    auto* hooks =
        static_cast<inferscope::HookManager*>(user_data);

    if (ask) {
        return true;    // Capture every tensor
    }

    // Extract tensor metadata and invoke callbacks
    // ... see examples/

    return true;
};

params.cb_eval_user_data = &hook_manager;
```

---

## Integrating with Custom C++ Models

For custom inference engines, wrap individual layers with a `LayerProbe`.

```cpp
inferscope::LayerProbe probe(
    layer_id,
    "encoder.layer.0.attention"
);

probe.beginCapture();

auto output = my_attention_layer.forward(input);

probe.endCapture(
    output.data(),
    output.size(),
    output.shape()
);

// Retrieve captured information
auto stats      = probe.getStats();
auto latency    = probe.getLatency();
auto anomalies  =
    inferscope::LayerProbe::detectAnomalies(stats);
```

---

# Configuration

Configuration values are defined in:

```text
include/inferscope/core/config.hpp
```

## Ring Buffer

| Constant                   | Default | Description                            |
| -------------------------- | ------- | -------------------------------------- |
| `MAX_RING_BUFFER_SIZE`     | 1024    | Maximum supported ring buffer capacity |
| `DEFAULT_RING_BUFFER_SIZE` | 512     | Default buffer capacity                |

---

## TUI

| Constant              | Default | Description                               |
| --------------------- | ------- | ----------------------------------------- |
| `REFRESH_RATE_MS`     | 50      | Refresh interval (20 FPS)                 |
| `SPARKLINE_HISTORY`   | 60      | Number of samples retained for sparklines |
| `MIN_TERMINAL_WIDTH`  | 120     | Minimum supported terminal width          |
| `MIN_TERMINAL_HEIGHT` | 35      | Minimum supported terminal height         |

---

## Anomaly Detection

| Constant                     | Default | Description                                |
| ---------------------------- | ------- | ------------------------------------------ |
| `ANOMALY_SPARSITY_THRESHOLD` | 0.95    | Flag layers with sparsity greater than 95% |
| `ANOMALY_ACTIVATION_MAX`     | `1e6`   | Threshold for unusually large activations  |
| `LATENCY_WARNING_MS`         | 50.0    | Warning latency threshold                  |
| `LATENCY_CRITICAL_MS`        | 100.0   | Critical latency threshold                 |

---

# Project Structure

```text
InferScope/
├── CMakeLists.txt
├── README.md
├── include/
│   └── inferscope/
│       ├── core/
│       │   ├── ring_buffer.hpp
│       │   ├── types.hpp
│       │   └── config.hpp
│       ├── instrument/
│       │   ├── hook_manager.hpp
│       │   ├── layer_probe.hpp
│       │   └── preload_hook.hpp
│       ├── model/
│       │   ├── transformer.hpp
│       │   └── tensor.hpp
│       ├── trace/
│       │   ├── trace_writer.hpp
│       │   └── trace_reader.hpp
│       └── tui/
│           ├── app.hpp
│           ├── theme.hpp
│           ├── keybindings.hpp
│           └── panels/
│               ├── topology_panel.hpp
│               ├── live_panel.hpp
│               ├── attention_panel.hpp
│               ├── metrics_panel.hpp
│               └── stats_panel.hpp
├── src/
│   ├── main.cpp
│   ├── core/
│   ├── instrument/
│   │   ├── hook_manager.cpp
│   │   ├── layer_probe.cpp
│   │   └── preload_hook.cpp
│   ├── model/
│   │   ├── tensor.cpp
│   │   └── transformer.cpp
│   ├── trace/
│   │   ├── trace_writer.cpp
│   │   └── trace_reader.cpp
│   └── tui/
│       ├── app.cpp
│       ├── theme.cpp
│       ├── keybindings.cpp
│       └── panels/
│           ├── topology_panel.cpp
│           ├── live_panel.cpp
│           ├── attention_panel.cpp
│           ├── metrics_panel.cpp
│           └── stats_panel.cpp
└── traces/
```

---

# Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/my-feature
```

3. Implement your changes and add any relevant tests.
4. Verify that the project builds successfully.

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

5. Open a pull request describing your changes.

---

## Code Style

* C++17
* 4-space indentation
* `snake_case` for variables and functions
* `PascalCase` for classes and types
* Header guards implemented with `#pragma once`

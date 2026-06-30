# Local-LLM-Instrumentation-Tracing-and-Replay-Platform


transformer-tui/
├── CMakeLists.txt
├── third_party/                  # via FetchContent, not committed
├── src/
│   ├── main.cpp                  # entry point, ties everything together (empty/minimal for now)
│   ├── inference/
│   │   ├── model_runner.hpp      # wraps llama.cpp load + generate
│   │   └── model_runner.cpp
│   ├── telemetry/
│   │   ├── ring_buffer.hpp       # Day 3, stub for now
│   │   ├── hook.hpp              # Day 2, stub for now
│   │   └── hook.cpp
│   ├── tui/
│   │   ├── app.hpp               # FTXUI app wrapper
│   │   └── app.cpp
│   └── common/
│       └── types.hpp             # shared structs (LayerEvent etc.)
├── models/
│   └── .gitkeep                  # GGUF file goes here, gitignored
├── tests/
├── .gitignore
└── README.md
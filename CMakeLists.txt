cmake_minimum_required(VERSION 3.16)
project(inferscope
    VERSION 1.0.0
    DESCRIPTION "Local LLM Instrumentation, Tracing & Replay Platform"
    LANGUAGES CXX
)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Build type defaults
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# ============================================================================
# Fetch FTXUI
# ============================================================================
include(FetchContent)

FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG v5.0.0
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(ftxui)
if(NOT ftxui_POPULATED)
    FetchContent_Populate(ftxui)
    set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${ftxui_SOURCE_DIR} ${ftxui_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ============================================================================
# inferscope Core Library
# ============================================================================
add_library(inferscope_core STATIC
    src/model/tensor.cpp
    src/model/transformer.cpp
    src/instrument/hook_manager.cpp
    src/instrument/layer_probe.cpp
    src/trace/trace_writer.cpp
    src/trace/trace_reader.cpp
)

target_include_directories(inferscope_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(inferscope_core PUBLIC
    pthread
)

# ============================================================================
# inferscope TUI Library
# ============================================================================
add_library(inferscope_tui STATIC
    src/tui/theme.cpp
    src/tui/keybindings.cpp
    src/tui/panels/topology_panel.cpp
    src/tui/panels/live_panel.cpp
    src/tui/panels/attention_panel.cpp
    src/tui/panels/metrics_panel.cpp
    src/tui/panels/stats_panel.cpp
    src/tui/app.cpp
)

target_include_directories(inferscope_tui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(inferscope_tui PUBLIC
    inferscope_core
    ftxui::screen
    ftxui::dom
    ftxui::component
)

# ============================================================================
# Main Executable
# ============================================================================
add_executable(inferscope
    src/main.cpp
)

target_link_libraries(inferscope PRIVATE
    inferscope_tui
    inferscope_core
)

# ============================================================================
# LD_PRELOAD Hook Library (optional)
# ============================================================================
add_library(ns_preload SHARED
    src/instrument/preload_hook.cpp
)

target_include_directories(ns_preload PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ns_preload PRIVATE
    dl
)

set_target_properties(ns_preload PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "ns_preload"
)

# ============================================================================
# Create traces directory
# ============================================================================
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/traces)

# ============================================================================
# Install
# ============================================================================
install(TARGETS inferscope DESTINATION bin)
install(TARGETS ns_preload DESTINATION lib)

# ============================================================================
# CPack Packaging
# ============================================================================
set(CPACK_PACKAGE_NAME "inferscope")
set(CPACK_PACKAGE_VENDOR "inferscope Authors")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Local LLM Instrumentation, Tracing & Replay Platform")
set(CPACK_PACKAGE_CONTACT "support@inferscope.io")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/packages")

# Debian settings
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

include(CPack)

# ============================================================================
# Print build info
# ============================================================================
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════╗")
message(STATUS "║         inferscope v${PROJECT_VERSION}                 ║")
message(STATUS "║  Local LLM Instrumentation & Tracing        ║")
message(STATUS "╚══════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ standard:   C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

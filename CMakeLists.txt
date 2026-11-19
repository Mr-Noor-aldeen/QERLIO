cmake_minimum_required(VERSION 3.22)
project(qerlio VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

# Core include directories
include_directories(
    include
    core
    backends
)

# QERLIO Core Library
add_library(qerlio_core STATIC
    src/qerlio_abi.cpp
    src/engine.cpp
    src/model.cpp
    src/session.cpp
    core/model_registry/gguf_parser.cpp
)

target_include_directories(qerlio_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optional shared library wrapper for foreign language bindings (JNI / CFFI)
add_library(qerlio SHARED
    src/qerlio_abi.cpp
    src/engine.cpp
    src/model.cpp
    src/session.cpp
    core/model_registry/gguf_parser.cpp
)

target_compile_definitions(qerlio PRIVATE QERLIO_BUILD_SHARED)
target_include_directories(qerlio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install targets
include(GNUInstallDirs)
install(TARGETS qerlio qerlio_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/qerlio DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

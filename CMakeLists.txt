cmake_minimum_required(VERSION 3.22)
project(qerlio VERSION 0.3.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
)

set(QERLIO_SOURCES
    src/qerlio_abi.cpp
    core/model_registry/gguf_parser.cpp
    core/memory/memory_planner.cpp
    core/scheduler/adaptive_scheduler.cpp
    core/compression/universal_quant.cpp
)

add_library(qerlio_core STATIC ${QERLIO_SOURCES})

target_include_directories(qerlio_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(qerlio SHARED ${QERLIO_SOURCES})

target_compile_definitions(qerlio PRIVATE QERLIO_BUILD_SHARED)
target_include_directories(qerlio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(GNUInstallDirs)
install(TARGETS qerlio qerlio_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/qerlio DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

option(QERLIO_BUILD_TESTS "Build C++20 memory-safety test suite" ON)
if(QERLIO_BUILD_TESTS)
    enable_testing()
    add_executable(qerlio_cpp_safety_test tests/cpp_safety_test.cpp)
    target_link_libraries(qerlio_cpp_safety_test PRIVATE qerlio_core)
    add_test(NAME qerlio_cpp_safety_test COMMAND qerlio_cpp_safety_test)
endif()

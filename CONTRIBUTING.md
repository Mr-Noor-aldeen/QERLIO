# Contributing to QERLIO Runtime

We welcome contributions to the QERLIO local runtime platform.

## Development Workflow

1. **Prerequisites:**
   - C++20 compliant compiler
   - CMake 3.22+ and Ninja
   - Python 3.10+ with `pytest`

2. **Building the Core:**
   ```bash
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

3. **Running the Test Suite:**
   Before submitting changes or pull requests, verify all automated tests pass:
   ```bash
   python -m pytest tests/ -v
   ```

4. **Code Standards:**
   - Follow standard modern C++20 idioms (RAII, const correctness, no leaked raw pointers).
   - Public C ABI functions in `include/qerlio/qerlio.h` must not throw exceptions across the binary boundary.
   - Any new quantization or compression scheme must include automated unit tests.

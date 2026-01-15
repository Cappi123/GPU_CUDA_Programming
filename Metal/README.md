# Metal GPU Programming (Apple)

This folder contains GPU programming experiments using **Metal**, Apple’s low-level graphics and compute API.
Target: **Apple Silicon GPUs** (M-series, including M4).

## Prerequisites

### Hardware
- Apple Silicon Mac (M1 / M2 / M3 / M4)

### Software
- macOS with Metal support
- Xcode or Xcode Command Line Tools
- `clang`

## Build & Run

```bash

# build (output into build/)

cd Metal/HelloWorld

mkdir -p build

clang HelloWorld.m \
 -framework Metal -framework Foundation \
 -o build/HelloWorld

# run
./build/HelloWorld

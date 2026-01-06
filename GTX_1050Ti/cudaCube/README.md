# CUDA GPU vs CPU Performance Benchmark

A demonstration of GPU parallel computing using CUDA. 
Renders thousands of 3D cubes to compare GPU vs CPU performance across different workload sizes.

## Building

### Prerequisites
- CUDA Toolkit 12.x (or compatible)
- NVIDIA GPU with CUDA support
- Windows 10/11
- Python 3.x with matplotlib, pandas, numpy (for graphs)

### Compilation

```powershell
nvcc -I./include -O2 src/main.cu src/cube_renderer.cu -o cube_renderer.exe
```

## Running

```powershell
.\cube_renderer.exe
```

Select mode:
- **Option 1**: Benchmark Mode - Automated performance comparison
- **Option 2**: Visualization Mode - Interactive 3D cube rotation

## Benchmark Mode

Runs 13 tests with increasing dataset sizes:
- 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000 cubes

Each test performs 100 iterations and measures:
- Frame time (milliseconds)
- Throughput (frames/second, vertices/second)
- Speedup factor (CPU time / GPU time)

## Output Files

After benchmark completes:

**benchmark_results.csv**
- Simplified data for graphing
- Columns: NumCubes, GPUTime_ms, CPUTime_ms, Speedup

**benchmark_details.csv**
- Complete performance metrics
- Columns: NumCubes, NumVertices, NumTriangles, GPUTime_ms, CPUTime_ms, Speedup, GPU_FPS, CPU_FPS, GPU_MVertices_per_sec, CPU_MVertices_per_sec

**plot_results.py**
- Auto-generated Python plotting script
- Creates benchmark_graph.png

## Generating Graphs

### Automatic
The program attempts to run Python automatically after benchmarking.

### Manual
```bash
python plot_results.py
```

Requires: `pip install matplotlib pandas numpy`

### Output
Creates two graphs:
1. **Performance Scaling** - Log-log plot showing GPU vs CPU time
2. **Speedup Factor** - Bar chart showing relative performance

## Understanding Results

### Expected Behavior

**Small Datasets (10-100 cubes)**
- CPU may be faster
- GPU overhead (kernel launch) dominates
- Speedup < 1.0x

**Medium Datasets (100-1000 cubes)**
- Crossover point where GPU becomes faster
- Speedup crosses 1.0x threshold

**Large Datasets (10000+ cubes)**
- GPU advantage grows dramatically
- Parallel processing benefits clear
- Speedup 10-50x or higher

### Why GPU Scales Better

**CPU Performance: O(n) - Linear**
- Limited to 4-16 cores
- Processes cubes sequentially or in small batches
- Doubling data doubles time

**GPU Performance: O(n/p) - Sub-linear**
- Thousands of CUDA cores (2000-10000+)
- Processes all cubes simultaneously (up to core limit)
- Doubling data increases time by 10-50% (until memory saturated)

### Crossover Point

The dataset size where GPU becomes faster than CPU depends on:
- GPU generation and core count
- CPU SIMD optimization (AVX/SSE)
- Memory bandwidth

Typical crossover points:
- Entry GPU (GTX 1050): 500-1000 cubes
- Mid-range GPU (RTX 3060): 100-500 cubes
- High-end GPU (RTX 4090): 50-100 cubes

## Workload Description

For each cube, the benchmark performs:
- 24 rotation transformations (3 axes × 8 vertices)
- 8 perspective projections
- 6 surface normal calculations
- 6 lighting calculations

Total: ~44 operations per cube per frame

## Visualization Mode

Interactive 3D cube with controls:
- W/S - Rotate around X axis
- A/D - Rotate around Y axis
- Q/E - Rotate around Z axis
- R - Reset rotation
- ESC - Exit

Features:
- Real-time shading with dual-light setup
- Z-buffering for proper depth
- Zero-flicker console rendering
- FPS counter

## Project Structure

```
cuda_cube_project/
├── include/
│   └── cube_renderer.cuh      # CUDA declarations and structures
├── src/
│   ├── main.cu                # Main program and benchmark orchestration
│   └── cube_renderer.cu       # GPU kernels and CPU implementations
├── plot_results.py            # Graph generation script (auto-created)
└── README.md                  # This file
```

## Technical Details

**GPU Kernels**
- benchmarkRotateMultipleCubes - Parallel rotation of all cubes
- benchmarkProjectMultipleCubes - Parallel perspective projection
- benchmarkCalculateLighting - Parallel lighting calculations

**CPU Baseline**
- Equivalent sequential implementations for comparison
- Single-threaded to show pure algorithmic performance

**Timing Methodology**
- GPU: CUDA events (microsecond precision)
- CPU: QueryPerformanceCounter (high-resolution timer)
- Each test averaged over 100 iterations


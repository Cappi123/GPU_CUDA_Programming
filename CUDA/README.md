# GPU Parallel Computing with CUDA

Found my old gaming laptop, blessed with an Nvidia GPU.
This repository documents my journey into GPU parallel computing using NVIDIA CUDA. 
Progress went from basic kernels to performance benchmarks and 3D visualization.

## Hardware Specifications

- CPU: Intel Core i7-8750H @ 2.20GHz (12 logical cores)
- GPU: NVIDIA GeForce GTX 1050 Ti (768 CUDA cores, 4GB VRAM)
- RAM: 15 GB
- CUDA Compute Capability: 6.1
- CUDA Toolkit: 12.4

Details in `Specs.txt`

## Repository Structure

```
GPU_CUDA_Programming/
├── HelloWorld.cu           # First CUDA program
├── vectorAdd.cu            # Vector addition with kernels
├── naive_matMul.cu         # Basic matrix multiplication
├── opt_matMul.cu           # Optimized matrix multiplication
├── compare/                # CPU vs GPU performance comparison
├── hot_or_not/             # GPU warmup performance study
├── cudaCube/               # 3D visualization and benchmarking
├── Specs.txt               # Hardware specifications
└── Results.txt             # Benchmark results
```

## Learning Progression

### 1. HelloWorld.cu

First CUDA kernel demonstrating:
- `__global__` kernel function
- Kernel launch syntax `<<<blocks, threads>>>`
- GPU-CPU synchronization with `cudaDeviceSynchronize()`

**Concepts**: Kernel execution, thread/block indexing

### 2. vectorAdd.cu

Vector addition implementing:
- Memory allocation with `cudaMalloc()`
- Host-to-device transfers with `cudaMemcpy()`
- Parallel element-wise operations
- Memory cleanup with `cudaFree()`

**Concepts**: Memory management, asynchronous execution, CUDA memory model

### 3. naive_matMul.cu

Basic matrix multiplication exploring:
- 2D thread indexing
- Global memory access patterns
- Thread hierarchy (grids and blocks)

**Concepts**: Thread organization, computational patterns

### 4. opt_matMul.cu

Optimized matrix multiplication introducing:
- Shared memory tiling
- Memory coalescing
- Loop unrolling
- Thread synchronization with `__syncthreads()`

**Concepts**: Memory hierarchy optimization, latency hiding

## Projects

### compare/

Comprehensive performance comparison of CPU vs GPU matrix multiplication.

**Files**:
- `main.cu` - Benchmark orchestration
- `cpu.cu` - Sequential CPU implementation
- `gpu_naive.cu` - Naive GPU kernel
- `gpu_opt.cu` - Tiled GPU kernel with shared memory
- `inc.h` / `inc.cu` - Timing utilities

**Results** (1024x1024 matrices):
```
CPU:         6626.000 ms | 0.32 GFLOP/s
GPU naive:     18.848 ms | 113.94 GFLOP/s (356x faster)
GPU opt:        9.617 ms | 223.29 GFLOP/s (688x faster)
```

**Key Findings**:
- GPU provides 300-700x speedup over CPU for large matrix operations
- Memory optimization (tiling) doubles GPU performance
- Parallel processing advantage scales with problem size

### hot_or_not/

Investigates GPU warmup effects on performance.

**Files**:
- `main.cu` - Benchmark controller
- `cold_gpu.cu` - Single cold execution
- `hot_gpu.cu` - Warmed-up averaged execution

**Results** (1024x1024 optimized matrix multiplication):
```
COLD (single run):      8.721 ms | 246.23 GFLOP/s
HOT (2 warmup + 10 avg): 7.367 ms | 291.49 GFLOP/s
```

**Key Findings**:
- First kernel launch includes initialization overhead
- GPU performs ~18% better after warmup
- Production benchmarks should include warmup runs
- Averaging multiple runs provides more reliable measurements

### cudaCube/

Interactive 3D cube renderer and GPU/CPU performance benchmark suite.

**Features**:
- Real-time 3D rendering in terminal with ASCII shading
- Zero-flicker double buffering
- Comprehensive GPU vs CPU benchmark (10 to 100,000 cubes)
- Automatic graph generation with Python
- Performance scaling analysis

**Files**:
- `src/main.cu` - Main program and benchmark orchestration
- `src/cube_renderer.cu` - GPU kernels and CPU implementations
- `include/cube_renderer.cuh` - Header file
- `plot_results.py` - Automatic graph generation
- `benchmark_results.csv` - Performance data
- `benchmark_details.csv` - Detailed metrics
- `benchmark_graph.png` - Visual comparison

**Benchmark Results**: See `cudaCube/benchmark_graph.png` and CSV files

**Key Features**:
- Demonstrates GPU advantage at scale (10-100,000 cubes)
- Shows crossover point where GPU becomes faster than CPU
- Visualizes performance scaling characteristics
- Includes throughput metrics (FPS, vertices/second)

## Compilation

All programs compiled with NVCC 12.4:

```bash
# Simple programs
nvcc HelloWorld.cu -o HelloWorld.exe
nvcc vectorAdd.cu -o vectorAdd.exe
nvcc naive_matMul.cu -o naive.exe
nvcc opt_matMul.cu -o opt.exe

# Compare project
cd compare
nvcc -I. main.cu cpu.cu gpu_naive.cu gpu_opt.cu inc.cu -o compare.exe

# Hot or not project
cd hot_or_not
nvcc main.cu cold_gpu.cu hot_gpu.cu -o hotornot.exe

# CudaCube project
cd cudaCube
nvcc -I./include -O2 src/main.cu src/cube_renderer.cu -o cube_renderer.exe
```

## Running Programs

```bash
# Basic examples
.\HelloWorld.exe
.\vectorAdd.exe
.\naive.exe
.\opt.exe

# Benchmarks
cd compare
.\compare.exe

cd hot_or_not
.\hotornot.exe

cd cudaCube
.\cube_renderer.exe
# Select mode:
#   1 - Benchmark Mode (automated performance tests)
#   2 - Visualization Mode (interactive 3D cube)
```

## Key Learnings

### Thread Hierarchy
- Threads organized in blocks
- Blocks organized in grids
- Each level has indexing: `threadIdx.x`, `blockIdx.x`
- Maximum threads per block: 1024 (hardware dependent)

### Memory Hierarchy
- **Global Memory**: Slow, large (4GB), accessible by all threads
- **Shared Memory**: Fast, small (~48KB per block), shared within block
- **Registers**: Fastest, limited, private to thread
- **Constant Memory**: Read-only, cached, good for broadcast patterns

### Performance Optimization
1. **Coalesce Memory Access**: Adjacent threads access adjacent memory
2. **Use Shared Memory**: Cache frequently accessed data
3. **Minimize Divergence**: Avoid conditional branching within warps
4. **Occupancy**: Balance threads/blocks to maximize hardware utilization
5. **Loop Unrolling**: Reduce instruction overhead
6. **Warmup**: First kernel launch includes initialization overhead

### When to Use GPU
GPU excels when:
- Large datasets (thousands to millions of elements)
- Data-parallel operations (same operation on many data points)
- High arithmetic intensity (many operations per memory access)
- Regular memory access patterns

CPU better for:
- Small datasets (overhead dominates)
- Sequential/branching algorithms
- Irregular memory access
- Low parallelism workloads

## Performance Metrics

**Matrix Multiplication (1024x1024)**:
- CPU: 0.32 GFLOP/s
- GPU Naive: 113.94 GFLOP/s (356x speedup)
- GPU Optimized: 223.29 GFLOP/s (698x speedup, 2x over naive)

**GPU Warmup Impact**:
- Cold start: 246.23 GFLOP/s
- Hot (warmed): 291.49 GFLOP/s (18% improvement)

**3D Rendering Scalability**:
- Small datasets (<100 cubes): CPU competitive
- Medium datasets (100-1000 cubes): GPU 2-6x faster
- Large datasets (>10,000 cubes): GPU 10-50x faster






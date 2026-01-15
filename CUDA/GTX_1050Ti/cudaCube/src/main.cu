#include <iostream>
#include <fstream>
#include <vector>
#include <conio.h>
#include <windows.h>
#include <cuda_runtime.h>
#include "cube_renderer.cuh"

// Get actual console window size
void getConsoleSize(int& width, int& height) {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
}

const int NUM_VERTICES = 8;
const int NUM_EDGES = 12;
const int NUM_FACES = 6;

void hideCursor(HANDLE hConsole) {
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(hConsole, &cursorInfo);
    cursorInfo.bVisible = false;
    SetConsoleCursorInfo(hConsole, &cursorInfo);
}

void displayControls() {
    std::cout << "\n";
    std::cout << "Controls:\n";
    std::cout << "  W/S - Rotate around X axis\n";
    std::cout << "  A/D - Rotate around Y axis\n";
    std::cout << "  Q/E - Rotate around Z axis\n";
    std::cout << "  R   - Reset rotation\n";
    std::cout << "  ESC - Exit\n";
}

void printHeader() {
    std::cout << "========================================================\n";
    std::cout << "     CUDA GPU vs CPU BENCHMARK - 3D Cube Renderer\n";
    std::cout << "========================================================\n\n";
}

void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable GPU found!\n";
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "GPU Information:\n";
    std::cout << "  Device: " << prop.name << "\n";
    std::cout << "  CUDA Cores: ~" << prop.multiProcessorCount * 128 << " (estimated)\n";
    std::cout << "  Clock Speed: " << prop.clockRate / 1000 << " MHz\n";
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n\n";
}

void printCPUInfo() {
    std::cout << "CPU Information:\n";
    
    // Get CPU name from registry
    HKEY hKey;
    char cpuName[256] = "Unknown CPU";
    DWORD bufferSize = sizeof(cpuName);
    
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, 
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL, 
                        (LPBYTE)cpuName, &bufferSize);
        RegCloseKey(hKey);
    }
    
    // Trim leading spaces
    char* trimmed = cpuName;
    while (*trimmed == ' ') trimmed++;
    
    std::cout << "  Processor: " << trimmed << "\n";
    
    // Get number of cores
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    std::cout << "  Logical Cores: " << sysInfo.dwNumberOfProcessors << "\n";
    
    // Get RAM
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    std::cout << "  Total RAM: " << memInfo.ullTotalPhys / (1024*1024*1024) << " GB\n\n";
}

void displayBenchmarkResults(BenchmarkResult result) {
    std::cout << "\n========================================================\n";
    std::cout << "               BENCHMARK RESULTS\n";
    std::cout << "========================================================\n\n";
    
    std::cout << "Workload:\n";
    std::cout << "  Number of Cubes: " << result.numCubes << "\n";
    std::cout << "  Total Vertices:  " << result.numVertices << " (" << result.numCubes << " x 8)\n";
    std::cout << "  Total Triangles: " << result.numTriangles << " (" << result.numCubes << " x 12)\n\n";
    
    std::cout << "Operations per frame:\n";
    std::cout << "  - Rotation matrices:   " << result.numVertices * 3 << " transformations\n";
    std::cout << "  - Projections:         " << result.numVertices << " calculations\n";
    std::cout << "  - Lighting:            " << result.numCubes * 6 << " faces\n";
    std::cout << "  - Normal calculations: " << result.numCubes * 6 << " cross products\n\n";
    
    std::cout << "========================================================\n";
    std::cout << "Performance (Average time per frame):\n";
    std::cout << "========================================================\n\n";
    
    std::cout << "  GPU Time: " << result.gpuTimeMs << " ms\n";
    std::cout << "  CPU Time: " << result.cpuTimeMs << " ms\n\n";
    
    float speedup = result.cpuTimeMs / result.gpuTimeMs;
    
    if (speedup > 1.0f) {
        std::cout << "  >> GPU is " << speedup << "x FASTER than CPU! <<\n\n";
    } else {
        std::cout << "  >> CPU is " << (1.0f/speedup) << "x FASTER than GPU! <<\n\n";
    }
    
    std::cout << "Throughput:\n";
    std::cout << "  GPU: " << (1000.0f / result.gpuTimeMs) << " frames/sec\n";
    std::cout << "  CPU: " << (1000.0f / result.cpuTimeMs) << " frames/sec\n\n";
    
    float gpuVerticesPerSec = (float)result.numVertices * (1000.0f / result.gpuTimeMs);
    float cpuVerticesPerSec = (float)result.numVertices * (1000.0f / result.cpuTimeMs);
    
    std::cout << "  GPU: " << (gpuVerticesPerSec / 1000000.0f) << " million vertices/sec\n";
    std::cout << "  CPU: " << (cpuVerticesPerSec / 1000000.0f) << " million vertices/sec\n\n";
    
    // Visual bar chart
    std::cout << "========================================================\n";
    std::cout << "Visual Comparison:\n";
    std::cout << "========================================================\n\n";
    
    int maxBarLength = 50;
    int gpuBarLength = 5; // Always show at least a small bar
    int cpuBarLength = (int)(speedup * gpuBarLength);
    if (cpuBarLength > maxBarLength) {
        float scale = (float)maxBarLength / cpuBarLength;
        cpuBarLength = maxBarLength;
        gpuBarLength = (int)(gpuBarLength * scale);
        if (gpuBarLength < 1) gpuBarLength = 1;
    }
    
    std::cout << "GPU: [";
    for (int i = 0; i < gpuBarLength; i++) std::cout << "=";
    std::cout << "] " << result.gpuTimeMs << " ms\n";
    
    std::cout << "CPU: [";
    for (int i = 0; i < cpuBarLength; i++) std::cout << "=";
    std::cout << "] " << result.cpuTimeMs << " ms\n\n";
    
    std::cout << "========================================================\n\n";
}

void runBenchmarkMode() {
    printHeader();
    printCPUInfo();
    printGPUInfo();
    
    std::cout << "This benchmark will compare GPU vs CPU performance\n";
    std::cout << "for rendering thousands of 3D cubes simultaneously.\n\n";
    
    std::cout << "Press ENTER to start benchmark...\n";
    std::cin.get();
    
    // Expanded benchmark - starting very small to show CPU advantage
    int cubeCounts[] = {10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000};
    int numTests = 13;
    int iterations = 100; // Average over 100 runs
    
    // Store results for CSV export
    std::vector<BenchmarkResult> allResults;
    
    std::cout << "\nCollecting data for " << numTests << " test points...\n";
    std::cout << "This data can be used to plot GPU vs CPU performance scaling.\n\n";
    
    for (int i = 0; i < numTests; i++) {
        int numCubes = cubeCounts[i];
        
        std::cout << "\n========================================================\n";
        std::cout << "Test " << (i+1) << "/" << numTests << ": Processing " << numCubes << " cubes...\n";
        std::cout << "========================================================\n\n";
        
        std::cout << "Running GPU benchmark (" << iterations << " iterations)...";
        std::cout.flush();
        BenchmarkResult gpuResult = runGPUBenchmark(numCubes, iterations);
        std::cout << " DONE\n";
        
        std::cout << "Running CPU benchmark (" << iterations << " iterations)...";
        std::cout.flush();
        BenchmarkResult cpuResult = runCPUBenchmark(numCubes, iterations);
        std::cout << " DONE\n";
        
        // Combine results
        BenchmarkResult combined = gpuResult;
        combined.cpuTimeMs = cpuResult.cpuTimeMs;
        allResults.push_back(combined);
        
        displayBenchmarkResults(combined);
        
        if (i < numTests - 1) {
            std::cout << "Press ENTER for next test...\n";
            std::cin.get();
        }
    }
    
    // Export to CSV for plotting
    std::ofstream csvFile("benchmark_results.csv");
    if (csvFile.is_open()) {
        csvFile << "NumCubes,GPUTime_ms,CPUTime_ms,Speedup\n";
        for (const auto& result : allResults) {
            float speedup = result.cpuTimeMs / result.gpuTimeMs;
            csvFile << result.numCubes << ","
                   << result.gpuTimeMs << ","
                   << result.cpuTimeMs << ","
                   << speedup << "\n";
        }
        csvFile.close();
    }
    
    // Export detailed results
    std::ofstream detailFile("benchmark_details.csv");
    if (detailFile.is_open()) {
        detailFile << "NumCubes,NumVertices,NumTriangles,GPUTime_ms,CPUTime_ms,Speedup,";
        detailFile << "GPU_FPS,CPU_FPS,GPU_MVertices_per_sec,CPU_MVertices_per_sec\n";
        
        for (const auto& result : allResults) {
            float speedup = result.cpuTimeMs / result.gpuTimeMs;
            float gpuFPS = 1000.0f / result.gpuTimeMs;
            float cpuFPS = 1000.0f / result.cpuTimeMs;
            float gpuMVertices = (float)result.numVertices * gpuFPS / 1000000.0f;
            float cpuMVertices = (float)result.numVertices * cpuFPS / 1000000.0f;
            
            detailFile << result.numCubes << ","
                      << result.numVertices << ","
                      << result.numTriangles << ","
                      << result.gpuTimeMs << ","
                      << result.cpuTimeMs << ","
                      << speedup << ","
                      << gpuFPS << ","
                      << cpuFPS << ","
                      << gpuMVertices << ","
                      << cpuMVertices << "\n";
        }
        detailFile.close();
    }
    
    std::cout << "\n========================================================\n";
    std::cout << "              BENCHMARK COMPLETE!\n";
    std::cout << "========================================================\n\n";
    
    std::cout << "Results saved:\n";
    std::cout << "  benchmark_results.csv  - For plotting\n";
    std::cout << "  benchmark_details.csv  - Full performance metrics\n\n";
    
    // Auto-generate and run Python plotting script
    std::ofstream pyFile("plot_results.py");
    if (pyFile.is_open()) {
        pyFile << "import matplotlib.pyplot as plt\n";
        pyFile << "import pandas as pd\n";
        pyFile << "import numpy as np\n\n";
        
        pyFile << "# Read data\n";
        pyFile << "data = pd.read_csv('benchmark_results.csv')\n\n";
        
        pyFile << "# Create figure with two subplots\n";
        pyFile << "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n\n";
        
        pyFile << "# Plot 1: Performance Comparison (Log scale)\n";
        pyFile << "ax1.plot(data['NumCubes'], data['GPUTime_ms'], 'b-o', linewidth=2, markersize=8, label='GPU Time')\n";
        pyFile << "ax1.plot(data['NumCubes'], data['CPUTime_ms'], 'r-s', linewidth=2, markersize=8, label='CPU Time')\n";
        pyFile << "ax1.set_xscale('log')\n";
        pyFile << "ax1.set_yscale('log')\n";
        pyFile << "ax1.set_xlabel('Number of Cubes', fontsize=12, fontweight='bold')\n";
        pyFile << "ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')\n";
        pyFile << "ax1.set_title('GPU vs CPU Performance Scaling (Lower is Better)', fontsize=14, fontweight='bold')\n";
        pyFile << "ax1.grid(True, alpha=0.3, linestyle='--')\n";
        pyFile << "ax1.legend(fontsize=11, loc='upper left')\n\n";
        
        pyFile << "# Find actual crossover point (interpolation)\n";
        pyFile << "crossover_idx = None\n";
        pyFile << "for i in range(len(data) - 1):\n";
        pyFile << "    if data['GPUTime_ms'].iloc[i] > data['CPUTime_ms'].iloc[i] and \\\n";
        pyFile << "       data['GPUTime_ms'].iloc[i+1] <= data['CPUTime_ms'].iloc[i+1]:\n";
        pyFile << "        # Interpolate to find exact crossover\n";
        pyFile << "        x1, y1_gpu, y1_cpu = data['NumCubes'].iloc[i], data['GPUTime_ms'].iloc[i], data['CPUTime_ms'].iloc[i]\n";
        pyFile << "        x2, y2_gpu, y2_cpu = data['NumCubes'].iloc[i+1], data['GPUTime_ms'].iloc[i+1], data['CPUTime_ms'].iloc[i+1]\n";
        pyFile << "        # Linear interpolation\n";
        pyFile << "        diff1 = y1_gpu - y1_cpu\n";
        pyFile << "        diff2 = y2_gpu - y2_cpu\n";
        pyFile << "        t = diff1 / (diff1 - diff2)\n";
        pyFile << "        crossover_x = x1 + t * (x2 - x1)\n";
        pyFile << "        crossover_y = y1_gpu + t * (y2_gpu - y1_gpu)\n";
        pyFile << "        ax1.annotate('GPU takes over',\n";
        pyFile << "                    xy=(crossover_x, crossover_y), xytext=(crossover_x*2, crossover_y*0.5),\n";
        pyFile << "                    arrowprops=dict(arrowstyle='->', color='green', lw=2),\n";
        pyFile << "                    fontsize=11, fontweight='bold', color='green',\n";
        pyFile << "                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))\n";
        pyFile << "        break\n\n";
        
        pyFile << "# Plot 2: Speedup Factor\n";
        pyFile << "colors = ['red' if s < 1 else 'green' for s in data['Speedup']]\n";
        pyFile << "ax2.bar(range(len(data)), data['Speedup'], color=colors, alpha=0.7, edgecolor='black')\n";
        pyFile << "ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Break-even (1.0x)')\n";
        pyFile << "ax2.set_xlabel('Test Number', fontsize=12, fontweight='bold')\n";
        pyFile << "ax2.set_ylabel('Speedup (CPU Time / GPU Time)', fontsize=12, fontweight='bold')\n";
        pyFile << "ax2.set_title('GPU Speedup Factor (Green = GPU Faster, Red = CPU Faster)', fontsize=14, fontweight='bold')\n";
        pyFile << "ax2.grid(True, alpha=0.3, axis='y', linestyle='--')\n";
        pyFile << "ax2.legend(fontsize=11)\n\n";
        
        pyFile << "# Add cube count labels\n";
        pyFile << "ax2.set_xticks(range(len(data)))\n";
        pyFile << "ax2.set_xticklabels([f\"{int(n)}\" for n in data['NumCubes']], rotation=45, ha='right')\n\n";
        
        pyFile << "# Add speedup values on bars\n";
        pyFile << "for i, v in enumerate(data['Speedup']):\n";
        pyFile << "    ax2.text(i, v + 0.1, f'{v:.1f}x', ha='center', va='bottom', fontweight='bold')\n\n";
        
        pyFile << "plt.tight_layout()\n";
        pyFile << "plt.savefig('benchmark_graph.png', dpi=300, bbox_inches='tight')\n";
        pyFile << "print('Graph saved: benchmark_graph.png')\n";
        pyFile << "plt.show()\n";
        
        pyFile.close();
        
        // Try to run Python automatically
        system("python plot_results.py >nul 2>&1");
    }
}

void runVisualizationMode() {
    // Get console size
    int WIDTH, HEIGHT;
    getConsoleSize(WIDTH, HEIGHT);
    
    // Reserve space for status at bottom
    HEIGHT -= 2;
    
    std::cout << "CUDA 3D Cube Renderer - Visualization Mode\n";
    std::cout << "===========================================\n";
    std::cout << "Detected console size: " << WIDTH << "x" << HEIGHT << "\n";
    displayControls();
    std::cout << "\nPress any key to start...\n";
    _getch();
    
    // Create two console screen buffers for double buffering
    HANDLE hConsole1 = CreateConsoleScreenBuffer(
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        CONSOLE_TEXTMODE_BUFFER,
        NULL
    );
    
    HANDLE hConsole2 = CreateConsoleScreenBuffer(
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        CONSOLE_TEXTMODE_BUFFER,
        NULL
    );
    
    // Hide cursor on both buffers
    hideCursor(hConsole1);
    hideCursor(hConsole2);
    
    // Start with first buffer
    SetConsoleActiveScreenBuffer(hConsole1);
    
    // Host memory
    Point3D h_vertices[NUM_VERTICES];
    int h_edges[NUM_EDGES * 2];
    Face h_faces[NUM_FACES];
    char* h_buffer = new char[WIDTH * HEIGHT];
    
    // Device memory
    Point3D* d_vertices;
    Point3D* d_rotatedVertices;
    Point2D* d_projected;
    int* d_edges;
    Face* d_faces;
    char* d_buffer;
    float* d_zBuffer;
    
    // Allocate device memory
    cudaMalloc(&d_vertices, NUM_VERTICES * sizeof(Point3D));
    cudaMalloc(&d_rotatedVertices, NUM_VERTICES * sizeof(Point3D));
    cudaMalloc(&d_projected, NUM_VERTICES * sizeof(Point2D));
    cudaMalloc(&d_edges, NUM_EDGES * 2 * sizeof(int));
    cudaMalloc(&d_faces, NUM_FACES * sizeof(Face));
    cudaMalloc(&d_buffer, WIDTH * HEIGHT * sizeof(char));
    cudaMalloc(&d_zBuffer, WIDTH * HEIGHT * sizeof(float));
    
    // Initialize cube
    initializeCube(h_vertices, h_edges, h_faces);
    
    // Copy to device
    cudaMemcpy(d_vertices, h_vertices, NUM_VERTICES * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, h_edges, NUM_EDGES * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, h_faces, NUM_FACES * sizeof(Face), cudaMemcpyHostToDevice);
    
    // Rotation angles
    float angleX = 0.3f;
    float angleY = 0.3f;
    float angleZ = 0.0f;
    
    const float rotationSpeed = 0.05f;
    
    bool running = true;
    bool useBuffer1 = true;
    
    // FPS counter
    int frameCount = 0;
    LARGE_INTEGER frequency, startTime, currentTime;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&startTime);
    float fps = 0.0f;
    
    while (running) {
        // Select current buffer to write to
        HANDLE hCurrentConsole = useBuffer1 ? hConsole1 : hConsole2;
        
        // Render frame
        renderFrame(d_vertices, d_rotatedVertices, d_projected, d_edges, d_faces,
                   d_buffer, d_zBuffer, angleX, angleY, angleZ, WIDTH, HEIGHT);
        
        // Copy buffer back to host
        cudaMemcpy(h_buffer, d_buffer, WIDTH * HEIGHT * sizeof(char), cudaMemcpyDeviceToHost);
        
        // Write entire frame to back buffer
        COORD coord = {0, 0};
        DWORD written;
        
        // Write the rendered cube
        for (int y = 0; y < HEIGHT; y++) {
            coord.Y = y;
            coord.X = 0;
            WriteConsoleOutputCharacterA(hCurrentConsole, 
                                        &h_buffer[y * WIDTH], 
                                        WIDTH, 
                                        coord, 
                                        &written);
        }
        
        // Calculate FPS
        frameCount++;
        QueryPerformanceCounter(&currentTime);
        double elapsed = (double)(currentTime.QuadPart - startTime.QuadPart) / frequency.QuadPart;
        if (elapsed >= 0.5) { // Update FPS every 0.5 seconds
            fps = frameCount / elapsed;
            frameCount = 0;
            startTime = currentTime;
        }
        
        // Write status line at bottom
        coord.Y = HEIGHT;
        coord.X = 0;
        char status[256];
        sprintf(status, "X:%.2f Y:%.2f Z:%.2f | FPS:%.1f | W/S:X A/D:Y Q/E:Z R:Reset ESC:Exit", 
                angleX, angleY, angleZ, fps);
        
        // Pad status to full width
        int statusLen = strlen(status);
        for (int i = statusLen; i < WIDTH && i < 256; i++) {
            status[i] = ' ';
        }
        status[WIDTH < 256 ? WIDTH : 255] = '\0';
        
        WriteConsoleOutputCharacterA(hCurrentConsole, status, strlen(status), coord, &written);
        
        // Swap buffers - this is atomic and eliminates all flickering!
        SetConsoleActiveScreenBuffer(hCurrentConsole);
        useBuffer1 = !useBuffer1;
        
        // Check for input (non-blocking)
        if (_kbhit()) {
            int key = _getch();
            
            switch (key) {
                case 'w': case 'W':
                    angleX += rotationSpeed;
                    break;
                case 's': case 'S':
                    angleX -= rotationSpeed;
                    break;
                case 'a': case 'A':
                    angleY -= rotationSpeed;
                    break;
                case 'd': case 'D':
                    angleY += rotationSpeed;
                    break;
                case 'q': case 'Q':
                    angleZ -= rotationSpeed;
                    break;
                case 'e': case 'E':
                    angleZ += rotationSpeed;
                    break;
                case 'r': case 'R':
                    angleX = 0.3f;
                    angleY = 0.3f;
                    angleZ = 0.0f;
                    break;
                case 27: // ESC
                    running = false;
                    break;
            }
        }
        
        Sleep(16); // ~60 FPS cap
    }
    
    // Cleanup
    delete[] h_buffer;
    cudaFree(d_vertices);
    cudaFree(d_rotatedVertices);
    cudaFree(d_projected);
    cudaFree(d_edges);
    cudaFree(d_faces);
    cudaFree(d_buffer);
    cudaFree(d_zBuffer);
    
    // Restore default console
    SetConsoleActiveScreenBuffer(GetStdHandle(STD_OUTPUT_HANDLE));
    CloseHandle(hConsole1);
    CloseHandle(hConsole2);
    
    system("cls");
}

int main() {
    // Initialize CUDA
    cudaSetDevice(0);
    
    printHeader();
    
    std::cout << "Select Mode:\n\n";
    std::cout << "  1. Benchmark Mode  - Compare GPU vs CPU performance\n";
    std::cout << "  2. Visualization Mode - Interactive 3D cube rotation\n\n";
    std::cout << "Enter choice (1 or 2): ";
    
    int choice;
    std::cin >> choice;
    std::cin.ignore(); // Clear newline
    
    if (choice == 1) {
        runBenchmarkMode();
    } else if (choice == 2) {
        runVisualizationMode();
    } else {
        std::cout << "Invalid choice. Exiting.\n";
        return 1;
    }
    
    std::cout << "\nThank you for using CUDA 3D Cube Renderer!\n";
    std::cout << "Press ENTER to exit...";
    std::cin.get();
    
    return 0;
}

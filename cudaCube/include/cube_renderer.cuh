#ifndef CUBE_RENDERER_CUH
#define CUBE_RENDERER_CUH

#include <cuda_runtime.h>

// Structure to hold 3D point
struct Point3D {
    float x, y, z;
};

// Structure to hold 2D point (projection)
struct Point2D {
    int x, y;
};

// Structure for a face (quad)
struct Face {
    int v[4];  // 4 vertex indices
    float avgZ; // Average Z for depth sorting
};

// Structure for benchmark results
struct BenchmarkResult {
    float gpuTimeMs;
    float cpuTimeMs;
    float speedup;
    int numCubes;
    int numVertices;
    int numTriangles;
};

// CUDA kernel declarations
__global__ void rotateAndProjectMultipleCubes(
    Point3D* vertices, 
    Point2D* projected,
    float* angles,
    int numCubes,
    int screenWidth,
    int screenHeight
);

__global__ void calculateFaceDepthMultipleCubes(
    Point3D* rotatedVertices,
    Face* faces,
    int numCubes
);

__global__ void rotateAndProject(
    Point3D* vertices, 
    Point2D* projected, 
    int numVertices,
    float angleX, 
    float angleY, 
    float angleZ,
    int screenWidth,
    int screenHeight
);

__global__ void calculateFaceDepth(
    Point3D* rotatedVertices,
    Face* faces,
    int numFaces
);

__global__ void clearBuffer(char* buffer, float* zBuffer, int width, int height);

__global__ void drawFaces(
    char* buffer,
    float* zBuffer,
    Point2D* projected,
    Point3D* rotatedVertices,
    Face* faces,
    int numFaces,
    int width,
    int height
);

__global__ void drawEdges(
    char* buffer,
    Point2D* projected,
    int* edges,
    int numEdges,
    int width,
    int height
);

// Host function declarations
void initializeCube(Point3D* h_vertices, int* h_edges, Face* h_faces);
void renderFrame(
    Point3D* d_vertices,
    Point3D* d_rotatedVertices,
    Point2D* d_projected,
    int* d_edges,
    Face* d_faces,
    char* d_buffer,
    float* d_zBuffer,
    float angleX,
    float angleY,
    float angleZ,
    int width,
    int height
);

// Benchmark functions
BenchmarkResult runGPUBenchmark(int numCubes, int iterations);
BenchmarkResult runCPUBenchmark(int numCubes, int iterations);
void displayBenchmarkResults(BenchmarkResult gpu, BenchmarkResult cpu);

#endif // CUBE_RENDERER_CUH

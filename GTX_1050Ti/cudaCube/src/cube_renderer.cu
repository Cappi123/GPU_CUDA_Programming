#include "cube_renderer.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <windows.h>  // For QueryPerformanceCounter

// Rotation matrices and projection
__host__ __device__ Point3D rotatePointReturnCopy(Point3D p, float ax, float ay, float az) {
    // Rotate around X axis
    float y = p.y * cosf(ax) - p.z * sinf(ax);
    float z = p.y * sinf(ax) + p.z * cosf(ax);
    p.y = y;
    p.z = z;
    
    // Rotate around Y axis
    float x = p.x * cosf(ay) + p.z * sinf(ay);
    z = -p.x * sinf(ay) + p.z * cosf(ay);
    p.x = x;
    p.z = z;
    
    // Rotate around Z axis
    x = p.x * cosf(az) - p.y * sinf(az);
    y = p.x * sinf(az) + p.y * cosf(az);
    p.x = x;
    p.y = y;
    
    return p;
}

__host__ __device__ Point2D projectPoint(Point3D p, int screenWidth, int screenHeight) {
    // Simple perspective projection with automatic scaling
    float distance = 5.0f;
    
    // Scale FOV based on smaller dimension to fit cube nicely
    float minDim = fminf((float)screenWidth, (float)screenHeight * 2.0f);
    float fov = minDim * 0.6f; // Cube will be ~60% of screen size - much larger!
    
    float factor = fov / (distance + p.z);
    
    Point2D result;
    result.x = (int)(p.x * factor + screenWidth / 2);
    result.y = (int)(p.y * factor + screenHeight / 2);
    
    return result;
}

__global__ void rotateAndProject(
    Point3D* vertices, 
    Point2D* projected, 
    int numVertices,
    float angleX, 
    float angleY, 
    float angleZ,
    int screenWidth,
    int screenHeight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numVertices) {
        Point3D p = rotatePointReturnCopy(vertices[idx], angleX, angleY, angleZ);
        projected[idx] = projectPoint(p, screenWidth, screenHeight);
    }
}

__global__ void calculateFaceDepth(
    Point3D* rotatedVertices,
    Face* faces,
    int numFaces
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numFaces) {
        // Calculate average Z depth for sorting
        float sumZ = 0.0f;
        for (int i = 0; i < 4; i++) {
            sumZ += rotatedVertices[faces[idx].v[i]].z;
        }
        faces[idx].avgZ = sumZ / 4.0f;
    }
}

__global__ void clearBuffer(char* buffer, float* zBuffer, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx < total) {
        buffer[idx] = ' ';
        zBuffer[idx] = 1000.0f; // Far distance
    }
}

__device__ char getShadingChar(float brightness) {
    // Enhanced ASCII shading with better gradients
    const char shades[] = " .,-~:;=!*#$@";
    int numShades = 13;
    int index = (int)(brightness * (numShades - 1));
    if (index < 0) index = 0;
    if (index >= numShades) index = numShades - 1;
    return shades[index];
}

__device__ void drawTriangle(
    char* buffer,
    float* zBuffer,
    Point2D p0, Point2D p1, Point2D p2,
    float z0, float z1, float z2,
    float brightness,
    int width, int height
) {
    // Simple triangle rasterization with z-buffering
    // Find bounding box
    int minX = min(min(p0.x, p1.x), p2.x);
    int maxX = max(max(p0.x, p1.x), p2.x);
    int minY = min(min(p0.y, p1.y), p2.y);
    int maxY = max(max(p0.y, p1.y), p2.y);
    
    // Clamp to screen bounds
    minX = max(0, minX);
    maxX = min(width - 1, maxX);
    minY = max(0, minY);
    maxY = min(height - 1, maxY);
    
    char shade = getShadingChar(brightness);
    
    // Barycentric coordinates
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            // Calculate barycentric coordinates
            float w0 = (p1.x - p0.x) * (y - p0.y) - (p1.y - p0.y) * (x - p0.x);
            float w1 = (p2.x - p1.x) * (y - p1.y) - (p2.y - p1.y) * (x - p1.x);
            float w2 = (p0.x - p2.x) * (y - p2.y) - (p0.y - p2.y) * (x - p2.x);
            
            // Check if point is inside triangle
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                int idx = y * width + x;
                
                // Simple depth approximation
                float z = (z0 + z1 + z2) / 3.0f;
                
                if (z < zBuffer[idx]) {
                    zBuffer[idx] = z;
                    buffer[idx] = shade;
                }
            }
        }
    }
}

__global__ void drawFaces(
    char* buffer,
    float* zBuffer,
    Point2D* projected,
    Point3D* rotatedVertices,
    Face* faces,
    int numFaces,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numFaces) {
        Face face = faces[idx];
        
        // Get face normal for lighting (simple calculation)
        Point3D v0 = rotatedVertices[face.v[0]];
        Point3D v1 = rotatedVertices[face.v[1]];
        Point3D v2 = rotatedVertices[face.v[2]];
        
        // Calculate normal using cross product
        float dx1 = v1.x - v0.x, dy1 = v1.y - v0.y, dz1 = v1.z - v0.z;
        float dx2 = v2.x - v0.x, dy2 = v2.y - v0.y, dz2 = v2.z - v0.z;
        
        float nx = dy1 * dz2 - dz1 * dy2;
        float ny = dz1 * dx2 - dx1 * dz2;
        float nz = dx1 * dy2 - dy1 * dx2;
        
        // Normalize
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 0.001f) {
            nx /= len; ny /= len; nz /= len;
        }
        
        // Enhanced lighting with multiple light sources
        // Main light (top-right-front)
        float lx1 = 0.6f, ly1 = -0.5f, lz1 = -0.6f;
        float len1 = sqrtf(lx1*lx1 + ly1*ly1 + lz1*lz1);
        lx1 /= len1; ly1 /= len1; lz1 /= len1;
        
        // Fill light (left side, subtle)
        float lx2 = -0.3f, ly2 = 0.0f, lz2 = -0.5f;
        float len2 = sqrtf(lx2*lx2 + ly2*ly2 + lz2*lz2);
        lx2 /= len2; ly2 /= len2; lz2 /= len2;
        
        // Calculate brightness (Lambertian shading)
        float brightness1 = nx * lx1 + ny * ly1 + nz * lz1;
        float brightness2 = nx * lx2 + ny * ly2 + nz * lz2;
        
        // Combine lights
        brightness1 = fmaxf(0.0f, brightness1) * 0.8f; // Main light
        brightness2 = fmaxf(0.0f, brightness2) * 0.3f; // Fill light
        
        float brightness = brightness1 + brightness2 + 0.15f; // Add ambient
        brightness = fminf(1.0f, brightness); // Clamp
        
        // Enhanced contrast
        brightness = powf(brightness, 0.8f);
        
        // Back-face culling (only draw faces facing camera)
        if (nz < 0) {
            // Draw face as two triangles
            Point2D p0 = projected[face.v[0]];
            Point2D p1 = projected[face.v[1]];
            Point2D p2 = projected[face.v[2]];
            Point2D p3 = projected[face.v[3]];
            
            float z0 = rotatedVertices[face.v[0]].z;
            float z1 = rotatedVertices[face.v[1]].z;
            float z2 = rotatedVertices[face.v[2]].z;
            float z3 = rotatedVertices[face.v[3]].z;
            
            drawTriangle(buffer, zBuffer, p0, p1, p2, z0, z1, z2, brightness, width, height);
            drawTriangle(buffer, zBuffer, p0, p2, p3, z0, z2, z3, brightness, width, height);
        }
    }
}

__device__ void drawLine(char* buffer, int x0, int y0, int x1, int y1, int width, int height, char ch) {
    // Bresenham's line algorithm with anti-aliasing effect
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    while (true) {
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            buffer[y0 * width + x0] = ch;
        }
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

__global__ void drawEdges(
    char* buffer,
    Point2D* projected,
    int* edges,
    int numEdges,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numEdges) {
        int v1 = edges[idx * 2];
        int v2 = edges[idx * 2 + 1];
        
        Point2D p1 = projected[v1];
        Point2D p2 = projected[v2];
        
        // Use different characters for better edge visibility
        drawLine(buffer, p1.x, p1.y, p2.x, p2.y, width, height, '#');
    }
}

// Host functions
void initializeCube(Point3D* h_vertices, int* h_edges, Face* h_faces) {
    // Define cube vertices (centered at origin)
    float size = 1.0f;
    h_vertices[0] = {-size, -size, -size};
    h_vertices[1] = { size, -size, -size};
    h_vertices[2] = { size,  size, -size};
    h_vertices[3] = {-size,  size, -size};
    h_vertices[4] = {-size, -size,  size};
    h_vertices[5] = { size, -size,  size};
    h_vertices[6] = { size,  size,  size};
    h_vertices[7] = {-size,  size,  size};
    
    // Define cube edges (12 edges)
    int edges_temp[24] = {
        0, 1,  1, 2,  2, 3,  3, 0,  // Back face
        4, 5,  5, 6,  6, 7,  7, 4,  // Front face
        0, 4,  1, 5,  2, 6,  3, 7   // Connecting edges
    };
    
    for (int i = 0; i < 24; i++) {
        h_edges[i] = edges_temp[i];
    }
    
    // Define 6 faces (each is a quad)
    h_faces[0] = {{0, 1, 2, 3}, 0.0f}; // Back
    h_faces[1] = {{4, 5, 6, 7}, 0.0f}; // Front
    h_faces[2] = {{0, 1, 5, 4}, 0.0f}; // Bottom
    h_faces[3] = {{2, 3, 7, 6}, 0.0f}; // Top
    h_faces[4] = {{0, 3, 7, 4}, 0.0f}; // Left
    h_faces[5] = {{1, 2, 6, 5}, 0.0f}; // Right
}

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
) {
    int numVertices = 8;
    int numEdges = 12;
    int numFaces = 6;
    
    // Clear buffers
    int bufferSize = width * height;
    int blockSize = 256;
    int numBlocks = (bufferSize + blockSize - 1) / blockSize;
    clearBuffer<<<numBlocks, blockSize>>>(d_buffer, d_zBuffer, width, height);
    cudaDeviceSynchronize();
    
    // Rotate vertices and store rotated positions
    Point3D* h_vertices = new Point3D[numVertices];
    cudaMemcpy(h_vertices, d_vertices, numVertices * sizeof(Point3D), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numVertices; i++) {
        h_vertices[i] = rotatePointReturnCopy(h_vertices[i], angleX, angleY, angleZ);
    }
    cudaMemcpy(d_rotatedVertices, h_vertices, numVertices * sizeof(Point3D), cudaMemcpyHostToDevice);
    delete[] h_vertices;
    
    // Project vertices
    rotateAndProject<<<1, numVertices>>>(
        d_vertices, d_projected, numVertices,
        angleX, angleY, angleZ,
        width, height
    );
    cudaDeviceSynchronize();
    
    // Calculate face depths
    calculateFaceDepth<<<1, numFaces>>>(d_rotatedVertices, d_faces, numFaces);
    cudaDeviceSynchronize();
    
    // Draw filled faces
    drawFaces<<<1, numFaces>>>(d_buffer, d_zBuffer, d_projected, d_rotatedVertices, d_faces, numFaces, width, height);
    cudaDeviceSynchronize();
    
    // Draw edges on top for clarity
    drawEdges<<<1, numEdges>>>(d_buffer, d_projected, d_edges, numEdges, width, height);
    cudaDeviceSynchronize();
}

// ============================================================================
// BENCHMARK FUNCTIONS - GPU vs CPU Performance Comparison
// ============================================================================

// GPU Benchmark: Process thousands of cubes in parallel
__global__ void benchmarkRotateMultipleCubes(
    Point3D* vertices,      // Input vertices for all cubes
    Point3D* rotatedOut,    // Output rotated vertices
    float* angles,          // Random angles for each cube
    int numCubes
) {
    int cubeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (cubeIdx < numCubes) {
        int baseIdx = cubeIdx * 8; // 8 vertices per cube
        float angleX = angles[cubeIdx * 3 + 0];
        float angleY = angles[cubeIdx * 3 + 1];
        float angleZ = angles[cubeIdx * 3 + 2];
        
        // Rotate all 8 vertices of this cube
        for (int v = 0; v < 8; v++) {
            Point3D vertex = vertices[baseIdx + v];
            rotatedOut[baseIdx + v] = rotatePointReturnCopy(vertex, angleX, angleY, angleZ);
        }
    }
}

__global__ void benchmarkProjectMultipleCubes(
    Point3D* rotatedVertices,
    Point2D* projected,
    int numCubes,
    int screenWidth,
    int screenHeight
) {
    int vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalVertices = numCubes * 8;
    
    if (vertexIdx < totalVertices) {
        projected[vertexIdx] = projectPoint(rotatedVertices[vertexIdx], screenWidth, screenHeight);
    }
}

__global__ void benchmarkCalculateLighting(
    Point3D* rotatedVertices,
    float* brightness,
    int numCubes
) {
    int faceIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalFaces = numCubes * 6;
    
    if (faceIdx < totalFaces) {
        int cubeIdx = faceIdx / 6;
        int localFaceIdx = faceIdx % 6;
        int baseIdx = cubeIdx * 8;
        
        // Define face vertices (same as initializeCube)
        int faceVertices[6][4] = {
            {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
            {2, 3, 7, 6}, {0, 3, 7, 4}, {1, 2, 6, 5}
        };
        
        Point3D v0 = rotatedVertices[baseIdx + faceVertices[localFaceIdx][0]];
        Point3D v1 = rotatedVertices[baseIdx + faceVertices[localFaceIdx][1]];
        Point3D v2 = rotatedVertices[baseIdx + faceVertices[localFaceIdx][2]];
        
        // Calculate normal
        float dx1 = v1.x - v0.x, dy1 = v1.y - v0.y, dz1 = v1.z - v0.z;
        float dx2 = v2.x - v0.x, dy2 = v2.y - v0.y, dz2 = v2.z - v0.z;
        
        float nx = dy1 * dz2 - dz1 * dy2;
        float ny = dz1 * dx2 - dx1 * dz2;
        float nz = dx1 * dy2 - dy1 * dx2;
        
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 0.001f) {
            nx /= len; ny /= len; nz /= len;
        }
        
        // Lighting
        float lx = 0.6f, ly = -0.5f, lz = -0.6f;
        float llen = sqrtf(lx*lx + ly*ly + lz*lz);
        lx /= llen; ly /= llen; lz /= llen;
        
        float b = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
        brightness[faceIdx] = b * 0.8f + 0.15f;
    }
}

// CPU equivalent - Single threaded for comparison
void cpuRotateMultipleCubes(
    Point3D* vertices,
    Point3D* rotatedOut,
    float* angles,
    int numCubes
) {
    for (int cubeIdx = 0; cubeIdx < numCubes; cubeIdx++) {
        int baseIdx = cubeIdx * 8;
        float angleX = angles[cubeIdx * 3 + 0];
        float angleY = angles[cubeIdx * 3 + 1];
        float angleZ = angles[cubeIdx * 3 + 2];
        
        for (int v = 0; v < 8; v++) {
            Point3D vertex = vertices[baseIdx + v];
            rotatedOut[baseIdx + v] = rotatePointReturnCopy(vertex, angleX, angleY, angleZ);
        }
    }
}

void cpuProjectMultipleCubes(
    Point3D* rotatedVertices,
    Point2D* projected,
    int numCubes
) {
    int totalVertices = numCubes * 8;
    for (int i = 0; i < totalVertices; i++) {
        projected[i] = projectPoint(rotatedVertices[i], 80, 40);
    }
}

void cpuCalculateLighting(
    Point3D* rotatedVertices,
    float* brightness,
    int numCubes
) {
    int faceVertices[6][4] = {
        {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4},
        {2, 3, 7, 6}, {0, 3, 7, 4}, {1, 2, 6, 5}
    };
    
    int totalFaces = numCubes * 6;
    for (int faceIdx = 0; faceIdx < totalFaces; faceIdx++) {
        int cubeIdx = faceIdx / 6;
        int localFaceIdx = faceIdx % 6;
        int baseIdx = cubeIdx * 8;
        
        Point3D v0 = rotatedVertices[baseIdx + faceVertices[localFaceIdx][0]];
        Point3D v1 = rotatedVertices[baseIdx + faceVertices[localFaceIdx][1]];
        Point3D v2 = rotatedVertices[baseIdx + faceVertices[localFaceIdx][2]];
        
        float dx1 = v1.x - v0.x, dy1 = v1.y - v0.y, dz1 = v1.z - v0.z;
        float dx2 = v2.x - v0.x, dy2 = v2.y - v0.y, dz2 = v2.z - v0.z;
        
        float nx = dy1 * dz2 - dz1 * dy2;
        float ny = dz1 * dx2 - dx1 * dz2;
        float nz = dx1 * dy2 - dy1 * dx2;
        
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 0.001f) {
            nx /= len; ny /= len; nz /= len;
        }
        
        float lx = 0.6f, ly = -0.5f, lz = -0.6f;
        float llen = sqrtf(lx*lx + ly*ly + lz*lz);
        lx /= llen; ly /= llen; lz /= llen;
        
        float b = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
        brightness[faceIdx] = b * 0.8f + 0.15f;
    }
}

BenchmarkResult runGPUBenchmark(int numCubes, int iterations) {
    BenchmarkResult result;
    result.numCubes = numCubes;
    result.numVertices = numCubes * 8;
    result.numTriangles = numCubes * 12; // 6 faces * 2 triangles each
    
    // Allocate memory
    int totalVertices = numCubes * 8;
    int totalFaces = numCubes * 6;
    
    Point3D* d_vertices;
    Point3D* d_rotated;
    Point2D* d_projected;
    float* d_angles;
    float* d_brightness;
    
    cudaMalloc(&d_vertices, totalVertices * sizeof(Point3D));
    cudaMalloc(&d_rotated, totalVertices * sizeof(Point3D));
    cudaMalloc(&d_projected, totalVertices * sizeof(Point2D));
    cudaMalloc(&d_angles, numCubes * 3 * sizeof(float));
    cudaMalloc(&d_brightness, totalFaces * sizeof(float));
    
    // Initialize data
    Point3D* h_vertices = new Point3D[totalVertices];
    float* h_angles = new float[numCubes * 3];
    
    float size = 1.0f;
    Point3D baseVertices[8] = {
        {-size, -size, -size}, { size, -size, -size},
        { size,  size, -size}, {-size,  size, -size},
        {-size, -size,  size}, { size, -size,  size},
        { size,  size,  size}, {-size,  size,  size}
    };
    
    for (int c = 0; c < numCubes; c++) {
        for (int v = 0; v < 8; v++) {
            h_vertices[c * 8 + v] = baseVertices[v];
        }
        h_angles[c * 3 + 0] = (float)(rand() % 360) * 3.14159f / 180.0f;
        h_angles[c * 3 + 1] = (float)(rand() % 360) * 3.14159f / 180.0f;
        h_angles[c * 3 + 2] = (float)(rand() % 360) * 3.14159f / 180.0f;
    }
    
    cudaMemcpy(d_vertices, h_vertices, totalVertices * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_angles, h_angles, numCubes * 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Kernel configuration
    int blockSize = 256;
    int cubeBlocks = (numCubes + blockSize - 1) / blockSize;
    int vertexBlocks = (totalVertices + blockSize - 1) / blockSize;
    int faceBlocks = (totalFaces + blockSize - 1) / blockSize;
    
    // Warmup
    benchmarkRotateMultipleCubes<<<cubeBlocks, blockSize>>>(d_vertices, d_rotated, d_angles, numCubes);
    cudaDeviceSynchronize();
    
    // Benchmark - only time GPU computation, not memory transfer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        benchmarkRotateMultipleCubes<<<cubeBlocks, blockSize>>>(d_vertices, d_rotated, d_angles, numCubes);
        benchmarkProjectMultipleCubes<<<vertexBlocks, blockSize>>>(d_rotated, d_projected, numCubes, 80, 40);
        benchmarkCalculateLighting<<<faceBlocks, blockSize>>>(d_rotated, d_brightness, numCubes);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    result.gpuTimeMs = milliseconds / iterations;
    
    // Cleanup
    delete[] h_vertices;
    delete[] h_angles;
    cudaFree(d_vertices);
    cudaFree(d_rotated);
    cudaFree(d_projected);
    cudaFree(d_angles);
    cudaFree(d_brightness);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

BenchmarkResult runCPUBenchmark(int numCubes, int iterations) {
    BenchmarkResult result;
    result.numCubes = numCubes;
    result.numVertices = numCubes * 8;
    result.numTriangles = numCubes * 12;
    
    int totalVertices = numCubes * 8;
    int totalFaces = numCubes * 6;
    
    Point3D* vertices = new Point3D[totalVertices];
    Point3D* rotated = new Point3D[totalVertices];
    Point2D* projected = new Point2D[totalVertices];
    float* angles = new float[numCubes * 3];
    float* brightness = new float[totalFaces];
    
    // Initialize
    float size = 1.0f;
    Point3D baseVertices[8] = {
        {-size, -size, -size}, { size, -size, -size},
        { size,  size, -size}, {-size,  size, -size},
        {-size, -size,  size}, { size, -size,  size},
        { size,  size,  size}, {-size,  size,  size}
    };
    
    for (int c = 0; c < numCubes; c++) {
        for (int v = 0; v < 8; v++) {
            vertices[c * 8 + v] = baseVertices[v];
        }
        angles[c * 3 + 0] = (float)(rand() % 360) * 3.14159f / 180.0f;
        angles[c * 3 + 1] = (float)(rand() % 360) * 3.14159f / 180.0f;
        angles[c * 3 + 2] = (float)(rand() % 360) * 3.14159f / 180.0f;
    }
    
    // Benchmark with high-resolution timer
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    
    for (int i = 0; i < iterations; i++) {
        cpuRotateMultipleCubes(vertices, rotated, angles, numCubes);
        cpuProjectMultipleCubes(rotated, projected, numCubes);
        cpuCalculateLighting(rotated, brightness, numCubes);
    }
    
    QueryPerformanceCounter(&end);
    double elapsed = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    result.cpuTimeMs = (float)(elapsed / iterations);
    
    // Cleanup
    delete[] vertices;
    delete[] rotated;
    delete[] projected;
    delete[] angles;
    delete[] brightness;
    
    return result;
}

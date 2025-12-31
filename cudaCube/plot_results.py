import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data
data = pd.read_csv('benchmark_results.csv')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Performance Comparison (Log scale)
ax1.plot(data['NumCubes'], data['GPUTime_ms'], 'b-o', linewidth=2, markersize=8, label='GPU Time')
ax1.plot(data['NumCubes'], data['CPUTime_ms'], 'r-s', linewidth=2, markersize=8, label='CPU Time')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Number of Cubes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('GPU vs CPU Performance Scaling (Lower is Better)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper left')

# Find actual crossover point (interpolation)
crossover_idx = None
for i in range(len(data) - 1):
    if data['GPUTime_ms'].iloc[i] > data['CPUTime_ms'].iloc[i] and \
       data['GPUTime_ms'].iloc[i+1] <= data['CPUTime_ms'].iloc[i+1]:
        # Interpolate to find exact crossover
        x1, y1_gpu, y1_cpu = data['NumCubes'].iloc[i], data['GPUTime_ms'].iloc[i], data['CPUTime_ms'].iloc[i]
        x2, y2_gpu, y2_cpu = data['NumCubes'].iloc[i+1], data['GPUTime_ms'].iloc[i+1], data['CPUTime_ms'].iloc[i+1]
        # Linear interpolation
        diff1 = y1_gpu - y1_cpu
        diff2 = y2_gpu - y2_cpu
        t = diff1 / (diff1 - diff2)
        crossover_x = x1 + t * (x2 - x1)
        crossover_y = y1_gpu + t * (y2_gpu - y1_gpu)
        ax1.annotate('GPU takes over',
                    xy=(crossover_x, crossover_y), xytext=(crossover_x*2, crossover_y*0.5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        break

# Plot 2: Speedup Factor
colors = ['red' if s < 1 else 'green' for s in data['Speedup']]
ax2.bar(range(len(data)), data['Speedup'], color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Break-even (1.0x)')
ax2.set_xlabel('Test Number', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup (CPU Time / GPU Time)', fontsize=12, fontweight='bold')
ax2.set_title('GPU Speedup Factor (Green = GPU Faster, Red = CPU Faster)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.legend(fontsize=11)

# Add cube count labels
ax2.set_xticks(range(len(data)))
ax2.set_xticklabels([f"{int(n)}" for n in data['NumCubes']], rotation=45, ha='right')

# Add speedup values on bars
for i, v in enumerate(data['Speedup']):
    ax2.text(i, v + 0.1, f'{v:.1f}x', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_graph.png', dpi=300, bbox_inches='tight')
print('Graph saved: benchmark_graph.png')
plt.show()

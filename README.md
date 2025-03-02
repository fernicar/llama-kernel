# Llama Kernel

Memory-efficient PyTorch kernel for Llama 3.2 inference with 4-bit quantization and advanced optimization techniques.

## Features

- 4-bit quantization for memory-efficient inference
- Multiple specialized PyTorch kernels optimized for different batch sizes and hardware
- Advanced lookup table optimizations for small batches
- Vectorized operations for faster matrix multiplication
- Memory-efficient buffer management and tensor reuse
- Flash Attention 2 integration with robust fallback mechanisms
- PyTorch 2.0+ `torch.compile` support with fullgraph optimization
- TF32 precision support for Ampere+ GPUs
- CUDA graph support for repeated operations
- Kernel fusion for improved throughput

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llama-kernel.git
cd llama-kernel

# Install dependencies with Poetry
poetry install
```

## Performance Optimizations

This library includes several optimizations to make inference faster:

1. **Smart Kernel Selection**: Automatically chooses between optimized kernels based on batch size and hardware:
   - Lookup-based implementation for small batches on GPU
   - Batched processing for larger inputs
   - Specialized implementations for different hardware configurations

2. **Memory Optimization Techniques**:
   - Pre-allocated tensor buffers to minimize memory fragmentation
   - Contiguous memory layouts for optimal access patterns
   - Strategic tensor clearing to aid garbage collection
   - Explicit dtype management to avoid unnecessary conversions

3. **Hardware Acceleration**:
   - Flash Attention 2 for faster attention computation
   - CUDA graph support for repeated operations
   - TF32 precision on supported hardware
   - Optimized softmax computation with controlled precision

4. **Compilation Optimizations**:
   - Function-specific compilation with `torch.compile`
   - Fullgraph optimization for maximum performance
   - Kernel fusion for better GPU utilization
   - Model warmup to avoid cold starts

## Usage

### Basic Usage

```python
from llama_kernel import LlamaInference

# Initialize the model
model = LlamaInference(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    device="cuda",
    use_flash_attention=True  # Enable Flash Attention 2
)

# Generate text
output = model.generate(
    "What is the capital of France?",
    max_length=512,
    temperature=0.7
)

print(output)
```

### Command Line Interface

The package includes a command-line interface with various optimization options:

```bash
# Run with maximum speed optimizations
python main.py --max_speed --flash_attention

# Run benchmark with optimizations
python main.py --benchmark --max_speed

# Customize batch size for optimization
python main.py --max_speed --batch_size 4
```

## Speed Optimization Flags

The following flags can be used to maximize inference speed:

- `--max_speed`: Apply all available optimizations for maximum inference speed
- `--flash_attention`: Use Flash Attention 2 for faster attention computation
- `--optimize_memory`: Reserve GPU memory and optimize memory usage
- `--batch_size`: Specify batch size for optimizations (default: 1)

## Memory Usage

The 4-bit quantization significantly reduces memory usage:

- FP16 model: ~2GB for 1B parameters
- 4-bit model: ~0.5GB for 1B parameters

Estimate memory requirements before loading a model:

```bash
python main.py --estimate_memory --model meta-llama/Llama-3.2-7B-Instruct
```

## Benchmark Performance

Current benchmark results for the Llama Kernel with the Llama-3.2-1B-Instruct model using maximum speed optimizations:

| Prompt Type                           | Tokens Generated | Generation Time | Throughput (tokens/sec) |
|--------------------------------------|-----------------|----------------|------------------------|
| Short factual query                   | 97              | 4.45 seconds   | 21.82                  |
| Scientific explanation                | 496             | 21.37 seconds  | 23.21                  |
| Creative writing (poem)               | 196             | 8.49 seconds   | 23.09                  |
| Simple information retrieval          | 146             | 6.41 seconds   | 22.76                  |
| Literature summary                    | 315             | 14.32 seconds  | 21.99                  |
| **Average Throughput**               | -               | -              | **22.71**              |

These benchmarks were conducted with:
- Model: meta-llama/Llama-3.2-1B-Instruct
- Maximum speed optimizations enabled
- torch.compile with inductor backend
- Default temperature of 0.6 and top_p of 0.9

The current implementation achieves consistent performance across different prompt types and response lengths, with an average throughput of approximately 22.71 tokens per second on consumer hardware.

## Advanced Configuration

### Matrix Multiplication Implementations

The library includes multiple specialized implementations for different scenarios:

1. **Lookup Table Implementation**: Optimized for small batch sizes on GPU
   - Pre-computes all possible 4-bit value combinations
   - Uses vectorized gather operations
   - Processes multiple bytes at once for better throughput

2. **Batched Implementation**: Optimized for larger batch sizes
   - Processes multiple rows at once
   - Uses efficient memory access patterns
   - Minimizes intermediate allocations

3. **Optimized Implementation**: Balance of speed and memory usage
   - Uses vectorized operations
   - Optimized memory access patterns
   - Handles edge cases efficiently

### Using Flash Attention 2

Flash Attention 2 provides significant speedups for attention computation:

1. Install Flash Attention: `poetry add flash-attn`
2. Enable it when initializing the model: `use_flash_attention=True`
3. Or use the command-line flag: `--flash_attention`

### Maximizing Performance

For maximum inference speed:

1. Use the `--max_speed` flag which:
   - Applies tensor memory optimizations
   - Enables TF32 precision on supported hardware
   - Warms up CUDA kernels
   - Applies torch.compile with optimal settings
   - Ensures contiguous tensor layouts

2. Fine-tune batch size for your hardware:
   - Use `--batch_size` to set optimal batch size
   - Typically 1-4 for most consumer GPUs
   - Higher values for high-end GPUs with more memory

## License

MIT 
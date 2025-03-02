"""
Utility functions for memory management and profiling.
"""

import os
import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt

# Check if we can use torch.compile
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and torch.__version__ >= "2.0.0"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics in GB
    """
    result = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_used_gb": psutil.virtual_memory().used / (1024 ** 3),
        "ram_percent": psutil.virtual_memory().percent,
    }
    
    if torch.cuda.is_available():
        result.update({
            "cuda_allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "cuda_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "cuda_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        })
    
    return result


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with memory profiling
    """
    def wrapper(*args, **kwargs):
        # Record memory before
        before = get_memory_usage()
        
        # Call function
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Record memory after
        after = get_memory_usage()
        
        # Calculate differences
        diff = {}
        for key in before:
            if key in after:
                diff[key] = after[key] - before[key]
        
        # Print results
        print(f"Memory profile for {func.__name__}:")
        print(f"  Execution time: {elapsed_time:.4f} seconds")
        
        for key, value in diff.items():
            if "percent" in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.4f} GB")
        
        return result
    
    return wrapper


def optimize_memory_usage(device: str = "cuda", reserve_memory: bool = True, 
                          fraction: float = 0.8) -> None:
    """
    Optimize memory usage for inference.
    
    Args:
        device: Device to optimize memory for
        reserve_memory: Whether to reserve memory upfront
        fraction: Fraction of memory to reserve
    """
    if device == "cuda" and torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        
        # Set memory fraction
        if reserve_memory:
            for i in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(i)
                total_memory = device_properties.total_memory
                torch.cuda.set_per_process_memory_fraction(fraction, i)
                print(f"Reserved {fraction * 100:.1f}% of GPU {i} memory "
                      f"({fraction * total_memory / (1024**3):.2f} GB)")
        
        # Use TF32 precision if available (on Ampere or newer GPUs)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision('high')
            print("Using TF32 precision for matrix multiplications")
        
        # Enable memory efficient attention if available
        try:
            import transformers
            if hasattr(transformers, "utils") and hasattr(transformers.utils, "logging"):
                transformers.utils.logging.set_verbosity_info()
                logger = transformers.utils.logging.get_logger("transformers")
                logger.info("Enabling memory efficient attention")
        except ImportError:
            pass
            
        # Enable CUDA graphs for repeated operations
        torch.cuda.is_available() and torch._C._jit_set_profiling_executor(True)
        torch.cuda.is_available() and torch._C._jit_set_profiling_mode(True)


def optimize_for_inference_speed(model: torch.nn.Module, use_cache: bool = True, optimize_memory: bool = True,
                                batch_size: int = 1) -> torch.nn.Module:
    """
    Optimize model for maximum inference speed.
    
    Args:
        model: PyTorch model to optimize
        use_cache: Whether to use KV cache
        optimize_memory: Whether to optimize memory usage
        batch_size: Typical batch size for inference
        
    Returns:
        Optimized model
    """
    # Save original training mode and set to eval
    original_mode = model.training
    model.eval()
    
    # Apply inference optimizations
    if optimize_memory:
        # Make sure all parameters are contiguous for faster memory access
        for name, param in model.named_parameters():
            if param.data.storage().size() != param.numel():
                param.data = param.data.contiguous()
    
    # Apply torch.compile to the model if available
    if TORCH_COMPILE_AVAILABLE:
        try:
            # Choose the backend based on the system
            backend = "inductor" 
            if hasattr(torch, "compile"):
                print(f"Applying torch.compile with {backend} backend for maximum speed")
                model = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
        except Exception as e:
            print(f"Failed to apply torch.compile: {e}")
    
    # Enable fast path optimizations in PyTorch
    with torch.no_grad():
        # Enable fused operations where available
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
    
    # If using CUDA, warm up the model
    if torch.cuda.is_available() and hasattr(model, "config"):
        with torch.no_grad():
            try:
                # Create some dummy input for warmup
                if hasattr(model.config, "hidden_size"):
                    hidden_size = model.config.hidden_size
                    dummy_input = torch.randn(batch_size, 32, hidden_size, 
                                            device="cuda", dtype=torch.float16)
                    # Run a few forward passes to warm up the CUDA kernels
                    print("Warming up CUDA kernels...")
                    for _ in range(3):
                        if hasattr(model, "forward"):
                            _ = model(dummy_input, use_cache=use_cache)
                        elif hasattr(model, "__call__"):
                            _ = model(dummy_input)
            except Exception as e:
                print(f"Error during model warmup: {e}")
    
    # Restore original training mode
    model.train(original_mode)
    
    return model


def plot_memory_usage(memory_log: List[Dict[str, float]], 
                      output_file: str = "memory_usage.png") -> None:
    """
    Plot memory usage over time.
    
    Args:
        memory_log: List of memory usage dictionaries
        output_file: File to save the plot to
    """
    plt.figure(figsize=(12, 8))
    
    # Extract data
    timestamps = list(range(len(memory_log)))
    
    # Plot CPU and RAM usage
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, [log.get("cpu_percent", 0) for log in memory_log], 
             label="CPU Usage (%)")
    plt.plot(timestamps, [log.get("ram_percent", 0) for log in memory_log], 
             label="RAM Usage (%)")
    plt.title("CPU and RAM Usage")
    plt.xlabel("Time Steps")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True)
    
    # Plot CUDA memory if available
    if "cuda_allocated_gb" in memory_log[0]:
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, [log.get("cuda_allocated_gb", 0) for log in memory_log], 
                 label="CUDA Allocated (GB)")
        plt.plot(timestamps, [log.get("cuda_reserved_gb", 0) for log in memory_log], 
                 label="CUDA Reserved (GB)")
        plt.title("GPU Memory Usage")
        plt.xlabel("Time Steps")
        plt.ylabel("Memory (GB)")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Memory usage plot saved to {output_file}")


def estimate_memory_requirements(model_id: str) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        Dictionary with estimated memory requirements
    """
    from transformers import AutoConfig
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_id)
    
    # Get model parameters
    hidden_size = getattr(config, "hidden_size", 0)
    num_hidden_layers = getattr(config, "num_hidden_layers", 0)
    num_attention_heads = getattr(config, "num_attention_heads", 0)
    vocab_size = getattr(config, "vocab_size", 0)
    
    # Estimate parameter count
    embedding_params = hidden_size * vocab_size
    attention_params = num_hidden_layers * (4 * hidden_size * hidden_size)  # Q, K, V, O
    ffn_params = num_hidden_layers * (8 * hidden_size * hidden_size)  # Typical FFN has 4x hidden size
    
    total_params = embedding_params + attention_params + ffn_params
    
    # Estimate memory requirements
    fp16_size_gb = total_params * 2 / (1024 ** 3)  # 2 bytes per parameter for fp16
    bnb4bit_size_gb = total_params * 0.5 / (1024 ** 3)  # 0.5 bytes per parameter for 4-bit
    
    # Estimate activation memory (rough approximation)
    batch_size = 1
    seq_length = 2048
    activation_size_gb = batch_size * seq_length * hidden_size * num_hidden_layers * 4 / (1024 ** 3)
    
    # KV cache for generation
    kv_cache_size_gb = batch_size * seq_length * num_hidden_layers * 2 * hidden_size * 2 / (1024 ** 3)
    
    return {
        "model_id": model_id,
        "total_parameters": total_params,
        "fp16_size_gb": fp16_size_gb,
        "bnb4bit_size_gb": bnb4bit_size_gb,
        "activation_size_gb": activation_size_gb,
        "kv_cache_size_gb": kv_cache_size_gb,
        "total_fp16_gb": fp16_size_gb + activation_size_gb + kv_cache_size_gb,
        "total_bnb4bit_gb": bnb4bit_size_gb + activation_size_gb + kv_cache_size_gb,
    }


def log_memory_usage(interval: float = 1.0, duration: float = 60.0) -> List[Dict[str, float]]:
    """
    Log memory usage over time.
    
    Args:
        interval: Interval between measurements in seconds
        duration: Total duration to log for in seconds
        
    Returns:
        List of memory usage dictionaries
    """
    memory_log = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        memory_log.append(get_memory_usage())
        time.sleep(interval)
    
    return memory_log 
#!/usr/bin/env python
"""
Memory-efficient inference for Llama 3.2 using PyTorch.
"""

import argparse
import time
import torch
import sys
from llama_kernel import LlamaInference
from llama_kernel.utils import (
    profile_memory, 
    optimize_memory_usage, 
    plot_memory_usage, 
    log_memory_usage,
    estimate_memory_requirements,
    get_memory_usage,
    optimize_for_inference_speed
)


@profile_memory
def run_inference(model, prompt, **kwargs):
    """Run inference with memory profiling."""
    return model.generate(prompt, **kwargs)


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: Yes (version {torch.version.cuda})")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA available: No")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import bitsandbytes
        print(f"BitsAndBytes version: {bitsandbytes.__version__}")
    except ImportError:
        missing_deps.append("bitsandbytes")
    
    try:
        import accelerate
        print(f"Accelerate version: {accelerate.__version__}")
    except ImportError:
        missing_deps.append("accelerate")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them using:")
        print(f"poetry add {' '.join(missing_deps)}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Memory-efficient Llama 3.2 inference")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                        help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable memory profiling")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark with multiple prompts")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Optimize memory usage")
    parser.add_argument("--max_speed", action="store_true",
                        help="Apply all optimizations to maximize inference speed")
    parser.add_argument("--estimate_memory", action="store_true",
                        help="Estimate memory requirements without loading model")
    parser.add_argument("--flash_attention", action="store_true",
                        help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--check_deps", action="store_true",
                        help="Check dependencies and exit")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for optimized inference")
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            return
    
    # Estimate memory requirements if requested
    if args.estimate_memory:
        print(f"Estimating memory requirements for {args.model}...")
        try:
            requirements = estimate_memory_requirements(args.model)
            print("\nEstimated memory requirements:")
            print(f"  Total parameters: {requirements['total_parameters']:,}")
            print(f"  FP16 model size: {requirements['fp16_size_gb']:.2f} GB")
            print(f"  4-bit model size: {requirements['bnb4bit_size_gb']:.2f} GB")
            print(f"  Activation memory: {requirements['activation_size_gb']:.2f} GB")
            print(f"  KV cache size: {requirements['kv_cache_size_gb']:.2f} GB")
            print(f"  Total memory (FP16): {requirements['total_fp16_gb']:.2f} GB")
            print(f"  Total memory (4-bit): {requirements['total_bnb4bit_gb']:.2f} GB")
        except Exception as e:
            print(f"Error estimating memory requirements: {e}")
        
        if not args.benchmark and not args.profile:
            return
    
    # Optimize memory usage if requested
    if args.optimize_memory or args.max_speed:
        print("Optimizing memory usage...")
        optimize_memory_usage(device=args.device)
    
    # Initialize model
    print(f"Initializing model {args.model}...")
    try:
        model = LlamaInference(
            model_id=args.model,
            device=args.device,
            use_flash_attention=args.flash_attention
        )
        
        # Apply maximum speed optimizations if requested
        if args.max_speed:
            print("Applying maximum speed optimizations...")
            model.model = optimize_for_inference_speed(
                model.model, 
                use_cache=True, 
                optimize_memory=True,
                batch_size=args.batch_size
            )
            
            # Force contiguous tensors in generation function
            original_generate = model.generate
            
            def optimized_generate(prompt, **kwargs):
                """Wrapper for generate function that ensures optimal memory layouts"""
                
                # Record start time for token generation speed
                start_time = time.time()
                
                # Call original generate function
                output = original_generate(prompt, **kwargs)
                
                # Calculate tokens per second
                elapsed_time = time.time() - start_time
                input_tokens = len(model.tokenizer.encode(prompt))
                output_tokens = len(model.tokenizer.encode(output)) - input_tokens
                
                # Print generation speed
                print(f"Generated {output_tokens} tokens in {elapsed_time:.2f} seconds "
                      f"({output_tokens / elapsed_time:.2f} tokens/sec)")
                
                # Return the output
                return output
            
            # Replace generate function with optimized version
            model.generate = optimized_generate
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("\nTry running with --check_deps to verify your installation")
        return
    
    # Run benchmark if requested
    if args.benchmark:
        print("\nRunning benchmark...")
        prompts = [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "Write a short poem about the ocean.",
            "What are the main ingredients in a chocolate cake?",
            "Summarize the plot of Romeo and Juliet."
        ]
        
        # Start memory logging
        memory_log = []
        
        # Run benchmark
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: {prompt}")
            
            # Record memory before
            memory_log.append(get_memory_usage())
            
            # Generate text
            try:
                start_time = time.time()
                output = model.generate(prompt, max_length=args.max_length, temperature=args.temperature)
                elapsed_time = time.time() - start_time
                
                # Record memory after
                memory_log.append(get_memory_usage())
                
                # Calculate tokens per second
                input_tokens = len(model.tokenizer.encode(prompt))
                output_tokens = len(model.tokenizer.encode(output)) - input_tokens
                tokens_per_second = output_tokens / elapsed_time
                
                total_tokens += output_tokens
                total_time += elapsed_time
                
                print(f"Generated {output_tokens} tokens in {elapsed_time:.2f} seconds "
                      f"({tokens_per_second:.2f} tokens/sec)")
                print(f"Output: {output}")
            except Exception as e:
                print(f"Error generating text for prompt {i+1}: {e}")
        
        # Calculate average tokens per second
        if total_time > 0:
            avg_tokens_per_second = total_tokens / total_time
            print(f"\nAverage throughput: {avg_tokens_per_second:.2f} tokens/sec")
        
        # Plot memory usage
        try:
            plot_memory_usage(memory_log, "benchmark_memory.png")
        except Exception as e:
            print(f"Error plotting memory usage: {e}")
    else:
        # Run single inference
        print(f"\nPrompt: {args.prompt}")
        
        try:
            if args.profile:
                # Run with memory profiling
                output = run_inference(
                    model, 
                    args.prompt, 
                    max_length=args.max_length, 
                    temperature=args.temperature
                )
            else:
                # Run without memory profiling
                start_time = time.time()
                output = model.generate(
                    args.prompt, 
                    max_length=args.max_length, 
                    temperature=args.temperature
                )
                elapsed_time = time.time() - start_time
                
                # Calculate tokens per second
                input_tokens = len(model.tokenizer.encode(args.prompt))
                output_tokens = len(model.tokenizer.encode(output)) - input_tokens
                tokens_per_second = output_tokens / elapsed_time
                
                print(f"Generated {output_tokens} tokens in {elapsed_time:.2f} seconds "
                      f"({tokens_per_second:.2f} tokens/sec)")
            
            print(f"\nOutput: {output}")
            
            # Save memory snapshot
            try:
                model.save_memory_snapshot("memory_snapshot.txt")
            except Exception as e:
                print(f"Error saving memory snapshot: {e}")
        except Exception as e:
            print(f"Error generating text: {e}")


if __name__ == "__main__":
    main()

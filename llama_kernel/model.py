"""
Model wrapper for efficient inference with Llama 3.2 using PyTorch.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils import logging
from typing import List, Optional, Union, Dict, Any, Tuple

from .pytorch_kernels import Linear4Bit
from .utils import get_memory_usage


class LlamaInference:
    """
    Memory-efficient inference for Llama 3.2 models using PyTorch.
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda",
        use_flash_attention: bool = True,
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        offload_folder: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Llama model with PyTorch for memory-efficient inference.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to load the model on ('cuda', 'cpu')
            use_flash_attention: Whether to use flash attention for faster inference
            max_memory: Maximum memory to use for each GPU
            offload_folder: Folder to offload weights to
            low_cpu_mem_usage: Whether to use low CPU memory usage when loading
            cache_dir: Directory to cache models
        """
        self.model_id = model_id
        self.device = device
        
        # Set up logging
        logging.set_verbosity_info()
        self.logger = logging.get_logger("transformers")
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer from {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        
        # Set pad token to eos token if not set
        if self.tokenizer.pad_token is None:
            self.logger.info("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Handle case where eos_token_id is a list
            if isinstance(self.tokenizer.eos_token_id, list) and len(self.tokenizer.eos_token_id) > 0:
                self.logger.info(f"Setting pad_token_id to first eos_token_id: {self.tokenizer.eos_token_id[0]}")
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id[0]
        
        # Load model configuration
        self.logger.info(f"Loading model configuration from {model_id}")
        self.config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        
        # Prepare device map
        device_map = None
        use_4bit = False  # Disable 4-bit quantization by default
        
        if device == "cuda" and torch.cuda.is_available():
            device_map = device
        elif device == "cpu":
            device_map = device
        else:
            self.logger.warning(f"Device {device} not available, falling back to CPU")
            device_map = "cpu"
            self.device = "cpu"
        
        try:
            # Load the model without quantization
            self.logger.info(f"Loading model from {model_id} without quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                load_in_4bit=False,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch.float16,
                max_memory=max_memory,
                offload_folder=offload_folder,
                cache_dir=cache_dir,
            )
        except Exception as e:
            self.logger.warning(f"Error loading model with device mapping: {e}")
            self.logger.info("Falling back to standard loading without device mapping")
            
            # Fallback to standard loading without device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=False,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
            
            # Move model to device manually
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
        
        # Enable flash attention if requested and available
        if use_flash_attention and hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "flash_attention_2"
            self.logger.info("Using Flash Attention 2 for faster inference")
        
        # Set up generation config
        if self.tokenizer.pad_token_id is None and self.model.config.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
    
    def _replace_linear_layers(self):
        """
        Replace nn.Linear layers with our custom Linear4Bit layers.
        """
        self.logger.info("Replacing linear layers with PyTorch 4-bit linear layers")
        
        # Count of replaced layers
        replaced_count = 0
        
        # Recursively replace linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get the parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                
                # Create a new PyTorch 4-bit linear layer
                try:
                    quantized_linear = Linear4Bit.from_float(module)
                    
                    # Replace the linear layer
                    setattr(parent, child_name, quantized_linear)
                    replaced_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to replace layer {name}: {e}")
        
        self.logger.info(f"Replaced {replaced_count} linear layers with PyTorch 4-bit linear layers")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt or list of prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text or list of generated texts
        """
        # Handle single prompt or list of prompts
        is_single_prompt = isinstance(prompt, str)
        prompts = [prompt] if is_single_prompt else prompt
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Reshape outputs if multiple sequences per prompt
        if num_return_sequences > 1:
            decoded_outputs = [
                decoded_outputs[i:i+num_return_sequences]
                for i in range(0, len(decoded_outputs), num_return_sequences)
            ]
        
        # Return single output or list of outputs
        if is_single_prompt and num_return_sequences == 1:
            return decoded_outputs[0]
        return decoded_outputs
    
    def embed(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Get embeddings for text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tensor of embeddings
        """
        # Handle single text or list of texts
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model.model(**inputs, output_hidden_states=True)
            # Use the last hidden state of the last token as the embedding
            embeddings = outputs.hidden_states[-1][:, -1, :]
        
        return embeddings
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for generation
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated texts
        """
        results = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_results = self.generate(batch_prompts, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def get_memory_usage(self):
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory usage statistics
        """
        return get_memory_usage()
    
    def save_memory_snapshot(self, filename: str = "memory_snapshot.txt"):
        """
        Save a snapshot of memory usage.
        
        Args:
            filename: Name of the file to save the snapshot to
        """
        if self.device == "cuda":
            # Get current memory usage
            current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            # Save to file
            with open(filename, "w") as f:
                f.write(f"Current memory usage: {current_memory:.2f} GB\n")
                f.write(f"Maximum memory usage: {max_memory:.2f} GB\n")
                
                # Log model size
                f.write("\nModel size breakdown:\n")
                total_params = 0
                for name, param in self.model.named_parameters():
                    param_size = param.numel() * param.element_size() / (1024 ** 2)  # MB
                    f.write(f"{name}: {param_size:.2f} MB\n")
                    total_params += param.numel()
                
                f.write(f"\nTotal parameters: {total_params:,}\n")
            
            self.logger.info(f"Memory snapshot saved to {filename}")
        else:
            self.logger.warning("Memory snapshot only available for CUDA devices")
    
    def clear_cache(self):
        """
        Clear CUDA cache to free up memory.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")
        else:
            self.logger.warning("Cache clearing only available for CUDA devices")
    
    def _check_bnb_compatibility(self):
        """
        Check if bitsandbytes is compatible with the current environment.
        
        Returns:
            bool: True if bitsandbytes is compatible, False otherwise
        """
        try:
            import bitsandbytes
            # Check if the version is compatible
            version = getattr(bitsandbytes, "__version__", "0.0.0")
            major, minor, patch = map(int, version.split(".")[:3])
            
            # Require at least version 0.41.0
            if major > 0 or (major == 0 and minor >= 41):
                return True
            else:
                self.logger.warning(f"bitsandbytes version {version} is too old, need at least 0.41.0")
                return False
        except ImportError:
            self.logger.warning("bitsandbytes not installed")
            return False
        except Exception as e:
            self.logger.warning(f"Error checking bitsandbytes compatibility: {e}")
            return False 
#!/usr/bin/env python
"""
An OpenAI-compatible API server for Llama Kernel. V0.7.6

V0.7.6 Changes:
- Resolved final Pylance type-checking warning by simplifying the `fixed_generate`
  monkeypatch to match its single-prompt usage within the server.
- The monkeypatch now unambiguously returns a single `str`, satisfying all
  type hints and ensuring type safety.
"""

import argparse
import time
import uuid
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Tuple
from types import MethodType

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- Dynamic Import for Transformers & Quantization ---
import transformers
try:
    from transformers import BitsAndBytesConfig
    BB_ACCELERATE_AVAILABLE = True
except ImportError:
    BB_ACCELERATE_AVAILABLE = False

# --- Core LlamaInference Import ---
from llama_kernel import LlamaInference
from llama_kernel.utils import get_memory_usage

# --- Globals and Logger ---
logger = logging.getLogger("uvicorn")
llama_model: Optional[LlamaInference] = None
server_args: Optional[argparse.Namespace] = None
request_logger: Optional['BaseLogger'] = None

# --- Constants ---
DEFAULT_LLAMA3_SYSTEM_PROMPT_PATTERN = re.compile(
    r"Cutting Knowledge Date: .*\nToday Date: .*\n\n"
)
SYSTEM_HEADER_START = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"

# --- MONKEYPATCH SECTION ---
# --- THIS IS THE FIX ---
def fixed_generate(
    self,
    prompt: str, # Simplified to only accept a single string, matching our usage
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs
) -> str: # Simplified to only return a single string, resolving the type conflict
    """
    A simplified and corrected monkeypatch for generate, tailored for the server's
    single-prompt use case. It unambiguously returns a single string.
    """
    # Tokenize the single prompt. No need for batch logic.
    inputs = self.tokenizer(prompt, return_tensors="pt")
    prompt_token_count = len(inputs.input_ids[0])
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    # Decode only the new tokens from the single output tensor
    new_tokens = outputs[0][prompt_token_count:]
    decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded_text


# --- Advanced Logging Classes ---
def log_vram_usage(logger_instance: Optional['BaseLogger'], stage: str):
    if isinstance(logger_instance, FidelityLogger) and torch.cuda.is_available():
        vram_stats = get_memory_usage()
        vram_data = {
            "cuda_allocated_gb": vram_stats.get("cuda_allocated_gb", 0),
            "cuda_reserved_gb": vram_stats.get("cuda_reserved_gb", 0),
            "cuda_max_allocated_gb": vram_stats.get("cuda_max_allocated_gb", 0),
        }
        logger_instance.log(stage, vram_data)

class BaseLogger:
    def log(self, stage: str, data: Any): pass

class FidelityLogger(BaseLogger):
    def log(self, stage: str, data: Any):
        header = f"--- STAGE: {stage} ---"
        content = ""
        if isinstance(data, (dict, list)): content = json.dumps(data, indent=2)
        elif isinstance(data, BaseModel): content = data.model_dump_json(indent=2)
        else: content = str(data)
        logger.info(f"\n{header}\n{content}")

class ChatLogger(BaseLogger):
    def __init__(self, limit: Optional[int]):
        self.limit = None if limit is None or limit < 0 else limit
    def log(self, prompt: str, response: str):
        log_str = "\n"
        try:
            parts = prompt.split('<|start_header_id|>')
            if len(parts) > 1:
                for part in parts[1:]:
                    if '<|end_header_id|>' in part and '<|eot_id|>' in part:
                        header, content_full = part.split('<|end_header_id|>\n\n', 1)
                        role, content = header.strip(), content_full.split('<|eot_id|>', 1)[0].strip()
                        if self.limit and len(content) > self.limit: content = content[:self.limit] + '...'
                        log_str += f"{role.capitalize()}: {content}\n"
        except Exception: pass
        response_content = response
        if self.limit and len(response_content) > self.limit: response_content = response_content[:self.limit] + '...'
        log_str += f"Assistant: {response_content}"
        logger.info(log_str)

def create_logger(args: argparse.Namespace) -> BaseLogger:
    if args.fidelity_log: return FidelityLogger()
    if args.chat_log is not None: return ChatLogger(args.chat_log)
    return BaseLogger()


# --- Pydantic Models ---
class ChatMessage(BaseModel): role: Optional[str] = None; content: Optional[str] | List[str] = None
class ChatCompletionRequest(BaseModel): model: str; messages: List[ChatMessage]; temperature: float = 0.7; max_tokens: int = 512; stream: bool = False; top_p: Optional[float] = None
class ChatCompletionResponseChoice(BaseModel): index: int; message: ChatMessage; finish_reason: str = "stop"
class UsageInfo(BaseModel): prompt_tokens: int; completion_tokens: int; total_tokens: int
class ChatCompletionResponse(BaseModel): id: str; object: str = "chat.completion"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[ChatCompletionResponseChoice]; usage: UsageInfo
class StreamDelta(BaseModel): content: Optional[str] | List[str] = None; role: Optional[str] = None
class StreamChoice(BaseModel): index: int; delta: StreamDelta; finish_reason: Optional[str] = None
class ChatCompletionStreamResponse(BaseModel): id: str; object: str = "chat.completion.chunk"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[StreamChoice]
class ModelCard(BaseModel): id: str; object: str = "model"; owned_by: str = "user"; context_window: int
class ModelList(BaseModel): object: str = "list"; data: List[ModelCard]

# --- Prompt Engineering ---
def prepare_chat_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """
    Groups consecutive messages of the same role into a single message entry.
    This function is now type-safe and correctly handles content that may be
    a string, a list of strings, or None.
    """
    if not messages:
        return []

    processed_messages = []
    
    # Initialize the first group safely, handling potential None values
    first_msg = messages[0]
    current_group = {
        'role': first_msg.role or "unknown", # Default role if None
        'content': ""
    }
    # Safely process the content of the first message
    if isinstance(first_msg.content, str):
        current_group['content'] = first_msg.content
    elif isinstance(first_msg.content, list):
        # Join list items with a newline, filtering out any non-string elements
        current_group['content'] = "\n".join(str(item) for item in first_msg.content)

    # Iterate through the rest of the messages
    for msg in messages[1:]:
        # Ensure the message has a role to compare against
        msg_role = msg.role or "unknown"
        
        # If the current message has the same role as the current group
        if msg_role == current_group['role']:
            # Append its content to the current group's content, handling different types
            if isinstance(msg.content, str):
                current_group['content'] += "\n" + msg.content
            elif isinstance(msg.content, list):
                current_group['content'] += "\n" + "\n".join(str(item) for item in msg.content)
            # If msg.content is None, we do nothing and effectively skip it
        else:
            # The role has changed, so the current group is complete.
            # Add the completed group to the processed list.
            processed_messages.append(current_group)
            
            # Start a new group with the current message, safely handling its content
            new_content = ""
            if isinstance(msg.content, str):
                new_content = msg.content
            elif isinstance(msg.content, list):
                new_content = "\n".join(str(item) for item in msg.content)
            
            current_group = {'role': msg_role, 'content': new_content}
    
    # Add the last group after the loop finishes
    processed_messages.append(current_group)
    
    return processed_messages


# --- Server Lifespan & Endpoints ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_model, server_args, request_logger
    server_args = get_server_args()
    assert server_args is not None
    request_logger = create_logger(server_args)
    logger.info("Server starting up...")

    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
    def patched_from_pretrained(*args, **kwargs):
        assert server_args is not None
        if server_args.load_in_4bit:
            logger.info("Monkeypatch: Intercepted from_pretrained. Applying 4-bit quantization config.")
            if not BB_ACCELERATE_AVAILABLE: raise RuntimeError("bitsandbytes/accelerate not available for 4-bit loading.")
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            kwargs['torch_dtype'] = torch.float16
            kwargs.pop('load_in_4bit', None)
        return original_from_pretrained(*args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained
    try:
        llama_model = LlamaInference(model_id=server_args.model, device=server_args.device, use_flash_attention=server_args.flash_attention)
        llama_model.generate = MethodType(fixed_generate, llama_model)
        logger.info(f"Model '{server_args.model}' loaded successfully.")
        if server_args.load_in_4bit: logger.info("Model is 4-bit quantized.")
        log_vram_usage(request_logger, "VRAM_AFTER_MODEL_LOAD")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model: {e}", exc_info=True)
        llama_model = None
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
        logger.info("Monkeypatch for from_pretrained has been restored.")
    yield
    logger.info("Server shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    if not llama_model: raise HTTPException(status_code=503, detail="Model not available.")
    assert server_args is not None
    model_id = server_args.model
    context_window = llama_model.model.config.max_position_embeddings
    response_data = ModelList(data=[ModelCard(id=model_id, context_window=context_window)])
    if request_logger: request_logger.log("MODEL_METADATA_RESPONSE", response_data)
    return response_data

async def perform_generation(request: ChatCompletionRequest) -> Tuple[str, str | List[str]]:
    if not llama_model: raise RuntimeError("Model not loaded")
    chat_history = prepare_chat_messages(request.messages)
    if request_logger: request_logger.log("PRE_TEMPLATE_MESSAGES", chat_history)
    prompt = llama_model.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    if prompt.startswith(SYSTEM_HEADER_START):
        prompt, num_replacements = DEFAULT_LLAMA3_SYSTEM_PROMPT_PATTERN.subn("", prompt)
        if num_replacements > 0: logger.warning(f"Removed default Llama 3 system prompt injection.")
    if request_logger: request_logger.log("FINAL_PROMPT", prompt)
    log_vram_usage(request_logger, "VRAM_BEFORE_GENERATION")
    try:
        response_content = llama_model.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=(request.top_p or 0.9)
        )
        log_vram_usage(request_logger, "VRAM_AFTER_GENERATION")
        if request_logger: request_logger.log("MODEL_RESPONSE", response_content)
        return prompt, response_content
    except Exception as e:
        logger.error(f"Error during model generation: {e}", exc_info=True)
        raise e

async def stream_generator(request: ChatCompletionRequest, response_id: str) -> AsyncGenerator[str, None]:
    try:
        _prompt, response_content = await perform_generation(request)
        chunk = ChatCompletionStreamResponse(id=response_id, model=request.model, choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content=response_content))])
        if request_logger: request_logger.log("OUTBOUND_CHUNK (content)", chunk)
        yield f"data: {chunk.model_dump_json()}\n\n"
        final_chunk = ChatCompletionStreamResponse(id=response_id, model=request.model, choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")])
        if request_logger: request_logger.log("OUTBOUND_CHUNK (finish)", final_chunk)
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_vram_usage(request_logger, "VRAM_AFTER_CACHE_CLEAR")

@app.post("/v1/chat/completions")
async def create_chat_completion(fastapi_request: Request):
    try:
        request_body = await fastapi_request.json()
        if request_logger: request_logger.log("INBOUND_REQUEST", request_body)
        request = ChatCompletionRequest.model_validate(request_body)
    except Exception as e: raise HTTPException(status_code=422, detail=f"Request validation error: {e}")
    if not llama_model: raise HTTPException(status_code=503, detail="Model is not available.")
    response_id = f"chatcmpl-{uuid.uuid4()}"
    if request.stream:
        return StreamingResponse(stream_generator(request, response_id), media_type="text/event-stream")
    else:
        try:
            prompt, response_content = await perform_generation(request)
            prompt_tokens_len = len(llama_model.tokenizer.encode(prompt))
            completion_tokens_len = len(llama_model.tokenizer.encode(response_content))
            response = ChatCompletionResponse(id=response_id, model=request.model, choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=response_content))], usage=UsageInfo(prompt_tokens=prompt_tokens_len, completion_tokens=completion_tokens_len, total_tokens=prompt_tokens_len + completion_tokens_len))
            if request_logger: request_logger.log("OUTBOUND_RESPONSE", response)
            return response
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            log_vram_usage(request_logger, "VRAM_AFTER_CACHE_CLEAR")

# --- Main Execution ---
def get_server_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for Llama Kernel")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention 2.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to.")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--fidelity-log", action="store_true", help="Log all 5 stages of request/response processing and VRAM usage.")
    log_group.add_argument("--chat-log", nargs='?', const=-1, type=int, default=None, help="Log a clean chat transcript.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the model with 4-bit quantization.")
    return parser.parse_args()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
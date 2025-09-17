#!/usr/bin/env python
"""
An OpenAI-compatible API server for Llama Kernel. V0.5

V0.5 Changes:
- Introduced a powerful RequestLogger for advanced log formatting.
- Added mutually exclusive command-line flags for log control:
  --raw-log: Compact, single-line JSON for requests.
  --indented-log: Pretty-printed JSON for requests.
  --formatted-log: Human-readable prompt/response blocks.
  --chat-log [chars]: A clean, truncated chat transcript view.
- Refactored generation logic to be cleaner and more reusable.
"""

import argparse
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from types import MethodType

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llama_kernel import LlamaInference

# --- Globals and Logger ---
logger = logging.getLogger("uvicorn")
llama_model: Optional[LlamaInference] = None
server_args: Optional[argparse.Namespace] = None
request_logger: 'RequestLogger' = None

# --- MONKEYPATCH SECTION ---
# (Same as V0.4, included for completeness)
def fixed_generate(
    self, prompt: Union[str, List[str]], max_new_tokens: int = 512, **kwargs
) -> Union[str, List[str]]:
    is_single_prompt = isinstance(prompt, str)
    prompts = [prompt] if is_single_prompt else prompt
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_token_counts = [len(x) for x in inputs.input_ids]
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    decoded_outputs = []
    for i, output_tensor in enumerate(outputs):
        new_tokens = output_tensor[prompt_token_counts[i]:]
        decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_outputs.append(decoded_text)
    return decoded_outputs[0] if is_single_prompt else decoded_outputs


# --- Advanced Logging Class ---
class RequestLogger:
    """Handles different logging formats based on server arguments."""
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def log_request_body(self, body: dict):
        if self.args.raw_log:
            logger.info(f"RAW REQUEST: {json.dumps(body, separators=(',', ':'))}")
        elif self.args.indented_log:
            logger.info(f"INDENTED REQUEST:\n{json.dumps(body, indent=2)}")

    def log_turn(self, prompt: str, response: str):
        if self.args.formatted_log:
            self._log_formatted(prompt, response)
        elif self.args.chat_log is not None:
            self._log_chat(prompt, response)

    def _log_formatted(self, prompt: str, response: str):
        log_str = (
            "\n--- FORMATTED LOG ---\n"
            f"PROMPT:\n{prompt}\n\n"
            f"RESPONSE:\n{response}\n"
            "--------------------"
        )
        logger.info(log_str)

    def _log_chat(self, prompt: str, response: str):
        limit = self.args.chat_log
        log_str = "\n--- CHAT LOG ---\n"
        
        # Parse the templated prompt back into a chat-like format
        message_parts = prompt.split('<|start_header_id|>')[1:]
        for part in message_parts:
            try:
                header, content_full = part.split('<|end_header_id|>\n\n', 1)
                content = content_full.split('<|eot_id|>')[0].strip()
                if limit and len(content) > limit:
                    content = content[:limit] + '...'
                log_str += f"{header.capitalize()}: {content}\n"
            except ValueError:
                continue # Skip malformed parts

        # Add the assistant's response
        response_content = response
        if limit and len(response_content) > limit:
            response_content = response_content[:limit] + '...'
        log_str += f"Assistant: {response_content}\n"
        log_str += "----------------"
        logger.info(log_str)


# --- Pydantic Models (unchanged) ---
class ChatMessage(BaseModel): role: str; content: str
class ChatCompletionRequest(BaseModel): model: str; messages: List[ChatMessage]; temperature: float = 0.7; max_tokens: int = 512; stream: bool = False; top_p: Optional[float] = None
class ChatCompletionResponseChoice(BaseModel): index: int; message: ChatMessage; finish_reason: str = "stop"
class UsageInfo(BaseModel): prompt_tokens: int; completion_tokens: int; total_tokens: int
class ChatCompletionResponse(BaseModel): id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}"); object: str = "chat.completion"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[ChatCompletionResponseChoice]; usage: UsageInfo
class StreamDelta(BaseModel): content: Optional[str] = None; role: Optional[str] = None
class StreamChoice(BaseModel): index: int; delta: StreamDelta; finish_reason: Optional[str] = None
class ChatCompletionStreamResponse(BaseModel): id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}"); object: str = "chat.completion.chunk"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[StreamChoice]
class ModelCard(BaseModel): id: str; object: str = "model"; owned_by: str = "user"; context_window: int
class ModelList(BaseModel): object: str = "list"; data: List[ModelCard]


# --- Prompt Engineering ---
def prepare_chat_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    filtered_messages = []
    for msg in messages:
        if msg.role == "system" and "Write Assistant's next reply" in msg.content:
            continue
        filtered_messages.append({"role": msg.role, "content": msg.content})
    return filtered_messages


# --- Server Lifespan & Endpoints ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_model, server_args, request_logger
    server_args = get_server_args()
    request_logger = RequestLogger(server_args)
    logger.info("Server starting up...")
    logger.info(f"Loading model: {server_args.model} on device: {server_args.device}")
    try:
        llama_model = LlamaInference(model_id=server_args.model, device=server_args.device, use_flash_attention=server_args.flash_attention)
        llama_model.generate = MethodType(fixed_generate, llama_model)
        logger.info("Model loaded and monkeypatched.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model: {e}", exc_info=True)
    yield
    logger.info("Server shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    if not llama_model: raise HTTPException(status_code=503, detail="Model not available.")
    model_id = server_args.model
    context_window = llama_model.model.config.max_position_embeddings
    return ModelList(data=[ModelCard(id=model_id, context_window=context_window)])

async def perform_generation(request: ChatCompletionRequest) -> (str, str):
    if not llama_model: raise RuntimeError("Model not loaded")
    chat_history = prepare_chat_messages(request.messages)
    prompt = llama_model.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    response_content = llama_model.generate(prompt=prompt, max_new_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p)
    return prompt, response_content

async def stream_generator(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    prompt, response_content = await perform_generation(request)
    request_logger.log_turn(prompt, response_content)
    
    chunk = ChatCompletionStreamResponse(model=request.model, choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content=response_content))])
    yield f"data: {chunk.model_dump_json()}\n\n"
    final_chunk = ChatCompletionStreamResponse(model=request.model, choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")])
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(fastapi_request: Request):
    try:
        request_body = await fastapi_request.json()
        request_logger.log_request_body(request_body)
        request = ChatCompletionRequest.model_validate(request_body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Request validation error: {e}")

    if not llama_model: raise HTTPException(status_code=503, detail="Model is not available.")

    if request.stream:
        return StreamingResponse(stream_generator(request), media_type="text/event-stream")
    else:
        prompt, response_content = await perform_generation(request)
        request_logger.log_turn(prompt, response_content)

        prompt_tokens_len = len(llama_model.tokenizer.encode(prompt))
        completion_tokens_len = len(llama_model.tokenizer.encode(response_content))
        return ChatCompletionResponse(
            model=request.model,
            choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=response_content))],
            usage=UsageInfo(prompt_tokens=prompt_tokens_len, completion_tokens=completion_tokens_len, total_tokens=prompt_tokens_len + completion_tokens_len)
        )

# --- Main Execution ---
def get_server_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for Llama Kernel")
    # Server and Model args
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention 2.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to.")
    
    # Logging args - mutually exclusive
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--raw-log", action="store_true", help="Log raw, single-line JSON requests.")
    log_group.add_argument("--indented-log", action="store_true", help="Log indented JSON requests.")
    log_group.add_argument("--formatted-log", action="store_true", help="Log human-readable prompt/response blocks.")
    log_group.add_argument("--chat-log", nargs='?', const=300, type=int, default=None,
                           help="Log a clean chat transcript. Optionally specify max characters per message (default: 300).")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Args are parsed in lifespan now
    uvicorn.run(app, host="127.0.0.1", port=5000)
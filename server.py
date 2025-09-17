#!/usr/bin/env python
"""
An OpenAI-compatible API server for Llama Kernel. V2.

This server loads a Llama 3.2 model using the LlamaInference class
and serves it through an API endpoint that mimics the OpenAI
chat completions API.

V2 Changes:
- Added "fake" streaming support to be compatible with clients like SillyTavern.
- Implemented the /v1/models endpoint.
- Correctly formats prompts using the tokenizer's chat template.
- Integrated detailed request logging for easier debugging.
"""

import argparse
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import the core inference class from your project
from llama_kernel import LlamaInference

# --- Globals for holding the model and arguments ---
llama_model: Optional[LlamaInference] = None
server_args: Optional[argparse.Namespace] = None

# Set up a logger that integrates with Uvicorn
logger = logging.getLogger("uvicorn")


# --- Pydantic Models for OpenAI API Compatibility ---

# Models for Request
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    # Add other common fields to prevent validation errors from clients
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

# Models for Non-Streaming Response
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

# Models for Streaming Response
class StreamDelta(BaseModel):
    content: str
    role: Optional[str] = None

class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]

# Models for /v1/models endpoint
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "user"
    permission: list = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# --- Server Lifespan Management (Model Loading/Unloading) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_model, server_args
    logger.info("Server starting up...")
    logger.info(f"Loading model: {server_args.model} on device: {server_args.device}")

    try:
        llama_model = LlamaInference(
            model_id=server_args.model,
            device=server_args.device,
            use_flash_attention=server_args.flash_attention,
        )
        logger.info("Model loaded successfully. Server is ready to accept requests.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model. Error: {e}", exc_info=True)
        llama_model = None

    yield

    logger.info("Server shutting down...")
    llama_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleanup complete. Goodbye!")


# --- FastAPI App and Endpoint Definitions ---

app = FastAPI(lifespan=lifespan)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Provides a list of available models, mimicking the OpenAI API."""
    model_id = server_args.model if server_args else "unknown"
    return ModelList(data=[ModelCard(id=model_id)])

async def stream_generator(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """
    A "fake" streaming generator. It generates the full response and then
    yields it in the OpenAI SSE format in a single chunk.
    """
    # 1. Generate the full response
    if llama_model is None:
        # This case should ideally be caught before calling the generator
        raise RuntimeError("Model not loaded")

    # Correctly apply the chat template
    chat_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt = llama_model.tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    generated_text = llama_model.generate(
        prompt=prompt,
        max_length=request.max_tokens,
        temperature=request.temperature,
    )

    # The model might include the prompt in its output, so we find where the real response starts
    # This is common with apply_chat_template
    assistant_start_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if assistant_start_marker in generated_text:
        response_content = generated_text.split(assistant_start_marker, 1)[-1]
    else:
        response_content = generated_text

    # 2. Create the streaming response chunk
    chunk = ChatCompletionStreamResponse(
        model=request.model,
        choices=[StreamChoice(index=0, delta=StreamDelta(content=response_content, role="assistant"))]
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    # 3. Send the final [DONE] message
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(fastapi_request: Request):
    """
    Handles chat completion requests, supporting both streaming and non-streaming.
    """
    # --- START of Debugging Block ---
    try:
        # Read the raw JSON body from the request
        request_body = await fastapi_request.json()
        logger.info(f"Received raw request body:\n{json.dumps(request_body, indent=2)}")

        # Parse the JSON into our Pydantic model
        request = ChatCompletionRequest.model_validate(request_body)
        logger.info(f"Successfully parsed request for model: {request.model}")

    except Exception as e:
        logger.error(f"Error parsing request body: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Request validation error: {e}")
    # --- END of Debugging Block ---

    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    # Handle streaming vs. non-streaming
    if request.stream:
        return StreamingResponse(
            stream_generator(request),
            media_type="text/event-stream"
        )
    else:
        # Correctly apply the chat template
        chat_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = llama_model.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )

        logger.info(f"Formatted prompt (first 200 chars): '{prompt[:200]}...'")

        generated_text = llama_model.generate(
            prompt=prompt,
            max_length=request.max_tokens,
            temperature=request.temperature,
        )

        assistant_start_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if assistant_start_marker in generated_text:
            response_content = generated_text.split(assistant_start_marker, 1)[-1]
        else:
            response_content = generated_text

        logger.info(f"Generated response (first 100 chars): '{response_content[:100]}...'")

        tokenizer = llama_model.tokenizer
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + completion_tokens

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content)
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )

def get_server_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for Llama Kernel")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model ID to load.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu).")
    parser.add_argument("--flash_attention", action="store_true",
                        help="Use Flash Attention 2 for faster inference.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind the server to.")
    return parser.parse_args()

if __name__ == "__main__":
    server_args = get_server_args()
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port
    )
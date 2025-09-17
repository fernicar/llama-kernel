#!/usr/bin/env python
"""
An OpenAI-compatible API server for Llama Kernel.

This server loads a Llama 3.2 model using the LlamaInference class
and serves it through an API endpoint that mimics the OpenAI
chat completions API.

Features:
- Loads the model only once at startup.
- Provides a `/v1/chat/completions` endpoint.
- Accepts and responds with JSON data in OpenAI's format.
- Handles prompt formatting from the `messages` array.
- Calculates and returns token usage statistics.
"""

import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import the core inference class from your project
from llama_kernel import LlamaInference

# --- Globals for holding the model and arguments ---
llama_model: Optional[LlamaInference] = None
server_args: Optional[argparse.Namespace] = None

# --- Pydantic Models for OpenAI API Compatibility ---

class ChatMessage(BaseModel):
    """A message in the chat."""
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """Request model for the chat completions endpoint."""
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False  # Streaming is not supported in this basic server

class ChatCompletionResponseChoice(BaseModel):
    """Response choice model."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """The final response model for the chat completions endpoint."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


# --- Server Lifespan Management (Model Loading/Unloading) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown events for the FastAPI application.
    This is where we load the model into memory.
    """
    global llama_model, server_args
    print("Server starting up...")
    print(f"Loading model: {server_args.model} on device: {server_args.device}")

    try:
        # This is the "monkeypatch" or, more accurately, the context
        # where the model is loaded once and kept in memory.
        llama_model = LlamaInference(
            model_id=server_args.model,
            device=server_args.device,
            use_flash_attention=server_args.flash_attention,
        )
        print("Model loaded successfully. Server is ready to accept requests.")
    except Exception as e:
        print(f"FATAL: Failed to load model. Error: {e}")
        llama_model = None # Ensure it's None if loading fails

    yield

    # --- Shutdown logic ---
    print("Server shutting down...")
    llama_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleanup complete. Goodbye!")

# --- FastAPI App and Endpoint Definition ---

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Handles the chat completion requests.
    """
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this server.")

    # Format the prompt from the messages list.
    # A simple approach is to concatenate them. For instruct models,
    # often just the last user message is enough.
    prompt = "\n".join([msg.content for msg in request.messages if msg.role == "user"])
    if not prompt:
        # Fallback if there are no user messages
        prompt = request.messages[-1].content if request.messages else ""

    print(f"Received prompt: '{prompt[:100]}...'")

    # Generate text using the loaded model
    generated_text = llama_model.generate(
        prompt=prompt,
        max_length=request.max_tokens,
        temperature=request.temperature,
    )
    
    # Strip the original prompt from the generated output if it's included
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].lstrip()

    print(f"Generated response: '{generated_text[:100]}...'")

    # Calculate token usage
    tokenizer = llama_model.tokenizer
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(generated_text))
    total_tokens = prompt_tokens + completion_tokens

    # Construct and return the OpenAI-compatible response
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text)
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    )

def get_server_args():
    """
    Parses command-line arguments for the server, reusing relevant
    arguments from your original main.py.
    """
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
    
    # Start the Uvicorn server
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port
    )
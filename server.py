#!/usr/bin/env python
"""
An OpenAI-compatible API server for Llama Kernel. V0.4

V0.4 Changes:
- Monkeypatches LlamaInference.generate at runtime to avoid editing base code.
- Correctly handles the `max_tokens` parameter by using `max_new_tokens`.
- Adds comprehensive logging for each conversational turn (prompt -> response).
- Exposes the model's context window size in the /v1/models endpoint.
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

# --- MONKEYPATCH SECTION ---

def fixed_generate(
    self,
    prompt: Union[str, List[str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    **kwargs
) -> Union[str, List[str]]:
    """
    A monkeypatched version of LlamaInference.generate that correctly handles
    new token generation and returns only the completion.
    This function will replace the original method on the LlamaInference instance.
    """
    is_single_prompt = isinstance(prompt, str)
    prompts = [prompt] if is_single_prompt else prompt
    
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_token_counts = [len(x) for x in inputs.input_ids]
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Use the correct parameter
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
    
    decoded_outputs = []
    for i, output_tensor in enumerate(outputs):
        prompt_len = prompt_token_counts[i]
        new_tokens = output_tensor[prompt_len:]
        decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_outputs.append(decoded_text)
    
    if num_return_sequences > 1:
        # This logic is for multi-sequence returns, kept for completeness
        reshaped = [
            decoded_outputs[i:i+num_return_sequences]
            for i in range(0, len(decoded_outputs), num_return_sequences)
        ]
        return reshaped if not is_single_prompt else reshaped[0]

    return decoded_outputs[0] if is_single_prompt else decoded_outputs


# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False
    top_p: Optional[float] = None

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

class StreamDelta(BaseModel):
    content: Optional[str] = None
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

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "user"
    context_window: int

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# --- Prompt Engineering ---
def prepare_chat_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    filtered_messages = []
    for msg in messages:
        if msg.role == "system" and "Write Assistant's next reply" in msg.content:
            logger.info("Filtered out generic SillyTavern system prompt.")
            continue
        filtered_messages.append({"role": msg.role, "content": msg.content})
    return filtered_messages

# --- Server Lifespan & Endpoints ---
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
        # --- APPLYING THE MONKEYPATCH ---
        # This replaces the instance's generate method with our corrected version
        llama_model.generate = MethodType(fixed_generate, llama_model)
        logger.info("Model loaded successfully and generate() method has been monkeypatched.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load model. Error: {e}", exc_info=True)
        llama_model = None
    yield
    logger.info("Server shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    if llama_model is None or not hasattr(llama_model, 'model'):
        raise HTTPException(status_code=503, detail="Model is not available.")
    
    model_id = server_args.model
    context_window = llama_model.model.config.max_position_embeddings
    
    return ModelList(data=[ModelCard(id=model_id, context_window=context_window)])

async def perform_generation(request: ChatCompletionRequest) -> (str, str):
    if llama_model is None: raise RuntimeError("Model not loaded")

    chat_history = prepare_chat_messages(request.messages)
    prompt = llama_model.tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=True
    )
    
    response_content = llama_model.generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return prompt, response_content

async def stream_generator(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    prompt, response_content = await perform_generation(request)
    
    # --- ENHANCED LOGGING ---
    logger.info(f"PROMPT:\n{prompt}...")
    logger.info(f"RESPONSE:\n{response_content}...")

    chunk = ChatCompletionStreamResponse(
        model=request.model,
        choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content=response_content))]
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    final_chunk = ChatCompletionStreamResponse(
        model=request.model,
        choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(fastapi_request: Request):
    try:
        request_body = await fastapi_request.json()
        request = ChatCompletionRequest.model_validate(request_body)
    except Exception as e:
        logger.error(f"Error parsing request body: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Request validation error: {e}")

    if llama_model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    if request.stream:
        return StreamingResponse(stream_generator(request), media_type="text/event-stream")
    else:
        prompt, response_content = await perform_generation(request)

        # --- ENHANCED LOGGING ---
        logger.info(f"PROMPT:\n{prompt}...")
        logger.info(f"RESPONSE:\n{response_content}...")

        prompt_tokens_len = len(llama_model.tokenizer.encode(prompt))
        completion_tokens_len = len(llama_model.tokenizer.encode(response_content))
        
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0, message=ChatMessage(role="assistant", content=response_content)
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_tokens_len + completion_tokens_len,
            )
        )

# --- Main Execution ---
def get_server_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for Llama Kernel")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention 2.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to.")
    return parser.parse_args()

if __name__ == "__main__":
    server_args = get_server_args()
    uvicorn.run(app, host=server_args.host, port=server_args.port)
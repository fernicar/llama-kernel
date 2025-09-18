#!/usr/bin/env python
"""
An OpenAI-compatible API server for Llama Kernel. V0.6.3

V0.6.3 Changes:
- Implemented robust grouping of consecutive messages by role in prepare_chat_messages.
- Ensured the regex for removing default Llama 3 system prompts is applied correctly after template processing.
- All previous fixes (streaming IDs, logging fidelity, context window) are maintained.
"""

import argparse
import time
import uuid
import json
import logging
import re
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
request_logger: 'BaseLogger' = None

# --- Constants ---
# Regex to find and remove the default Llama 3 system prompt, which includes arbitrary dates.
DEFAULT_LLAMA3_SYSTEM_PROMPT_PATTERN = re.compile(
    r"Cutting Knowledge Date: .*\nToday Date: .*\n\n"
)
SYSTEM_HEADER_START = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
USER_TURN_START = "<|start_header_id|>user<|end_header_id|>"
ASSISTANT_TURN_START = "<|start_header_id|>assistant<|end_header_id|>"
EOT_ID = "<|eot_id|>"

# --- MONKEYPATCH SECTION ---
# This corrected generate method correctly handles max_new_tokens and returns only the completion.
def fixed_generate(
    self, prompt: Union[str, List[str]], max_new_tokens: int = 512, **kwargs
) -> Union[str, List[str]]:
    is_single_prompt = isinstance(prompt, str)
    prompts = [prompt] if is_single_prompt else prompt
    
    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    prompt_token_counts = [len(x) for x in inputs.input_ids]
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    decoded_outputs = []
    for i, output_tensor in enumerate(outputs):
        new_tokens = output_tensor[prompt_token_counts[i]:]
        decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_outputs.append(decoded_text)
    
    return decoded_outputs[0] if is_single_prompt else decoded_outputs


# --- Advanced Logging Classes ---
class BaseLogger:
    def log(self, stage: str, data: Any): pass

class FidelityLogger(BaseLogger):
    def log(self, stage: str, data: Any):
        header = f"--- STAGE: {stage} ---"
        content = ""
        if isinstance(data, (dict, list)):
            content = json.dumps(data, indent=2)
        elif isinstance(data, BaseModel):
            content = data.model_dump_json(indent=2)
        else:
            content = str(data)
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
                        header_content, rest_of_part = part.split('<|end_header_id|>\n\n', 1)
                        role = header_content.strip()
                        content_full = rest_of_part.split('<|eot_id|>', 1)[0]
                        
                        content = content_full.strip()
                        if self.limit and len(content) > self.limit:
                            content = content[:self.limit] + '...'
                        log_str += f"{role.capitalize()}: {content}\n"
                    else:
                         if self.limit and len(part) > self.limit:
                            log_str += f"Unknown: {part[:self.limit]}...\n"
                         else:
                            log_str += f"Unknown: {part}\n"
        except Exception as e:
            logger.error(f"Error parsing prompt for chat log: {e}", exc_info=True)
            if self.limit and len(prompt) > self.limit:
                log_str += f"Raw Prompt: {prompt[:self.limit]}...\n"
            else:
                log_str += f"Raw Prompt: {prompt}\n"

        response_content = response
        if self.limit and len(response_content) > self.limit:
            response_content = response_content[:self.limit] + '...'
        log_str += f"Assistant: {response_content}"
        logger.info(log_str)

def create_logger(args: argparse.Namespace) -> BaseLogger:
    if args.fidelity_log:
        return FidelityLogger()
    if args.chat_log is not None:
        return ChatLogger(args.chat_log)
    return BaseLogger()


# --- Pydantic Models ---
class ChatMessage(BaseModel): role: str; content: str
class ChatCompletionRequest(BaseModel): model: str; messages: List[ChatMessage]; temperature: float = 0.7; max_tokens: int = 512; stream: bool = False; top_p: Optional[float] = None
class ChatCompletionResponseChoice(BaseModel): index: int; message: ChatMessage; finish_reason: str = "stop"
class UsageInfo(BaseModel): prompt_tokens: int; completion_tokens: int; total_tokens: int
class ChatCompletionResponse(BaseModel): id: str; object: str = "chat.completion"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[ChatCompletionResponseChoice]; usage: UsageInfo
class StreamDelta(BaseModel): content: Optional[str] = None; role: Optional[str] = None
class StreamChoice(BaseModel): index: int; delta: StreamDelta; finish_reason: Optional[str] = None
class ChatCompletionStreamResponse(BaseModel): id: str; object: str = "chat.completion.chunk"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[StreamChoice]
class ModelCard(BaseModel): id: str; object: str = "model"; owned_by: str = "user"; context_window: int
class ModelList(BaseModel): object: str = "list"; data: List[ModelCard]


# --- Prompt Engineering ---
def prepare_chat_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """
    Groups consecutive messages of the same role into a single message entry.
    This helps the tokenizer's chat template process them more cleanly.
    """
    if not messages:
        return []

    processed_messages = []
    # Start with the first message as the initial group
    current_group = {'role': messages[0].role, 'content': messages[0].content}

    # Iterate through the rest of the messages
    for msg in messages[1:]:
        # If the current message has the same role as the current group
        if msg.role == current_group['role']:
            # Append its content to the current group's content, separated by a newline
            current_group['content'] += "\n" + msg.content
        else:
            # The role has changed, so the current group is complete.
            # Add the completed group to the processed list.
            processed_messages.append(current_group)
            # Start a new group with the current message.
            current_group = {'role': msg.role, 'content': msg.content}
    
    # Add the last group after the loop finishes
    processed_messages.append(current_group)
    
    return processed_messages


# --- Server Lifespan & Endpoints ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_model, server_args, request_logger
    server_args = get_server_args()
    request_logger = create_logger(server_args)
    logger.info("Server starting up...")
    try:
        llama_model = LlamaInference(model_id=server_args.model, device=server_args.device, use_flash_attention=server_args.flash_attention)
        llama_model.generate = MethodType(fixed_generate, llama_model)
        logger.info(f"Model '{server_args.model}' loaded and generate() method monkeypatched.")
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
    
    response_data = ModelList(data=[ModelCard(id=model_id, context_window=context_window)])
    
    request_logger.log("MODEL_METADATA_RESPONSE", response_data)
    
    return response_data

async def perform_generation(request: ChatCompletionRequest, response_id: str) -> (str, str):
    """
    Performs the generation logic, returning prompt and response content.
    Includes steps for cleaning the prompt and generating the response.
    """
    if not llama_model: raise RuntimeError("Model not loaded")
    
    chat_history = prepare_chat_messages(request.messages)
    request_logger.log("PRE_TEMPLATE_MESSAGES", chat_history)

    prompt = llama_model.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    
    # --- FIX FOR UNWANTED DEFAULT Llama 3 SYSTEM PROMPT ---
    # Remove the default system prompt pattern using regex substitution if it exists.
    if prompt.startswith(SYSTEM_HEADER_START):
        prompt, num_replacements = DEFAULT_LLAMA3_SYSTEM_PROMPT_PATTERN.subn("", prompt)
        if num_replacements > 0:
            logger.warning(f"Removed default Llama 3 system prompt injection ({num_replacements} instance(s)) from FINAL_PROMPT.")
    
    request_logger.log("FINAL_PROMPT", prompt)

    response_content = llama_model.generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    request_logger.log("MODEL_RESPONSE", response_content)
    
    return prompt, response_content

async def stream_generator(request: ChatCompletionRequest, response_id: str) -> AsyncGenerator[str, None]:
    prompt, response_content = await perform_generation(request, response_id)
    
    # Content Chunk
    chunk = ChatCompletionStreamResponse(
        id=response_id,
        model=request.model,
        choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content=response_content))]
    )
    request_logger.log("OUTBOUND_CHUNK (content)", chunk)
    yield f"data: {chunk.model_dump_json()}\n\n"
    
    # Finish Chunk
    final_chunk = ChatCompletionStreamResponse(
        id=response_id,
        model=request.model,
        choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")]
    )
    request_logger.log("OUTBOUND_CHUNK (finish)", final_chunk)
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(fastapi_request: Request):
    try:
        request_body = await fastapi_request.json()
        request_logger.log("INBOUND_REQUEST", request_body)
        request = ChatCompletionRequest.model_validate(request_body)
    except Exception as e:
        logger.error(f"Request validation error: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Request validation error: {e}")

    if not llama_model: raise HTTPException(status_code=503, detail="Model is not available.")

    response_id = f"chatcmpl-{uuid.uuid4()}"
    logger.info(f"Generated response ID: {response_id}")

    if request.stream:
        return StreamingResponse(stream_generator(request, response_id), media_type="text/event-stream")
    else:
        prompt, response_content = await perform_generation(request, response_id)
        
        prompt_tokens_len = len(llama_model.tokenizer.encode(prompt))
        completion_tokens_len = len(llama_model.tokenizer.encode(response_content))
        
        response = ChatCompletionResponse(
            id=response_id,
            model=request.model,
            choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=response_content))],
            usage=UsageInfo(prompt_tokens=prompt_tokens_len, completion_tokens=completion_tokens_len, total_tokens=prompt_tokens_len + completion_tokens_len)
        )
        request_logger.log("OUTBOUND_RESPONSE", response)
        return response

# --- Main Execution ---
def get_server_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible server for Llama Kernel")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention 2.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to.")
    
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--fidelity-log", action="store_true", help="Log all 5 stages of request/response processing.")
    log_group.add_argument("--chat-log", nargs='?', const=-1, type=int, default=None,
                           help="Log a clean chat transcript. Optionally specify max characters (default: no limit).")
    return parser.parse_args()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
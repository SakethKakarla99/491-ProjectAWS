import os
import anyio
from typing import Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

os.environ.setdefault("SPOTTER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
os.environ.setdefault("DEVICE_MAP", "cpu")
os.environ.setdefault("LOAD_IN_4BIT", "0")

import hybrid_orchestrator
import supabase_service
from supabase_client import get_user_from_token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_up():
    print("Warming up orchestrator + RAG...")
    await anyio.to_thread.run_sync(hybrid_orchestrator.get_orchestrator)
    print("Warmup complete.")


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


@app.get("/history/{thread_id}")
async def get_history(thread_id: str, request: Request):
    """Return messages for a thread (authenticated users only)."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {"messages": []}
    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)
    if not user:
        return {"messages": []}
    messages = supabase_service.get_history_for_context(thread_id, max_messages=50)
    return {"messages": messages}


@app.get("/threads/latest")
async def get_latest_thread(request: Request):
    """Return the user's most recent thread id, or null if none."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {"thread_id": None}
    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)
    if not user:
        return {"thread_id": None}
    threads = supabase_service.list_threads(user.id, limit=1)
    if threads:
        return {"thread_id": threads[0]["id"]}
    return {"thread_id": None}


@app.get("/threads")
async def list_threads(request: Request):
    """Return all conversation threads for the authenticated user."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {"threads": []}
    token = auth_header.split(" ", 1)[1]
    user = get_user_from_token(token)
    if not user:
        return {"threads": []}
    threads = supabase_service.list_threads(user.id, limit=50)
    return {"threads": threads}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # --- 1. Identify user from JWT (optional — works without login too) ---
    user_id = None
    thread_id = req.thread_id

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        user = get_user_from_token(token)
        if user:
            user_id = user.id

    # --- 2. Get or create conversation thread (auto-title from first message) ---
    if user_id:
        if thread_id:
            thread = supabase_service.get_or_create_thread(user_id, thread_id)
        else:
            title = req.message[:60] + ("…" if len(req.message) > 60 else "")
            thread = supabase_service.create_thread(user_id, title=title)
        thread_id = thread["id"]

    # --- 3. Load user profile and message history ---
    profile = supabase_service.get_profile(user_id) if user_id else None
    history = []
    if thread_id:
        history = supabase_service.get_history_for_context(thread_id, max_messages=10)

    # --- 4. Save user message ---
    if thread_id:
        supabase_service.save_message(thread_id, role="user", content=req.message)

    # --- 5. Run the model ---
    answer = await anyio.to_thread.run_sync(
        lambda: hybrid_orchestrator.smart_answer(req.message, history=history, profile=profile)
    )

    # --- 6. Save assistant response ---
    if thread_id:
        supabase_service.save_message(thread_id, role="assistant", content=answer)

    return {"answer": answer, "thread_id": thread_id}

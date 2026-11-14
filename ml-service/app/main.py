"""Run from /srv/Hakaton: uvicorn app.main:app --host 127.0.0.1 --port 8001"""
import logging
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Query, Request
from .classifier import load_model, classify
from .generator import generate_draft
from .schemas import GenerateRequest, GenerateResponse, Draft

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=5.0)) as client:
        app.state.http = client
        yield

app = FastAPI(title="Avito Draft Generator", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health(request: Request):
    return {"status": "ok" if getattr(request.app.state, "model", None) is not None else "not_ready"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest, request: Request, generate_text: bool = Query(True)):
    ids = classify(request.app.state.model, payload.description, payload.mcId)
    drafts = []
    for mid in ids:
        title, text = (None, None)
        if generate_text:
            title, text = await generate_draft(request.app.state.http, payload.description, mid)
        drafts.append(Draft(mcId=mid, title=title, text=text))
    return GenerateResponse(drafts=drafts)

"""Vercel entry: the same trained model, behind a small JSON API."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import app as demo

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class CheckIn(BaseModel):
    text: str = ""


@app.get("/api/health")
@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/check")
@app.post("/check")
def check(body: CheckIn) -> dict:
    verdict, details = demo.check_message(body.text)
    return {"verdict": verdict, "details": details}

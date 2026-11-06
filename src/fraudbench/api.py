from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .inference import predict_record

app = FastAPI(title="Fraud Benchmark API", version="0.1.0")


class Transaction(BaseModel):
    features: dict[str, Any]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Fraud benchmark API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction) -> dict:
    return predict_record(transaction.features)

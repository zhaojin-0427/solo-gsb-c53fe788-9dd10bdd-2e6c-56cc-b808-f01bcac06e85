"""FastAPI application exposing the chromatographic quantification service."""

from __future__ import annotations

import math

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .solver import quantify


def _finite(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("value must be a finite number")
    return float(value)


class Standard(BaseModel):
    times: list[float] = Field(
        description="Strictly increasing time axis of the pure standard")
    intensities: list[float] = Field(
        description="Non-negative pure-standard intensity trace")
    concentration: float = Field(
        gt=0, description="Known positive concentration of the pure standard")

    @field_validator("times", "intensities")
    @classmethod
    def _all_finite(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(x) for x in v):
            raise ValueError("all values must be finite")
        return v

    @field_validator("concentration")
    @classmethod
    def _conc_finite(cls, v: float) -> float:
        return _finite(v)


class QuantifyRequest(BaseModel):
    standards: list[Standard] = Field(
        min_length=1, max_length=6,
        description="1..6 pure standards (quantitative channel templates)")
    reference_times: list[float] = Field(
        description="Numbered internal-standard times in the reference channel; "
                    "must be strictly increasing")
    candidate_times: list[float] = Field(
        description="Unnumbered candidate internal-standard times in the sample "
                    "channel (may include missed ISs / distractor peaks); "
                    "must be non-decreasing")
    sample_times: list[float] = Field(
        description="Strictly increasing sample time grid")
    sample_intensities: list[float] = Field(
        description="Sample quantitative-channel intensities (>= 0); values >= "
                    "saturation are treated as saturated points")
    saturation: float = Field(gt=0, description="Detector saturation value")
    tolerance: float = Field(gt=0, description="Allowed per-point residual")
    slope_range: list[float] = Field(
        min_length=2, max_length=2,
        description="[low, high] allowed reference/sample timing slope, "
                    "requires 0 < low <= high")

    @field_validator("reference_times", "candidate_times",
                     "sample_times", "sample_intensities")
    @classmethod
    def _finite_lists(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(x) for x in v):
            raise ValueError("all values must be finite")
        return v

    @field_validator("saturation", "tolerance")
    @classmethod
    def _finite_scalars(cls, v: float) -> float:
        return _finite(v)

    @field_validator("slope_range")
    @classmethod
    def _finite_pair(cls, v: list[float]) -> list[float]:
        if any(not math.isfinite(x) for x in v):
            raise ValueError("slope_range values must be finite")
        return v


app = FastAPI(
    title="Chromatography Quantification API",
    version="1.0.0",
    description="Certified multi-component quantification with shared "
                "piecewise-linear internal-standard time alignment.",
)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    # Malformed bodies (wrong types, shape, NaN/Infinity, ...) surface as a
    # structured invalid_input status rather than an opaque 422/500.
    return JSONResponse(
        status_code=422,
        content={"status": "invalid_input",
                 "error": "request schema validation failed",
                 "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def _unexpected_handler(request: Request, exc: Exception):
    # JSON decode errors (e.g. bare NaN/Infinity tokens) and any other
    # unexpected parsing problem must still yield a structured status rather
    # than an HTML 500 page.
    return JSONResponse(
        status_code=400,
        content={"status": "invalid_input",
                 "error": f"could not parse request: {exc}"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/quantify")
async def quantify_endpoint(request: QuantifyRequest):
    payload = request.model_dump()
    # The solver is CPU-bound; run it outside the event loop thread.
    result = await run_in_threadpool(quantify, payload)
    # Defensive: the solver only ever emits finite results, but guarantee a
    # JSON-compliant response even if an unexpected error slips through.
    return result

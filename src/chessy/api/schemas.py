"""Strict browser-facing request schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateGameRequest(StrictModel):
    model_id: str = Field(min_length=1, max_length=100)
    color: Literal["white", "black", "random"] = "white"
    time_control: Literal["untimed", "3+2", "5+0", "10+0", "15+10"] = "untimed"
    profile: Literal["fast", "normal", "deep"] = "normal"
    feedback_opt_in: bool = False


class FeedbackRequest(StrictModel):
    confirm: Literal[True]


class MovePayload(StrictModel):
    uci: str = Field(pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$", max_length=5)


class EmptyPayload(StrictModel):
    pass


class ClientEnvelope(StrictModel):
    version: Literal["play-ws-v1"]
    type: Literal["move", "resign", "offer_draw", "ping"]
    payload: dict[str, object] = Field(default_factory=dict)

"""LangGraph agent workflow: lane preprocess → backend infer → validate → package."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph


class ProfileAgentState(TypedDict, total=False):
    audio: np.ndarray
    sr: int
    lane: str
    backend: str
    profile: dict[str, Any] | None
    errors: list[str]
    hitl_required: bool


def _preprocess(state: ProfileAgentState) -> ProfileAgentState:
    audio = np.asarray(state["audio"], dtype=np.float32)
    return {
        **state,
        "audio": audio,
        "sr": int(state.get("sr", 16_000)),
        "lane": state.get("lane", "clean"),
        "backend": state.get("backend", "wav2vec2-heads"),
        "errors": list(state.get("errors") or []),
        "hitl_required": bool(state.get("hitl_required", False)),
    }


def _infer(state: ProfileAgentState) -> ProfileAgentState:
    from .profile import build_profile

    profile = build_profile(
        state["audio"],
        sr=state["sr"],
        lane=state["lane"],
        backend=state["backend"],
    )
    return {**state, "profile": profile.to_dict()}


def _validate(state: ProfileAgentState) -> ProfileAgentState:
    errors = list(state.get("errors") or [])
    profile = state.get("profile") or {}
    hitl = bool(state.get("hitl_required", False))
    if profile.get("schema_version") != "1.0":
        errors.append("schema_version mismatch")
        hitl = True
    duration = float(profile.get("duration_s") or 0.0)
    if duration <= 0.0:
        errors.append("empty audio")
        hitl = True
    return {**state, "errors": errors, "hitl_required": hitl}


def _package(state: ProfileAgentState) -> ProfileAgentState:
    return state


def build_profile_agent_graph():
    """Checkpointed stage graph: preprocess → infer → validate → package."""
    graph = StateGraph(ProfileAgentState)
    graph.add_node("preprocess", _preprocess)
    graph.add_node("infer", _infer)
    graph.add_node("validate", _validate)
    graph.add_node("package", _package)
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "infer")
    graph.add_edge("infer", "validate")
    graph.add_edge("validate", "package")
    graph.add_edge("package", END)
    return graph.compile()


def run_profile_agent(
    audio: np.ndarray,
    *,
    sr: int = 16_000,
    lane: str = "clean",
    backend: str = "wav2vec2-heads",
) -> ProfileAgentState:
    app = build_profile_agent_graph()
    return app.invoke(
        {
            "audio": np.asarray(audio, dtype=np.float32),
            "sr": sr,
            "lane": lane,
            "backend": backend,
            "profile": None,
            "errors": [],
            "hitl_required": False,
        }
    )

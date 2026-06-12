"""Smoke test for the Graph-of-Trace backend.

Runs without any LLM API key by monkeypatching the node-builder, so it
exercises the full write path (config resolution -> path template -> file
lock -> atomic write -> redundant-parent dedupe) deterministically.

Run:
    python -m pytest tests/test_smoke.py        # if pytest is installed
    python tests/test_smoke.py                  # plain runner (no pytest needed)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Make the repo root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run() -> None:
    tmp = tempfile.mkdtemp(prefix="got_smoke_")

    # Point output at a temp dir BEFORE importing the writer/config consumers.
    os.environ["GOT_SMOKE_BASE"] = tmp

    from Monitor.config import parser as cfg_parser

    # Force an agent-agnostic temp output location regardless of config.yaml.
    _orig_get_output = cfg_parser.get_output_config

    def _patched_output(cfg):
        return {
            "base_dir": tmp,
            "path_template": "{base_dir}/{project_name}/{session_id}/got.json",
        }

    cfg_parser.get_output_config = _patched_output

    # Avoid any network/LLM call: stub the node builder with a deterministic node.
    from Monitor import steps_llm

    async def _fake_build_nodes(*, session_id, subtask, artifacts, steps):
        return [
            {
                "id": "N002",
                "title": subtask.get("title", ""),
                "description": subtask.get("description", ""),
                "parents": [{"id": "N001", "relation": "necessitated_by"}],
                "artifacts": artifacts,
            }
        ]

    steps_llm.build_nodes = _fake_build_nodes

    from Monitor.got_writer import write_got_from_build_trace, _resolve_got_path

    project_name = "smoke proj/../x"   # exercise sanitization
    session_id = "sess-1"

    payload = {
        "project": {"name": project_name},
        "session": {"id": session_id},
        "subtask": {
            "title": "Train baseline model",
            "description": "Train ResNet-18 baseline (seed=42).",
        },
        "artifacts": [{"path": "metrics.json", "type": "json"}],
    }

    res = asyncio.run(
        write_got_from_build_trace(
            project_name=project_name,
            session_id=session_id,
            payload=payload,
        )
    )

    assert res["status"] == "ok", res
    assert res["primary_node_id"] == "N002", res

    got_path = _resolve_got_path(project_name, session_id)
    assert got_path.exists(), f"got.json not written at {got_path}"

    # Sanitization: '..' must not escape base_dir.
    assert str(got_path).startswith(tmp), got_path
    assert ".." not in got_path.parts

    data = json.loads(got_path.read_text(encoding="utf-8"))
    ids = [n["id"] for n in data["nodes"]]
    assert ids[0] == "N001", "root node missing"
    assert "N002" in ids, "appended node missing"
    assert data["meta"]["project_name"] == project_name
    assert data["meta"]["session_id"] == session_id

    cfg_parser.get_output_config = _orig_get_output
    print(f"OK smoke test passed -> {got_path}")


def test_smoke():
    _run()


if __name__ == "__main__":
    _run()

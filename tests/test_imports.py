"""Import checks: the MCP entrypoint and tool module must import cleanly
after the cleanup (no dead imports, package structure intact)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_import_entrypoint():
    import tool  # noqa: F401
    import server  # noqa: F401

    assert hasattr(tool, "build_trace")
    assert hasattr(server, "mcp")


def test_import_package():
    import Monitor  # noqa: F401
    from Monitor.got_writer import write_got_from_build_trace  # noqa: F401
    from Monitor.steps_llm import build_nodes  # noqa: F401
    from Monitor.config.parser import get_output_config  # noqa: F401
    from Monitor.adapter.registry import get_chat_adapter  # noqa: F401


if __name__ == "__main__":
    test_import_entrypoint()
    test_import_package()
    print("OK imports passed")

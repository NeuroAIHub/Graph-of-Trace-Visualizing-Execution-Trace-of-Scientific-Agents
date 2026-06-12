"""get_output_config must honor GOT_OUTPUT_* env overrides (used by scripts/start.sh)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Monitor.config.parser import get_output_config  # noqa: E402


def test_env_overrides_win():
    os.environ["GOT_OUTPUT_BASE_DIR"] = "/tmp/got_test_base"
    os.environ["GOT_OUTPUT_PATH_TEMPLATE"] = "{base_dir}/got.json"
    try:
        oc = get_output_config({"output": {"base_dir": "~/ignored", "path_template": "x"}})
        assert oc["base_dir"] == "/tmp/got_test_base"
        assert oc["path_template"] == "{base_dir}/got.json"
    finally:
        del os.environ["GOT_OUTPUT_BASE_DIR"]
        del os.environ["GOT_OUTPUT_PATH_TEMPLATE"]


def test_config_used_when_no_env():
    oc = get_output_config({"output": {"base_dir": "/data/x", "path_template": "{base_dir}/g.json"}})
    assert oc["base_dir"] == "/data/x"


def test_defaults_when_empty():
    oc = get_output_config({})
    assert oc["base_dir"].endswith(".graph_of_trace")
    assert "{project_name}" in oc["path_template"]


if __name__ == "__main__":
    test_env_overrides_win()
    test_config_used_when_no_env()
    test_defaults_when_empty()
    print("OK output config env override")

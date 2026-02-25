"""Minimal step graph helpers for steps_llm: StepNode and node relationship analysis.

No file I/O, no STEPS directory, no skeleton nodes. Used only to build execution_order
and parent_child from a steps dict (meta + nodes) for LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StepNode:
    id: str
    parent: Optional[List[str]]
    description: str
    input: Optional[str] = None
    output: Optional[str] = None
    status: str = "待开始"
    note: str = ""
    file_path: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "parent": self.parent,
            "description": self.description,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "note": self.note,
        }
        if self.file_path is not None:
            data["file_path"] = self.file_path
        return {k: v for k, v in data.items() if v is not None}


def _calculate_execution_order(
    nodes: List[Dict[str, Any]], parent_child: Dict[str, List[str]]
) -> List[str]:
    """Topological order from parent_child (Kahn)."""
    all_node_ids = {n.get("id") for n in nodes if n.get("id")}
    in_degree = {nid: 0 for nid in all_node_ids}
    for parent_id, child_ids in parent_child.items():
        for cid in child_ids:
            if cid in in_degree:
                in_degree[cid] += 1
    queue = [nid for nid, d in in_degree.items() if d == 0]
    result: List[str] = []
    while queue:
        current = queue.pop(0)
        result.append(current)
        for cid in parent_child.get(current, []):
            if cid in in_degree:
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    queue.append(cid)
    if len(result) < len(all_node_ids):
        remaining = sorted(all_node_ids - set(result))
        result.extend(remaining)
    return result


def analyze_node_relationships(steps: Dict[str, Any]) -> Dict[str, Any]:
    """From steps (meta + nodes) build parent_child, child_parent, execution_order.

    No file or STEPS path; steps_dict is only meta + nodes. Dependencies left empty.
    """
    nodes = steps.get("nodes", [])
    parent_child: Dict[str, List[str]] = {}
    child_parent: Dict[str, List[str]] = {}

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        # Support both: parent = [id, ...] (legacy) and parents = [{id, relation, ...}, ...] (got.json)
        parent_ids: List[str] = []
        if "parents" in node and isinstance(node["parents"], list):
            for p in node["parents"]:
                if isinstance(p, dict) and p.get("id"):
                    parent_ids.append(str(p["id"]).strip())
                elif isinstance(p, str) and p.strip():
                    parent_ids.append(p.strip())
        elif "parent" in node:
            raw = node["parent"]
            if isinstance(raw, str):
                raw = [raw]
            if isinstance(raw, list):
                for pid in raw:
                    if pid and isinstance(pid, str):
                        parent_ids.append(pid.strip())
        for pid in parent_ids:
            if pid:
                parent_child.setdefault(pid, []).append(node_id)
                child_parent.setdefault(node_id, []).append(pid)

    execution_order = _calculate_execution_order(nodes, parent_child)
    return {
        "parent_child": parent_child,
        "child_parent": child_parent,
        "dependencies": {},
        "execution_order": execution_order,
    }

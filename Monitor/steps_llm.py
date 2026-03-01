from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .adapter.registry import get_chat_adapter
log = logging.getLogger("mcp_tools.monitor")


_ID_RE = re.compile(r"^N(\d+)$")
_ROOT_RESERVED = {"N001"}


def _next_node_id(existing_ids: List[str]) -> str:
    max_n = 0
    for s in existing_ids:
        if not isinstance(s, str):
            continue
        m = _ID_RE.match(s.strip())
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass
    return f"N{max_n + 1:03d}"


def _extract_json(text: str) -> Any:
    """Extract JSON from LLM response; handles markdown code blocks and trailing junk."""
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty response")
    # Strip markdown code block: ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        lines = s.split("\n")
        # First line is ``` or ```json
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        s = "\n".join(lines[start:end])
    s = s.strip()
    # Try to find a complete JSON array if there is leading/trailing text
    start_idx = s.find("[")
    if start_idx >= 0:
        depth = 0
        for i in range(start_idx, len(s)):
            c = s[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    s = s[start_idx : i + 1]
                    break
    return json.loads(s)


async def build_nodes(
    *,
    session_id: str,
    subtask: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    steps: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate **frontend GoT nodes** from a subtask (title/description) plus artifacts."""

    title = subtask.get("title")
    description = subtask.get("description")
    if not isinstance(title, str) or not title.strip():
        return []
    if not isinstance(description, str) or not description.strip():
        return []

    adapter = get_chat_adapter()

    current_nodes = steps.get("nodes", [])
    existing_node_ids = [n.get("id") for n in current_nodes if n.get("id")]
    allowed_parent_ids = set([x for x in existing_node_ids if isinstance(x, str)])
    project_name = steps.get("meta", {}).get("project_name")

    compact_subtask = {
        "title": title.strip(),
        "description": description.strip(),
        "depends_on": subtask.get("depends_on"),
        "status": subtask.get("status"),
    }

    compact_artifacts: List[Dict[str, Any]] = []
    for a in artifacts or []:
        if not isinstance(a, dict):
            continue
        p = a.get("path")
        t = a.get("type")
        if isinstance(p, str) and p.strip() and isinstance(t, str) and t.strip():
            compact_artifacts.append({"path": p.strip(), "type": t.strip()})

    system_hint = (
        """
        You are a strict information extractor. Convert the provided subtask summary into one or more Graph-of-Trace (GoT) DAG nodes.
        Node Definition:
        A node represents a research-relevant operation that contributes
        to the experimental, analytical, or infrastructural progression
        of the research.

        A node MUST:

        - Represent a research-level action or a foundational
        experimental setup step.
        - Produce a meaningful outcome, configuration,
        artifact, or decision.
        - Be necessary for reproducibility, experimentation,
        analysis, or reasoning.

        The following ARE valid nodes:

        - Installing core frameworks (e.g., PyTorch, CUDA)
        - Setting up experimental environments
        - Implementing model architectures
        - Acquiring datasets
        - Running experiments
        - Performing analysis
        - Drawing conclusions
        - Negative but informative experimental results

        The following MUST NOT form nodes:

        - Minor debugging or typo fixes
        - Re-running unchanged configurations without new findings
        - Trivial code refactoring
        - Maintenance operations without research-level impact

        Node Boundary and Splitting Rules:
        Split actions into separate nodes IF AND ONLY IF:
        1. One action explicitly depends on the result of another.
        2. Each action has independent research-level semantic significance.
        3. The text describes sequential reasoning stages
        (e.g., experiment → analysis → conclusion).
        4. Parallel experiments or variants are executed independently.

        Do NOT split when:
        - The actions form a single conceptual research step.
        - The description reflects unified execution without semantic separation.
        - The step is purely technical maintenance.

        Parent Selection:
        - Each node MUST have at least one parent.
        - Parents represent logical research justification, NOT chronological execution order.
        - You MUST select parent ids from `existing_node_ids` (or from ids you declare in the same output array).
        - Use `subtask.depends_on` (if provided) as the primary evidence for selecting parent(s).
        - A parent MUST correspond to a research-level result or decision without which the current node would not make semantic sense.
        - Sequential order alone does NOT justify parent-child relation.
        - Attach ONLY the minimal direct prerequisite(s). Do NOT attach redundant or transitive ancestors.
        - Parallel experiments MUST share the same parent and MUST NOT be chained unless explicit dependency exists.
        - Before assigning a parent, apply this test:
        If the proposed parent did not exist,
        would this node still make semantic sense in the research narrative?
        - If YES → do NOT attach it.
        - If NO → attach it.
        - If no suitable parent exists,
        attach the node to a special root node with id: "N001" (constraints.root_node_id).
        Do NOT fabricate dependencies.

        Artifacts Requirements:
        Each node MUST include an artifacts list (may be empty only if
        no concrete file artifact exists).
        Artifacts represent the verifiable output of the node.
        Rules:
        - Execution, visualization, analysis, and conclusion nodes MUST produce concrete, inspectable artifacts.
        - Setup or literature nodes MAY have empty artifacts if no file-level output exists.
        
        Examples (Definitive Reference)
        These examples define the expected granularity and dependency structure.
        Follow them strictly.

        Example 1 — Sequential dependency
        Input:
        "Trained the proposed model and computed accuracy."
        Output:
        Node A:
        title: Trained the proposed model
        parents: [{id: "P1", relation: "necessitated_by"}]
        Node B:
        title: Computed accuracy of the trained model
        parents: [{id: "A", relation: "necessitated_by"}]

        Example 2 — Parallel experiments or hypotheses
        Input:
        "Ran baseline model A and baseline model B."
        Output:
        Node A:
        title: Ran baseline model A
        parents: [{id: "P1", relation: "necessitated_by"}]
        Node B:
        title: Ran baseline model B
        parents: [{id: "P1", relation: "necessitated_by"}]


        Example 3 — Multi-stage reasoning
        Input:
        "Evaluated the model, analyzed errors, and concluded that it overfits."
        Output:
        Node A:
        title: Evaluated the model
        parents: [{id: "P1", relation: "necessitated_by"}]
        Node B:
        title: Analyzed errors
        parents: [{id: "A", relation: "necessitated_by"}]
        Node C:
        title: Concluded that the model overfits
        parents: [{id: "B", relation: "necessitated_by"}]


        Example 4 — Negative but meaningful result
        Input:
        "Tested the proposed approach but observed worse performance."
        Output:
        Node A:
        title: Tested the proposed approach and observed inferior performance
        parents: [{id: "P1", relation: "necessitated_by"}]

        Example 5 — Engineering maintenance (excluded)
        Input:
        "Fixed a bug and reran the experiment."
        Output:
        NO NODE

        Example 6 — Literature and research gap
        Input:
        "Surveyed related literature and identified research gaps."
        Output:
        Node A:
        title: Surveyed related literature
        parents: [{id: "N001", relation: "necessitated_by"}]
        Node B:
        title: Identified research gaps
        parents: [{id: "A", relation: "necessitated_by"}]
        """
    )

    user_prompt = {
        "existing_node_ids": existing_node_ids,
        "existing_nodes": current_nodes,
        "existing_nodes_note": "You MUST use only these existing node ids when selecting parents (or ids you declare in the same output array).",
        "subtask": compact_subtask,
        "artifacts": compact_artifacts,
        "constraints": {
            "root_node_id": "N001",
            "relation_allowlist": [
                "necessitated_by"
            ],
            "artifacts_rule": "Do NOT override artifacts: output node artifacts should reflect the model's output (it may copy the input artifacts if appropriate).",
        },
        "output_schema": {
            "node": {
                "id": "string (required, unique in existing_node_ids and in this output array; do NOT reuse existing ids)",
                "title": "string (required)",
                "description": "string (required)",
                "status": "string (optional)",
                "parents": [
                    {
                        "id": "string (required: existing_node_ids OR id of a node earlier in this output array)",
                        "relation": "string (required; must be one of constraints.relation_allowlist)",
                        "explanation": "string (optional)",
                    }
                ],
                "artifacts": [{"path": "string", "type": "string"}],
            }
        },
    }

    prompt = (
        system_hint
        + "\nHere is the JSON input:\n"
        + json.dumps(user_prompt, ensure_ascii=False, indent=2)
    )

    log.info(
        "got_llm subtask->nodes calling model session=%s desc_chars=%d artifacts=%d existing_nodes=%d",
        session_id,
        len(description),
        len(compact_artifacts),
        len(existing_node_ids),
    )
    log.debug("got_llm subtask->nodes prompt=%s", prompt)

    raw = await adapter.chat(prompt)
    if raw is None:
        raw = ""
    raw = (raw or "").strip()
    log.info("got_llm subtask->nodes model returned chars=%d", len(raw))
    log.debug("got_llm subtask->nodes raw=%s", raw)
    if not raw:
        log.warning("got_llm subtask->nodes empty model response")
        return []

    try:
        data = _extract_json(raw)
    except Exception:
        log.exception("got_llm subtask->nodes failed to parse JSON")
        return []

    if not isinstance(data, list):
        log.warning("got_llm subtask->nodes model returned non-array type=%s", type(data).__name__)
        return []

    # Build a stable id remap for this batch.
    # If the model repeats an existing id (or repeats within the batch), we replace it with a new N### id.
    used_ids = set(allowed_parent_ids) | set(_ROOT_RESERVED)
    id_remap: Dict[str, str] = {}

    for it in data:
        if not isinstance(it, dict):
            continue
        raw = it.get("id")
        if not isinstance(raw, str) or not raw.strip():
            continue
        old = raw.strip()
        if old in id_remap:
            it["id"] = id_remap[old]
            continue
        if old in used_ids or old in _ROOT_RESERVED:
            new_id = _next_node_id(list(used_ids))
            while new_id in used_ids:
                new_id = _next_node_id(list(used_ids))
            id_remap[old] = new_id
            it["id"] = new_id
            used_ids.add(new_id)
        else:
            used_ids.add(old)

    valid_parent_ids = set(allowed_parent_ids) | {it.get("id") for it in data if isinstance(it, dict) and isinstance(it.get("id"), str)}

    nodes: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            log.debug("got_llm subtask->nodes skipping non-dict item")
            continue
        node_id_raw = item.get("id")
        if not isinstance(node_id_raw, str) or not node_id_raw.strip():
            log.warning("got_llm subtask->nodes skipping item with missing or empty id")
            continue
        node_id = node_id_raw.strip()
        if node_id in allowed_parent_ids or any(n.get("id") == node_id for n in nodes):
            log.warning("got_llm subtask->nodes skipping duplicate id=%s", node_id)
            continue
        item["id"] = node_id

        parents = item.get("parents")
        if isinstance(parents, list):
            cleaned_parents: List[Dict[str, Any]] = []
            for p in parents:
                if not isinstance(p, dict):
                    continue
                pid = p.get("id")
                rel = p.get("relation")
                if isinstance(pid, str) and isinstance(rel, str) and rel.strip():
                    pid = pid.strip()
                    pid = id_remap.get(pid, pid)
                    rel = rel.strip()
                    if rel not in {"necessitated_by"}:
                        continue
                    if pid in valid_parent_ids:
                        cp: Dict[str, Any] = {"id": pid, "relation": rel}
                        if isinstance(p.get("explanation"), str) and p["explanation"].strip():
                            cp["explanation"] = p["explanation"].strip()
                        cleaned_parents.append(cp)
            if cleaned_parents:
                item["parents"] = cleaned_parents
            else:
                item.pop("parents", None)
        else:
            item.pop("parents", None)

        # Require at least one parent so DAG stays connected.
        # If the model output has no valid parents, attach to the configured root.
        if not item.get("parents"):
            item["parents"] = [{"id": "N001", "relation": "necessitated_by", "explanation": "Fallback: no valid parent in model output; attach to root."}]
        # Ensure title/description present and non-empty
        if not (item.get("title") and str(item.get("title", "")).strip()):
            item["title"] = node_id
        if not (item.get("description") and str(item.get("description", "")).strip()):
            item["description"] = ""

        # Drop unknown keys to avoid malformed nodes (e.g., model output accidentally emits parent-like fields at top-level)
        allowed_keys = {"id", "title", "description", "status", "parents", "artifacts"}
        item = {k: v for k, v in item.items() if k in allowed_keys}

        nodes.append(item)

    log.info("got_llm subtask->nodes accepted nodes=%d from model array len=%d", len(nodes), len(data))
    return nodes



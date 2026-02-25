from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .adapter.registry import get_chat_adapter
log = logging.getLogger("mcp_tools.monitor")


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


async def build_llm_nodes_from_events(
    session_id: str,
    events: List[Dict[str, Any]],
    steps: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate **frontend GoT nodes** from factual events.

    Output schema (strict):
      [{ id: string, title?: string, parents?: [{id: string, relation: string, explanation?: string}], ... }]

    Notes:
    - Must not invent node IDs for parents; can only reference existing node IDs.
    - Node IDs you create MUST NOT collide with existing_node_ids.
    """
    if not events:
        return []

    adapter = get_chat_adapter()

    compact_events: List[Dict[str, Any]] = []
    for e in events:
        compact_events.append(
            {
                "event_id": e.get("event_id"),
                "seq_no": e.get("seq_no"),
                "tool_name": e.get("tool_name"),
                "command": e.get("command"),
                "cwd": e.get("cwd"),
                "exit_code": e.get("exit_code"),
                "security_risk": e.get("security_risk"),
            }
        )

    current_nodes = steps.get("nodes", [])
    existing_node_ids = [n.get("id") for n in current_nodes if n.get("id")]
    allowed_parent_ids = set([x for x in existing_node_ids if isinstance(x, str)])
    current_focus = steps.get("meta", {}).get("current_focus")
    project_name = steps.get("meta", {}).get("project_name")

    system_hint = (
            '''
            You are a strict information extractor. Convert factual tool execution events into a Graph-of-Thought (GoT) DAG.

            HARD CONSTRAINTS (MUST NOT VIOLATE):
            1) Use ONLY facts present in the provided events. Do NOT invent commands, paths, tools, outputs, files, or intentions.
            2) Output MUST be a pure JSON array (no markdown, no prose).
            3) Each node MUST have a unique id (string) that does NOT collide with any id in existing_node_ids or other new ids in this output.
            4) Each node MUST have a non-empty, human-readable title.
            5) Each node MUST have a non-empty, factual description.
            6) Each node MUST have parents (array) with at least ONE parent.
            7) Parent links MUST use the field:
            parents: [{ id, relation, explanation? }]
            8) parents[].id MUST be an existing node id from existing_node_ids.
            - Never omit parents[].id.
            - Never use placeholder ids (e.g., T000) unless they already exist.
            9) parents[].relation MUST be a short lowercase semantic label
            (e.g., produced, used, necessitated_by, ablation_of, variant_of).
            10) NEVER point to unknown or fabricated parent ids.
            11) Prefer linking to the most recent and most specific relevant node
                (use current_focus if it exists and is relevant).
           

            ────────────────────────────────
            ACTION-STEP NODE DEFINITION (MANDATORY):
            Each node MUST represent exactly ONE executable operation step.

            An operation step is defined as:
            - A single, concrete action that could be independently executed or rerun.
            - Produces a user-verifiable outcome (code, file, figure, metric, or written conclusion).
            - Cannot be meaningfully split without losing executability.

            INVALID nodes include:
            - High-level phases (e.g., "run experiments", "analyze results").
            - Multi-action bundles (e.g., implement + train + evaluate).
            - Plans, intentions, or exploratory reasoning without stable artifacts.

            ────────────────────────────────
            PARENT SELECTION & STRUCTURAL RULES (MUST COMPLY):

            A) Minimal Dependency Rule
            - Select the closest logical parent only.
            - DO NOT link to a parent if that parent is already an ancestor of another selected parent.
            - Grandparents and higher-order ancestors are implicitly inherited and MUST NOT be explicitly linked.

            B) Redundancy Prevention Rule
            - If two candidate parents share the same parent, prefer the more specific one.
            - NEVER connect a node to both a step and that step’s parent.

            ────────────────────────────────
            LOGICAL RELATIONSHIP RULES (MUST COMPLY):

            - Parent links MUST reflect logical dependency, NOT execution order.
            - Use relations consistently:
            - produced: parent directly produced this output
            - used: this step consumed an artifact or result from parent
            - ablation_of: this step removes or disables a component of the parent step
            - variant_of: this step is an alternative configuration or method based on the same parent

            - For ablations, baselines, or variants:
            - Nodes SHOULD share the same parent when they modify the same base step.
            - DO NOT chain ablations or variants linearly unless one explicitly depends on another.
            - Default assumption: ablations and variants are siblings, not ancestors.

            ────────────────────────────────
            FAILSAFE RULE:
            If no perfect parent exists, select the most relevant existing node
            , and keep the dependency minimal.
            '''
    )

    user_prompt = {
        "session_id": session_id,
        "project_name": project_name,
        "current_focus": current_focus,
        "existing_node_ids": existing_node_ids,
        "events": compact_events,
        "output_schema": {
            "node": {
                "id": "string (required)",
                "title": "string (required)",
                "description": "string (required)",
                "status": "string (optional)",
                "parents": [
                    {
                        "id": "string (required, must be in existing_node_ids)",
                        "relation": "string (required)",
                        "explanation": "string (optional)",
                    }
                ],
            }
        },
    }

    prompt = (
        system_hint
        + "\nHere is the JSON input:\n"
        + json.dumps(user_prompt, ensure_ascii=False, indent=2)
    )

    log.info(
        "got_llm events->nodes calling model session=%s events=%d existing_nodes=%d",
        session_id,
        len(compact_events),
        len(existing_node_ids),
    )
    log.debug("got_llm events->nodes prompt=%s", prompt)

    raw = await adapter.chat(prompt)
    if raw is None:
        raw = ""
    raw = (raw or "").strip()
    log.info("got_llm events->nodes model returned chars=%d", len(raw))
    log.debug("got_llm events->nodes raw=%s", raw)
    if not raw:
        log.warning("got_llm events->nodes empty model response")
        return []

    try:
        data = _extract_json(raw)
    except Exception:
        log.exception("got_llm events->nodes failed to parse JSON")
        return []

    if not isinstance(data, list):
        return []

    nodes: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        node_id = item.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        node_id = node_id.strip()

        # Enforce uniqueness across existing and newly created ids.
        if node_id in allowed_parent_ids:
            log.warning("got_llm events->nodes skipping duplicate id=%s", node_id)
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
                if isinstance(pid, str) and pid.strip() and isinstance(rel, str) and rel.strip():
                    pid = pid.strip()
                    if pid in allowed_parent_ids:
                        cp: Dict[str, Any] = {"id": pid, "relation": rel.strip()}
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
        if not item.get("parents"):
            log.warning("got_llm events->nodes skipping node id=%s (no valid parents)", node_id)
            continue
        # Optional: require non-empty title/description
        if not (item.get("title") and str(item.get("title", "")).strip()):
            item["title"] = item.get("title") or node_id
        if not (item.get("description") and str(item.get("description", "")).strip()):
            item["description"] = item.get("description") or ""

        nodes.append(item)
        allowed_parent_ids.add(node_id)

    return nodes


async def build_llm_nodes_from_subtask_summary(
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
    current_focus = steps.get("meta", {}).get("current_focus")
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
        "You are a strict information extractor. Convert the provided subtask summary into a Graph-of-Thought (GoT) DAG node or nodes.\n"
        "\n"
        "CONTEXT (align with MCP build_trace): Each call provides exactly ONE subtask = ONE executable action step (one verifiable deliverable). "
        "Output a JSON array of nodes: usually 1 node for this step; output 2-5 nodes ONLY when this single step has multiple distinct deliverables "
        "(e.g. one step that produces multiple figures, or multiple metric files). Do NOT bundle multiple unrelated steps into one response.\n"
        "\n"
        "Hard constraints (do not violate):\n"
        "1) Use ONLY facts present in the subtask JSON (title/description/depends_on) and artifacts. Do NOT invent commands, paths, files, or metrics.\n"
        "2) Output MUST be a pure JSON array (no markdown, no prose).\n"
        "3) Each node MUST have: id (string) that MUST NOT collide with any id in existing_node_ids OR other newly-created ids in this output.\n"
        "4) Each node MUST have: title (string, non-empty).\n"
        "5) Each node MUST have: description (string, non-empty, factual).\n"
        "6) Each node MUST have: parents (array) with at least 1 parent.\n"
        "7) Parent links MUST use the field parents: [{id, relation, explanation?}].\n"
        "8) parents[].id MUST reference an id in existing_node_ids OR an id created earlier within this same output array.\n"
        "9) parents[].relation MUST be a short lowercase string like: produced, used, necessitated_by, ablation_of.\n"
        "10) Use dependency hints: if subtask.depends_on mentions multiple distinct prior results/steps, use multiple parents (multi-parent).\n"
        "11) Prefer sibling grouping: if this is another output of the same experiment/run, choose the experiment/run parent rather than chaining linearly.\n"
        "12) If unsure, use current_focus as a parent IF it exists in existing_node_ids; otherwise choose the most relevant existing node.\n"
        "13) Titles must be concise and human-readable.\n"
        "\nRelationship examples (follow when applicable):\n"
        "- Ablation experiments: variants are siblings under the same experiment/run parent; relation=ablation_of to the baseline.\n"
        "- Multiple analysis angles: siblings under the same experiment/data parent (avoid chaining analyses linearly).\n"
        "- Multiple model implementations: siblings under the same task/experiment parent.\n"
        "- Alternative preprocessing: variants under the same experiment parent; relation=ablation_of or use to connect to baseline.\n"
    )

    user_prompt = {
        "session_id": session_id,
        "project_name": project_name,
        "current_focus": current_focus,
        "existing_node_ids": existing_node_ids,
        "existing_nodes": current_nodes,
        "subtask": compact_subtask,
        "artifacts": compact_artifacts,
        "output_schema": {
            "node": {
                "id": "string (required, unique in existing_node_ids and in this output array)",
                "title": "string (required)",
                "description": "string (required)",
                "status": "string (optional)",
                "parents": [
                    {
                        "id": "string (required: existing_node_ids OR id of a node earlier in this output array)",
                        "relation": "string (required)",
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

    # allowed_parent_ids is already set above; we mutate it in the loop so same-batch parent refs work.
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
        if node_id in allowed_parent_ids:
            log.warning("got_llm subtask->nodes skipping duplicate id=%s", node_id)
            continue
        item["id"] = node_id

        # Attach artifacts (if any) so output aligns with caller-provided file list.
        if compact_artifacts:
            item["artifacts"] = compact_artifacts

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
                    if pid in allowed_parent_ids:
                        cp: Dict[str, Any] = {"id": pid, "relation": rel.strip()}
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
        if not item.get("parents"):
            log.warning("got_llm subtask->nodes skipping node id=%s (no valid parents)", node_id)
            continue
        # Ensure title/description present and non-empty
        if not (item.get("title") and str(item.get("title", "")).strip()):
            item["title"] = item.get("title") or node_id
        if not (item.get("description") and str(item.get("description", "")).strip()):
            item["description"] = item.get("description") or ""

        nodes.append(item)
        allowed_parent_ids.add(node_id)

    log.info("got_llm subtask->nodes accepted nodes=%d from model array len=%d", len(nodes), len(data))
    return nodes



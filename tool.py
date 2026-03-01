from __future__ import annotations

from typing import Annotated, List, Dict, Optional, Any
from openai import OpenAI
import os
import json
import logging
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding
from pydantic import BaseModel


# -------- Monitor (build_trace / GoT) --------
# Monitor tool suite: records completed, user-verifiable subtasks into the task GoT.
# Registered by the combined MCP server entrypoint (server.py).
# GoT output: /***/openhands/<project_name>/.openhands/got/<session_id>/got.json

_log_monitor = logging.getLogger("mcp_tools.monitor")


class BuildTraceSession(BaseModel):
    id: str


class BuildTraceSubtask(BaseModel):
    title: str
    description: str
    # Optional dependency hints provided by the agent.
    # Monitor may use this (plus artifacts and existing nodes) to infer parents/relations.
    depends_on: Optional[List[str]] = None
    status: Optional[str] = None


class BuildTraceArtifact(BaseModel):
    path: str
    type: str


class BuildTraceProject(BaseModel):
    name: str


class BuildTracePayload(BaseModel):
    """Payload for Monitor.build_trace.

    Semantics:
    - In a single MCP session, `session.id` and `project.name` are **session-wide common identifiers**
      and should stay consistent across calls.
    - Each call records exactly **one** `subtask` and its corresponding `artifacts` (files involved in
      this step, such as code, images, data, etc.).
    - `subtask.depends_on` (optional) may provide dependency hints (natural language) about what this
      subtask depends on (prior results, artifacts, earlier steps).

    Required:
    - payload.session.id
    - payload.project.name
    - payload.subtask (one)
    - payload.artifacts (required but may be empty [])

    Output:
    - Writes GoT JSON to `/srv/openhands/<project_name>/.openhands/got/<session_id>/got.json`.
    """

    project: BuildTraceProject
    session: BuildTraceSession
    subtask: BuildTraceSubtask
    artifacts: List[BuildTraceArtifact]

    def model_post_init(self, __context: Any) -> None:
        name = self.project.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("project.name is required")


async def build_trace(
    payload: Annotated[
        BuildTracePayload,
        "Payload: project.name, session.id, one subtask (title, description, optional depends_on), artifacts list (required, may be empty).",
    ]
) -> Dict[str, str]:
    """Record a single **completed, user-verifiable subtask** into the task GoT.

    <MCP_TOOL_PURPOSE>
    * This tool records **what was done** (an executable, verifiable operation) and its artifact
      dependencies.
    * It is NOT for chain-of-thought, progress logs, explanations, or exploratory work.
    </MCP_TOOL_PURPOSE>

    <MCP_CALL_GATE>
    * Each MCP record MUST represent exactly ONE executable action step.
    * Do NOT call for planning, debugging, or rephrasing.
    * After completing any subtask, you must immediately record it via the MCP tool.
    * When in doubt about whether the step qualifies or not, you must still record it.
    * Each deliverable should be recorded **once only**.
    * Prefer multiple smaller calls over one big call when you have multiple distinct deliverables.
    </MCP_CALL_GATE>

    <MCP_INPUT_REQUIREMENTS>
    * Provide exactly ONE `subtask` describing a minimal executable unit (install dependencies/implement/run/compute/plot/
      evaluate/conclude) with a factual `title` and `description`.
    * If this subtask depends on prior results/steps, provide `subtask.depends_on` (string list) to
      describe those dependencies (e.g., which earlier experiment/result/plot is used).
    * `subtask.depends_on` MAY include multiple items.
      - If you list multiple dependencies, you are indicating the current deliverable is jointly
        supported/derived from multiple prior steps (multi-parent relationship).
      - For ablations / alternative implementations / alternative preprocessing / alternative
        visualizations: prefer creating sibling nodes under the same experiment/task parent, rather
        than chaining them linearly.
    * Provide `artifacts` (required, may be empty []) listing the files involved in this subtask. If no clear file artifacts exist (e.g., environment setup, literature reading, conceptual analysis), it is still allowed to record the subtask as long as the action is concrete, completed, and described factually.
    </MCP_INPUT_REQUIREMENTS>

    <MCP_EXAMPLES_OF_VALID_SUBTASKS>
    Valid subtasks MUST correspond to exactly ONE executable action step.
    Examples include, but are NOT limited to:
    * Literature understanding
    - Example: "Understand the literature on the effects of caffeine on neural activity"

    * Environment setup
    Installing required packages, setting up dependencies, or preparing execution conditions. 
    Each major installation step should be recorded separately when distinct and verifiable (e.g., different package groups, CUDA-specific installs).  
    Include specific package names, versions, or installation commands in the description when applicable. The artifact can be a requirements.txt file, installation log snippet, or confirmation of successful setup.
    - Example: "Installing the dependencies for the experiment, such as scipy, numpy, matplotlib, pandas, etc."
    - Example: "Installing the pytorch package which supports cuda, the version is 2.5.1+cu121"

    * Tool usage
    - Example: "Use the net_neuro_tools_instruct tool to get the information about the netneurotools package"

    * Data acquisition
    - Example: "Acquire the data for the experiment at /data directory"

    * Data preprocessing
    - Example: "Handle missing values in the dataset using median imputation for numerical features and mode for categorical features"
    - Example: "Remove duplicate rows and outliers using IQR method (threshold=1.5) on the housing price dataset"
    - Example: "Normalize numerical features to [0,1] range using Min-Max scaling"
    - Example: "Standardize features using z-score normalization (mean=0, std=1)"
    - Example: "One-hot encode categorical variables (e.g., 'region', 'product_type')"
    - Example: "Apply log transformation to skewed target variable (price) to reduce skewness"
    - Example: "Split dataset into train/val/test sets (80/10/10) with stratification on target class"

    * Code Editing
    Creating new code files or making significant modifications, fixes, or additions to existing code (excluding execution/training)
    - Example: "Implement SVM classifier with RBF kernel and C=1.0 using scikit-learn"
    - Example: "Implement a 3-layer MLP with ReLU activations and dropout=0.3 in PyTorch"
    - Example: "Define ResNet-18 architecture with custom first conv layer for 1-channel input (e.g., grayscale images)"
    - Example: "Implement a GCN layer with 2-hop aggregation and hidden dim=128 using PyTorch Geometric"
    - Example: "Create a custom Transformer encoder block with 4 heads and feed-forward dim=1024" 

    * Experiment execution
    Actually running model training, inference, or evaluation. Must include a complete reproducible configuration (seed, key hyperparameters, dataset subset, etc.) to produce concrete, independently verifiable artifacts (metrics files, logs, checkpoints, prediction outputs, etc.).  
    **Core Principle**: Each distinct configuration variant (ablation, hyperparameter change, component removal/addition, data subset variation, etc.) **must** be recorded as a separate subtask, forming sibling nodes at the same level. **It is strictly prohibited** to bundle multiple variants into a single subtask.  
    Recommended practice:  
        - Record the baseline experiment once  
        - Record each ablation/variant as a separate subtask (reuse the same seed for fair comparison when appropriate)  
        - Clearly label the variant name or “Ablation: XXX” in the title/description  
        - Subsequent analysis, visualization, and conclusion subtasks **must** reference these independent experiment nodes via `depends_on` 
    - Example: "Train and evaluate ResNet-18 baseline on CIFAR-10 (seed=42, batch_size=128, SGD lr=0.1, 200 epochs)"  
    - Example: "Ablation: Train ResNet-18 without any data augmentation (seed=42, batch_size=128, SGD lr=0.1, 200 epochs)"  
    - Example: "Variant: Train ResNet-18 with learning rate 0.05 (seed=42, batch_size=128, SGD lr=0.05, 200 epochs)"  
    - Example: "Ablation: Remove last residual block from ResNet-18 (seed=42, batch_size=128, SGD lr=0.1, 200 epochs)"  
    - Example: "Fine-tune BERT-base-uncased on SST-2 (seed=123, lr=2e-5, max_seq_len=128, weight_decay=0.01)"  
    - Example: "Ablation: Fine-tune BERT with first 8 layers frozen (seed=123, lr=3e-5, max_seq_len=128)"  
    - Example: "Run XGBoost baseline on Boston housing (seed=7, n_estimators=100, max_depth=5)"  
    - Example: "Ablation: XGBoost with max_depth=3 only (seed=7, n_estimators=100, max_depth=3)"

    * Visualization  
    Generating specific, named data visualization figures. Each distinct plot or figure must be recorded as a separate subtask — do not combine multiple unrelated plots into one record.  
    The artifact is usually the saved image file (e.g., .png, .pdf, .jpg) and optionally the code/script that produced it.  
    - Example: "Plot ROC curve for model A on test set"
    - Example: "Plot training and validation loss curves over epochs for experiment run seed=42"

    * Result analysis  
    Interpreting, comparing, or summarizing concrete results or figures that have already been recorded as artifacts.  
    Each analysis subtask should focus on one clear angle or comparison, and must explicitly depend on previously recorded artifacts (via depends_on). Do not record vague or general impressions — the output should be a verifiable summary file, table, or text segment.  
    - Example: "Compare ROC curves of model A and model B and compute the difference in AUC"
    - Example: "Analyze the training vs validation loss curves (from loss_plot_seed42.png) and identify signs of overfitting"

    * Conclusion Making  
    Drawing specific, verifiable conclusions or answering the original research question, based on multiple prior analysis steps or key results.  
    This is usually the final or near-final step. Each conclusion subtask must clearly depend on one or more previously recorded analysis/visualization subtasks (via depends_on). Do not make broad or unsupported claims — the artifact should be a written paragraph, bullet points, or short report section that can be independently verified against prior artifacts.  
    - Example: "Conclude whether model A outperforms model B based on AUC comparison and statistical significance test"
    - Example: "Summarize that the addition of data augmentation reduces overfitting, as evidenced by the loss curve analysis and final test performance"
    </MCP_EXAMPLES_OF_VALID_SUBTASKS>
    """
    if isinstance(payload, dict):
        payload = BuildTracePayload(**payload)

    project_name = payload.project.name.strip()
    session_id = payload.session.id

    payload_dict = payload.model_dump(exclude_none=True)
    artifacts_count = len(payload_dict.get("artifacts") or [])

    _log_monitor.info(
        "build_trace input project=%s session=%s artifacts=%d",
        project_name,
        session_id,
        artifacts_count,
    )
    _log_monitor.debug(
        "build_trace payload keys=%s",
        sorted(list(payload_dict.keys())),
    )

    from Monitor.got_writer import write_got_from_build_trace

    try:
        res = await write_got_from_build_trace(
            project_name=project_name,
            session_id=session_id,
            payload=payload_dict,
        )
        _log_monitor.info(
            "build_trace output status=%s primary_node_id=%s",
            res.get("status", "ok"),
            res.get("primary_node_id", ""),
        )
        _log_monitor.debug("build_trace raw result=%s", json.dumps(res, ensure_ascii=False))
        return {"status": res.get("status", "ok")}
    except Exception:
        _log_monitor.exception(
            "build_trace failed project=%s session=%s",
            project_name,
            session_id,
        )
        return {"status": "error", "message": "build_trace failed (see server logs)"}







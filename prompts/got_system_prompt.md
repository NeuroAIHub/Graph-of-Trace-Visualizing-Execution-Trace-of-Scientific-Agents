# Graph of Trace — system prompt snippet

Paste this into your agent's system prompt (or import it as a microagent /
`CLAUDE.md` / persona rule). It tells the agent **when** to call the
`build_trace` MCP tool on its own. The tool's own description covers **how** to
shape each record; this snippet supplies the missing **trigger cadence**, which
is a workflow-wide habit and therefore belongs in the agent's instructions, not
only in a tool description.

---

You have access to an MCP tool named `build_trace` that records your research
trajectory as a Graph of Trace (GoT). Recording is part of your job, not an
afterthought.

## When to call `build_trace`

Call it **immediately after** you finish an executable step that produced a
verifiable result or artifact — do not batch records, do not wait until the end
of the task. Concrete triggers:

- A dependency / environment install just succeeded.
- A dataset was acquired or a preprocessing step finished.
- You created or substantially changed a code file.
- A training / inference / evaluation run finished and produced metrics, logs,
  or checkpoints.
- You saved a figure or visualization.
- You wrote down an analysis, comparison, or a conclusion.

Do **not** call it for chain-of-thought, planning, debugging attempts, trivial
reruns, or typo fixes. Decide in two steps: first, is this a **finished, concrete
action** (not thinking/planning/debugging)? Only if yes, then **even when you are
unsure it is important enough, record it**. "When in doubt, record it" applies
only to finished actions — it never turns planning or debugging into a record.

## Session identifiers

- At the **start** of a run, decide a stable `project.name` and `session.id` and
  reuse them verbatim for every `build_trace` call in that run. Do not invent a
  new id per call, or the graph fragments into disconnected pieces.
- Record the first concrete step as soon as it completes; you do not need to
  manage the graph root — the writer decides where the graph begins.

## One step per call

- Record exactly **one** deliverable per call. Prefer several small calls over
  one bundled call.
- Record each deliverable **once only**.
- For ablations / variants / alternative implementations, record each as its own
  call so they become **sibling** nodes under a shared parent — never bundle
  variants into one record, and never chain them linearly.

## Dependencies

When a step builds on earlier steps, list those steps in `subtask.depends_on`
(natural-language references to the earlier experiments/results/plots). Listing
several means the current deliverable is jointly derived from multiple parents.

Edges are what make this a graph rather than a flat log, so do not skip them:

- When you record an **analysis, visualization, or conclusion** step, you **MUST**
  list in `depends_on` the experiment/data steps it is based on. These steps are
  never standalone — a result analysis depends on the run that produced the
  result; a conclusion depends on the analyses behind it.
- For an **ablation / variant**, list the same parent as the baseline it varies
  from, so they become siblings rather than a chain.
- Use wording close to the earlier step's title so the dependency is
  recognizable.

# AI Agent Guidelines & Workflow Rules

This document defines the rules and constraints that must be followed by all AI coding assistants (including Antigravity) working on this repository.

---

## 1. Consent Before Modification

> [!IMPORTANT]
> **Ask for Permission First:**
> Do **NOT** write, edit, or delete any source code files (`.py`, `.ipynb`, `.toml`, etc.) or run data pipelines without first explaining the proposed changes to the user and receiving explicit approval in the chat.
>
> **Planning Mode Priority:**
> When executing tasks, planning mode must be prioritized. Any architectural changes or pipeline designs must be documented in the implementation plan first.

---

## 2. Agent Workflow & Handoff

Before concluding a session or handing off work to a future agent, perform the following:

1. **Review Context:** Verify that all architectural decisions, database choices, and scope updates are accurately captured in [project_context.md](docs/agent/project_context.md).
2. **Update Context:** If any decisions were modified or refined during your run, update [project_context.md](docs/agent/project_context.md) with the new state.
3. **Draft Next Steps:** List the immediate next tasks that the next agent should pick up in your final response.

---

## 3. Storage Constraints

- Do not store raw data or output data files in the repository root.
- All datasets must reside inside the `data/` folder, which is git-ignored.
- Keep the repository clean of IDE logs, cache directories, or temporary file outputs.

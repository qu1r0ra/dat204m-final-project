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
>
> **Automatic Readiness Reviews:**
> Prior to executing any major next steps (such as deploying AWS infrastructure, establishing spoke connectivity, or running model scale-up pipelines), the agent must automatically perform a comprehensive codebase readiness and architecture review, presenting findings in a structured artifact before requesting permission to execute.

---

## 2. Agent Workflow & Handoff

Before concluding a session or handing off work to a future agent, perform the following:

1. **Review Context:** Verify that all architectural decisions, database choices, and scope updates are accurately captured in [project_context.md](docs/agent/project_context.md).
2. **Update Context:** If any decisions were modified or refined during your run, update [project_context.md](docs/agent/project_context.md) with the new state.
3. **Handoff Message**: When the user requests a "handoff message", generate a concise, copy-pasteable block of text summarizing current progress and exact next steps that the user can use to initiate the next agent session. Do not repeat information already present in context documents, but ensure it provides sufficient guiding context for the next agent to resume work immediately. Avoid filler words.
4. **Draft Next Steps:** List the immediate next tasks that the next agent should pick up in your final response.

---

## 3. Storage Constraints

- Do not store raw data or output data files in the repository root.
- All datasets must reside inside the `data/` folder, which is git-ignored.
- Keep the repository clean of IDE logs, cache directories, or temporary file outputs.

---

## 4. Writing & Formatting Preferences

- **Relative Links Only**: Do NOT use absolute links or `file:///` paths in markdown files (e.g., in documentation or plans). Always use relative markdown links so they remain functional for all team members.
- **No Emojis**: Do NOT use emojis in any markdown documents (including internal project files like plans, tasks, roles, and guides).
- **Professional Tone**: Keep the tone professional and objective in final/formal reports (avoiding flowery language or excessive adjectives). This strict constraint is relaxed for internal project markdown files (e.g., plans, roles, tasks) where formatting and structure are preferred to maximize readability, but flowery language must still be avoided.

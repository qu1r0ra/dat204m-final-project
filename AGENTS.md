# Binance K-Lines Analytics — Agent Index

**Start here for a new session:** [HANDOFF.md](HANDOFF.md) (workspace-wide living handoff).

Canonical domain rules are defined in Cursor Rules format under [.cursor/rules/](.cursor/rules/).

| Resource                   | Path                                                                       | Description                                                 |
| -------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Workspace handoff**      | [HANDOFF.md](HANDOFF.md)                                                   | Living document for task states, blockers, and timelines    |
| **Project Overview Rules** | [.cursor/rules/project-overview.mdc](.cursor/rules/project-overview.mdc)   | Background on Binance Spot 1m k-lines prediction            |
| **Tech Stack Rules**       | [.cursor/rules/tech-stack.mdc](.cursor/rules/tech-stack.mdc)               | Guidelines for Python, uv, DuckDB, Polars, and dependencies |
| **Data Directory Rules**   | [.cursor/rules/data-organization.mdc](.cursor/rules/data-organization.mdc) | Location and git-ignored configurations for datasets        |
| **Workflow Rules**         | [.cursor/rules/agent-workflows.mdc](.cursor/rules/agent-workflows.mdc)     | Consent rules, planning requirements, and reviews           |
| **Data Registry**          | [.cursor/project/data_registry.md](.cursor/project/data_registry.md)       | Registry tracking schema, datasets, and files               |
| **Model Registry**         | [.cursor/project/model_registry.md](.cursor/project/model_registry.md)     | Registry tracking ML experiments and classifiers            |

---

## Workspace Rules (Summary)

### 1. Consent Before Modification

Do **NOT** write, edit, or delete any source code files (`.py`, `.ipynb`, `.toml`, etc.) or run data pipelines without first explaining the proposed changes to the user and receiving explicit approval in the chat. Document all architectural changes in the implementation plan first.

### 2. Storage Constraints

- Do not store raw data or output data files in the repository root.
- All datasets must reside inside the `data/` folder, which is git-ignored.

### 3. Writing & Formatting Preferences

- **Relative Links Only**: Do NOT use absolute links or `file:///` paths in markdown files. Always use relative markdown links.
- **No Emojis**: Do NOT use emojis in any markdown documents.
- **Professional Tone**: Keep the tone professional, objective, and clear.

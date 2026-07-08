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

## Workspace Rules

Canonical domain and collaboration rules are managed in [.cursor/rules/](.cursor/rules/). Please refer to the specific rule files for detailed guidelines:

- **Collaboration & Consent**: Refer to [agent-workflows.mdc](.cursor/rules/agent-workflows.mdc) for consent, reviews, and formatting guidelines.
- **Data Policies**: Refer to [data-organization.mdc](.cursor/rules/data-organization.mdc) for local data constraints and AWS security settings.
- **Tech Stack & Configurations**: Refer to [tech-stack.mdc](.cursor/rules/tech-stack.mdc) for guidelines on approved technologies and configurations.

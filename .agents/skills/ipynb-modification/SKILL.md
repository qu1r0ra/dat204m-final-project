---
name: ipynb-modification
description: Modifying Jupyter Notebook (.ipynb) files programmatically using Python scripts since standard file editing tools restrict editing them.
---

# Jupyter Notebook Programmatic Modification

## Context

Standard file editing tools (such as `replace_file_content` and `multi_replace_file_content`) restrict modifications to files with `.ipynb` extension.

## Strategy

To modify a `.ipynb` notebook file:

1. Write a scratch Python script (typically saved in the session's scratch directory: `<appDataDir>/brain/<conversation-id>/scratch/modify_notebook.py`).
2. The Python script should load the notebook as JSON using the built-in `json` library, inspect and modify the notebook cells (under the `"cells"` key) programmatically, and then write the updated JSON structure back.
3. Run the script using Python through a terminal command (e.g., `uv run python path/to/script.py`).
4. Run any validation commands (like `ruff check` or `pytest`) to verify the modified notebook structure is correct.

## Example

Below is an example Python script pattern to modify notebook cells:

```python
import json
from pathlib import Path

notebook_path = Path("notebooks/my_notebook.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

modified = False
for cell in data["cells"]:
    if cell["cell_type"] == "code":
        # Search and replace in cell source strings
        source = cell["source"]
        for i, line in enumerate(source):
            if "old_pattern" in line:
                source[i] = line.replace("old_pattern", "new_pattern")
                modified = True

if modified:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
```

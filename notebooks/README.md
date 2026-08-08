# Notebooks

Notebooks are thin: they clone the repo, load a config, and call the `csne`
CLI. Logic lives in `src/csne/` so it is testable and diffable — a notebook
containing real logic cannot be reviewed or tested.

They are stored as jupytext-style **percent scripts** (`.py` with `# %%` cell
markers), not `.ipynb`. Notebooks diff badly and carry execution state; the
Colab MCP server builds the actual notebook by executing these cells.

| Notebook | Purpose | Runtime | Status |
|---|---|---|---|
| `02_train_retrieval.py` | Fine-tune MiniLM on 50K `(query, code)` pairs | Colab T4 | ready |
| `01_data_prep.py` | Standalone data preparation | CPU | not needed — `02` does it |
| `03_evaluate.py` | Broader evaluation sweep | Colab T4 | planned |
| `04_demo.py` | Interactive find-and-explain | CPU | planned (needs summarization) |

## Before running on Colab

`02_train_retrieval.py` **clones the pushed state of `main`**. Anything
uncommitted locally will not be there. Push first, or the run trains against
stale code.

Select a T4 runtime (Runtime → Change runtime type → T4 GPU). The notebook
asserts on `torch.cuda.is_available()` in its third cell, because Colab
silently hands out CPU runtimes and the run would take hours rather than
minutes.

## Driving Colab from Claude Code

The `colab-mcp` server is registered in `.mcp.json`:

```json
{"mcpServers": {"colab-mcp": {"command": "uvx",
  "args": ["git+https://github.com/googlecolab/colab-mcp"]}}}
```

It requires approval on first use — restart Claude Code and approve it, then
the Colab tools become available. It bridges to a browser Colab session, so
be signed in to Google in the same browser.

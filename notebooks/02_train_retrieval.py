# %% [markdown]
# # Fine-tune the retrieval encoder (Colab T4)
#
# Paired with `configs/experiments/finetune_minilm_50k.yaml`: 50K sampled
# `(query, code)` pairs, one epoch, MultipleNegativesRankingLoss.
#
# This file is a jupytext-style percent script rather than a checked-in
# `.ipynb`: notebooks diff badly, and the Colab MCP server creates the actual
# notebook by executing these cells. Run top to bottom.
#
# Expected wall time on a free T4: ~10-15 minutes for 50K pairs at batch 64.

# %% [markdown]
# ## 1. Environment
#
# Check the GPU first — Colab silently hands out CPU runtimes when no
# accelerator is selected, and the run would take hours instead of minutes.

# %%
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "NO GPU")

# %%
# Clone rather than `pip install git+...`: the CLI is driven entirely by the
# YAML files under `configs/`, and a wheel install would not bring them along.
#
# This clones the pushed state of `main` — anything still uncommitted locally
# will NOT be here. Push before running.
#
# ruff: noqa: E402
get_ipython().system(  # noqa: F821
    "git clone -q https://github.com/opnsrcntrbtr/codesearchnet-find-and-explain.git repo"
)
get_ipython().run_line_magic("cd", "repo")  # noqa: F821

# The `hf` extra is not needed: this trains the retrieval encoder only.
get_ipython().system('pip install -q -e ".[dev]"')  # noqa: F821

# %%
import torch

assert torch.cuda.is_available(), "No CUDA device. Runtime > Change runtime type > T4 GPU."
print(torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2. Data
#
# `prepare-data` streams both datasets and joins them on the GitHub permalink.
# The summary map alone is ~455K rows, so this is the slow step (~5-8 min);
# it caches to `data/cache/summary_map.json` and later runs skip it.

# %%
get_ipython().system(  # noqa: F821
    "csne prepare-data --config configs/experiments/finetune_minilm_50k.yaml"
)

# %% [markdown]
# ## 3. Baseline
#
# Score BM25 and the untouched encoder *before* training, on this machine.
# Comparing against numbers produced elsewhere would confound the fine-tuning
# effect with environment differences.

# %%
get_ipython().system(  # noqa: F821
    "csne evaluate --config configs/experiments/baseline.yaml --bm25"
)

# %% [markdown]
# ## 4. Fine-tune

# %%
get_ipython().system(  # noqa: F821
    "csne train --config configs/experiments/finetune_minilm_50k.yaml"
)

# %% [markdown]
# ## 5. Evaluate the fine-tuned encoder
#
# Same split, same distractor pool, same metrics — only the checkpoint differs.

# %%
get_ipython().system(  # noqa: F821
    "csne evaluate --config configs/experiments/finetune_minilm_50k.yaml "
    "--checkpoint checkpoints/retrieval-50k"
)

# %% [markdown]
# ## 6. Results
#
# Every run appended a row tagged with its config hash and git commit.

# %%
import pandas as pd

results = pd.read_csv("results/retrieval_results.csv")
print(results[["run_name", "mrr", "recall@1", "recall@5", "recall@10"]].to_string(index=False))

# %% [markdown]
# ## 7. Retrieve the artifacts
#
# The checkpoint and the results CSV are the only outputs worth keeping; the
# Colab filesystem is discarded when the runtime recycles.

# %%
get_ipython().system("tar czf retrieval-50k.tar.gz checkpoints/retrieval-50k")  # noqa: F821

# %%
# Copy `results/retrieval_results.csv` back into the repo and commit it, so the
# numbers in REPORT.md stay traceable to the run that produced them.
print(open("results/retrieval_results.csv").read())

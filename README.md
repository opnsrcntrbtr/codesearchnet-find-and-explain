# codesearchnet-find-and-explain

## Overview

`codesearchnet-find-and-explain` is a Python-focused semantic code search and
summarization tool. It is built on publicly available CodeSearchNet-style
datasets sourced from Hugging Face and GitHub, and is designed for
reproducible experiments run on Google Colab GPUs, orchestrated end-to-end
via Claude Code.

## Goals

- Fine-tune a retrieval model on `(query, code)` pairs drawn from
  CodeSearchNet Python data.
- Integrate a summarization model to explain retrieved functions in plain
  language ("find" the code, then "explain" it).
- Provide a Colab + Hugging Face demo, along with experiment logs, suitable
  for academic discussion and write-up.

## High-level architecture

- **Data layer** — Two Hugging Face datasets, joined on the GitHub permalink
  (verified 100% match on a 20K sample):
  `code-search-net/code_search_net` (config `python`) supplies the corpus and
  its official, repo-disjoint train/valid/test splits, so retrieval numbers
  stay comparable to published CodeSearchNet baselines;
  `Nan-Do/code-search-net-python` supplies a curated one-line `summary` per
  function, used as the summarization target. Docstrings are stripped of
  parameter blocks and doctests, code is stripped of its docstring, and
  near-duplicate functions are removed.
- **Retrieval layer** — A sentence-transformer or code-specific encoder is
  fine-tuned on Colab using information-retrieval losses to embed queries
  and code into a shared space.
- **Summarization layer** — A code LLM or Hugging Face summarization model
  generates concise, function-level summaries for retrieved code.
- **Evaluation layer** — Search quality is measured with MRR, Recall@k, and
  NDCG; summarization quality is measured with BLEU/BERTScore and,
  optionally, LLM-based scoring.
- **Orchestration** — Claude Code drives the project locally (files,
  configs, experiment scripts) while delegating GPU-heavy training and
  notebook execution to Colab via Colab MCP / `claude-colab`.

## Roadmap / plan (docs-first)

We are currently in **Phase 3 (implementation)**.

1. ~~**Phase 1 – Documentation**: extend `README.md`, write the `REPORT.md`
   outline, and add `CLAUDE.md`.~~ ✅
2. ~~**Phase 2 – Scaffold repo structure**: `src/`, `notebooks/`, `configs/`,
   `tests/`, `results/`.~~ ✅
3. **Phase 3 – Implement data loaders** and minimal retrieval +
   summarization models. ← current (data, retrieval, evaluation done;
   summarization backends pending)
4. **Phase 4 – Fine-tune the retrieval model** on Colab and log experiments.
5. **Phase 5 – Build demo notebooks** and/or a Hugging Face Space.
6. **Phase 6 – Refine evaluation** and prepare the academic write-up.

## Repository layout

```
src/csne/
  config.py              # typed configs loaded from configs/*.yaml
  data/                  # loader, preprocess, splits (grouped by repo)
  retrieval/             # encoder, exact-search index, fine-tuning
  summarization/         # Summarizer interface + anthropic / hf backends
  evaluation/            # retrieval + summarization metrics, results logging
  pipeline.py            # FindAndExplain: retrieve, then explain
  cli.py                 # csne prepare-data | train | evaluate | search
configs/                 # data, retrieval, summarization, eval + experiments/
prompts/                 # summarization prompt templates (versioned as files)
notebooks/               # thin Colab notebooks that call into csne
tests/                   # pytest suite
results/                 # appended experiment CSVs, committed
```

The data, retrieval, and evaluation layers are implemented and tested. The
summarization backends and `pipeline.py` are still stubs.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
uv pip install -e ".[dev,anthropic,hf,eval]"   # + summarization backends
```

## Running

```bash
csne prepare-data --config configs/experiments/baseline.yaml
csne evaluate     --config configs/experiments/baseline.yaml --bm25
csne train        --config configs/experiments/finetune_minilm.yaml
csne search       --config configs/experiments/baseline.yaml --query "parse a json file"
```

`evaluate` appends a row to `results/retrieval_results.csv` tagged with the
config hash and git commit, so every number traces back to the run that
produced it.

## Tests

```bash
pytest -m "not slow"   # fast, offline
pytest                 # includes tests that load the real MiniLM checkpoint
```

Choosing a summarization backend: set `backend: anthropic` (needs
`ANTHROPIC_API_KEY`) or `backend: hf` (local, no key) in
`configs/summarization.yaml`.

## Colab + Claude Code usage (high-level only)

- Colab MCP / `claude-colab` will be used to run training notebooks on
  GPUs, giving Claude Code direct control over Colab execution.
- Claude Code will manage files, configs, and experiment scripts locally,
  delegating heavy compute (fine-tuning, large-batch evaluation) to Colab
  rather than running it in-repo.

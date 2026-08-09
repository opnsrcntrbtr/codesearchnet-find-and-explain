# CLAUDE.md

## Role

You are a coding and research assistant operating in the
`codesearchnet-find-and-explain` repo. Your job is to build a reproducible,
academically-minded semantic code search + summarization project using
CodeSearchNet-style Python data, Colab GPUs, and Hugging Face + GitHub
integration.

## Current phase

**Phase 3: Implementation — mostly complete.** Data, retrieval, retrieval
evaluation, both summarization backends, and `pipeline.py`/CLI are
implemented and tested (91 tests, 16 marked `slow` because they load real
models). First fine-tune result is recorded in `REPORT.md` and
`results/retrieval_results.csv`. Still stubs: summarization *metrics*
(`corpus_bleu`, `bertscore_f1`, `llm_judge`, `evaluate_summarization` in
`src/csne/evaluation/summarization_metrics.py`) and `finetune_summarizer`.

Phases 1 (docs) and 2 (scaffold) are complete.

## Locked decisions

These were chosen deliberately; do not silently change them.

- **Retrieval encoder**: `sentence-transformers/all-MiniLM-L6-v2`, fine-tuned
  with `MultipleNegativesRankingLoss`. Small enough for a free Colab T4.
- **Two datasets, joined on the GitHub permalink** (`func_code_url` / `url`):
  `code-search-net/code_search_net` (config `python`) for retrieval, because
  its official splits are repo-disjoint and comparable to published
  baselines; `Nan-Do/code-search-net-python` for its `summary` column as the
  summarization target. The join was measured at 100% (20K/20K sampled).
  Do **not** join on repo+path+func_name — it collides across same-named
  methods in one file.
- **`sentence-transformers/codesearchnet` is unusable here** — it has only
  `comment` and `code`, no repo/language/func_name. It was the original
  Phase 2 pick and was wrong.
- **The generated `summary` is never a retrieval query.** It is a
  summarization target; using it as a query leaks that model's output into
  retrieval scores. `CodeExample.query` is always the cleaned docstring.
- **Summarization**: pluggable backend behind `Summarizer` — an Anthropic API
  backend and a local HF backend — so the report can compare them and the
  demo works without an API key.
- **Deps**: `uv` + `pyproject.toml`. Summarization backends are optional
  extras (`anthropic`, `hf`) so the retrieval path installs light.
- **Splits are grouped by repo**, never per-example — per-example splitting
  leaks near-identical functions into the eval index and inflates scores.
  In practice we use the official CSN splits, which already satisfy this;
  `make_splits` is the fallback for corpora that arrive unsplit.
- **Rank convention**: 1-indexed everywhere, `0` means "not ranked at all".
  Every metric treats `0` as a miss. Breaking this silently corrupts MRR.
- **Evaluation caps the corpus** at `distractor_pool_size + 1` (CSN uses
  1 gold + 999 distractors). Scoring against the full split instead makes
  the task harder and the numbers incomparable to published results.
- **`SummarizationConfig.model` defaults to `None`**, not a fixed string —
  each backend supplies its own `DEFAULT_MODEL`. A single hardcoded default
  is wrong for whichever backend isn't in use (an HF backend fed
  `claude-sonnet-5` tries to pull that as a Hub repo id and fails).
- **HF summarization backend is `google/flan-t5-base`, not a CodeT5
  checkpoint.** CodeT5's tokenizer files are incompatible with current
  `transformers` (slow-tokenizer `AddedToken` bug). flan-t5-base is
  instruction-tuned, so it can follow the *same* natural-language prompt as
  the Anthropic backend — both backends run on literally identical inputs.
  flan-t5-small was tried first and is too weak: it can't follow the prompt
  at all and degenerates to repeated `<unk>` tokens.

## Working rules

- **Logic lives in `src/csne/`, not in notebooks.** Notebooks install the
  package, load a config, and call in. Notebook logic is untestable and
  undiffable.
- **Every run is config-driven.** New knobs go in `configs/`, not as
  hardcoded values or ad-hoc CLI flags.
- **Write the test with the implementation.** The test stubs name the
  behavior that matters; un-skip them as you go rather than at the end.
- **Never train on the eval split.** If you touch preprocessing, it must
  apply identically to train and eval or the metrics are meaningless.
- **Heavy compute goes to Colab** via the `colab-mcp` server registered in
  `.mcp.json` (`uvx git+https://github.com/googlecolab/colab-mcp`). It needs
  approval on first use — restart Claude Code, approve, then the tools appear.
  Local runs use truncated corpora (`max_train`, `max_eval`) for smoke tests.
- **Colab notebooks clone the pushed `main`.** Uncommitted local work is
  invisible to a Colab run. Push before training, or you train stale code.
- **Notebooks are jupytext percent scripts** (`.py` with `# %%`), never
  committed `.ipynb` — notebooks diff badly and carry execution state.
- **`max_train` is applied at training time**, not only at prepare-data time,
  because the cache holds the full split. It samples rather than slices: the
  cache is in corpus order and its leading rows are dominated by a few repos.

## Scope and non-goals

**Allowed:** implementing modules under `src/csne/`, adding configs, adding
tests, adding notebooks that call into the package.

**Non-goals:** multi-language support (Python-first), serving
infrastructure, ANN indexes (exact search is correct at this corpus size).

## Key references and constraints

- Use Hugging Face-hosted CodeSearchNet-style Python data and respect
  dataset licenses.
- Follow CodeSearchNet challenge metrics and structure for evaluation
  (MRR, Recall@k, NDCG for retrieval; BLEU/BERTScore for summarization).
- Use Colab MCP or `claude-colab` for GPU access once training begins —
  do not attempt heavy training locally.
- Aim for experiment logging and reproducibility: results CSVs and
  versioned configs once scaffolding starts.

## Next phases (for later)

- **Phase 4**: Run the full 393K/3-epoch fine-tune (`finetune_minilm.yaml`);
  fine-tune the local HF summarizer (`finetune_summarizer`, currently a
  stub); evaluate summarization (BLEU/BERTScore, `evaluate_summarization`).
- **Phase 5**: Build demo notebooks and/or a Hugging Face Space.
- **Phase 6**: Refine evaluation and write up `REPORT.md`.

## Return format

Always summarize changes made and list modified files when finishing a
task.

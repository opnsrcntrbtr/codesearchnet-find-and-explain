# codesearchnet-find-and-explain: Semantic Code Search and Summarization for Python Code

## 1. Introduction

- The problem of semantic code search: developers spend significant time
  locating relevant code by keyword search, which fails to capture semantic
  intent; this hurts developer productivity.
- Motivation for combining retrieval and summarization ("find" + "explain"):
  a retrieved snippet is more actionable when paired with a concise,
  human-readable explanation of what it does.

## 2. Data

- Two Hugging Face datasets, joined on the GitHub permalink
  (`func_code_url` / `url`), which pins repo + commit sha + line range and is
  therefore unique per function:
  - `code-search-net/code_search_net`, config `python` — 412,178 train /
    23,107 validation / 22,176 test. Supplies the corpus and the official
    splits.
  - `Nan-Do/code-search-net-python` — 455,243 rows, adds a curated one-line
    `summary` column used as the summarization target.
  - Join rate measured at 100% on a 20,000-row sample.
  - Rejected: `sentence-transformers/codesearchnet`, which carries only
    `comment` and `code` — no repo, language, or function name, so neither
    Python filtering nor repo-disjoint splitting is possible.
- Splits: the official CodeSearchNet splits are used unchanged. They are
  already partitioned by repository, which prevents near-identical functions
  from a single repo appearing on both sides, and keeps results comparable
  to published CSN baselines.
- Preprocessing (applied identically to train and eval):
  - Docstrings truncated at the first structured section (reST `:param:`,
    Google-style `Args:`, doctests) to approximate a real search query.
  - Docstrings removed from the code side — they are the retrieval target,
    and leaving them in makes the task trivially lexical.
  - Dropped: docstrings under 3 tokens, boilerplate (`TODO`, `constructor`),
    predominantly non-ASCII docstrings, and functions over 512 tokens.
  - Deduplicated by whitespace-insensitive fingerprint of the normalized
    code, since vendored copies of one function recur across repos.

## 3. Methods

### 3.1 Retrieval model

- Base encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), a single
  shared encoder for both queries and code rather than two towers — small
  enough to fine-tune on a free Colab T4.
- Code is rendered as `func_name + body`: the function name is often the most
  query-like token in a snippet, and MiniLM truncates at 256 tokens, so it
  must precede any risk of truncation.
- Fine-tuned with `MultipleNegativesRankingLoss`, which treats other in-batch
  code snippets as negatives — no explicit negative mining, and larger
  batches directly yield harder negatives.
- Embeddings are L2-normalized, so the index reduces cosine similarity to a
  dot product; retrieval is exact (brute-force matmul) rather than ANN, since
  approximation error would confound the metrics at this corpus size.

### 3.2 Summarization model

- Model choice: a code-capable LLM or Hugging Face summarization model.
- Approach: prompting and/or fine-tuning for function-level summary
  generation.

### 3.3 Colab + Claude orchestration

- Use of Colab MCP / `claude-colab` to run training and evaluation
  notebooks on GPUs.
- Claude Code manages local files, configs, and experiment scripts, and
  coordinates Colab runs.

## 4. Experiments

- Protocol: each query is scored against its gold function plus 999
  distractors, following the CodeSearchNet challenge setup. Ranks are
  1-indexed; a gold that is never ranked counts as a miss (contributing 0 to
  MRR) rather than being dropped.
- Metrics: MRR, Recall@{1,5,10}, NDCG@{1,5,10}. With exactly one relevant
  document, ideal DCG is 1, so NDCG@k reduces to the mean of 1/log2(rank+1).
- Planned comparisons:
  - **BM25** — lexical baseline. Tokenization splits snake_case identifiers,
    so `parse_json_file` matches the query "parse json file"; without this
    the baseline is unfairly weak and the comparison proves nothing.
  - **Un-finetuned MiniLM** — zero-shot embedding baseline.
  - **Fine-tuned MiniLM** — the main result.
  - Summarization: zero-shot API backend vs. fine-tuned local model, scored
    against the curated `summary` column.
- Every run appends a row to `results/retrieval_results.csv` carrying the
  config hash and git commit, so each reported number traces to its run.

## 5. Results & Discussion

Baselines measured on the official CSN Python test split: 1,000 queries
against a 1,000-document corpus (1 gold + 999 distractors). Logged in
`results/retrieval_results.csv`.

| Run | MRR | R@1 | R@5 | R@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 (lexical) | 0.5272 | 0.4070 | 0.6680 | 0.7470 | 0.5739 |
| MiniLM, zero-shot | 0.6872 | 0.5670 | 0.8370 | 0.8980 | 0.7356 |
| MiniLM, fine-tuned (50K pairs, 1 epoch) | **0.7937** | 0.7010 | 0.9140 | 0.9450 | 0.8292 |

Checkpoint: [`opnsrcntrbtrian/csne-minilm-retrieval-50k`](https://huggingface.co/opnsrcntrbtrian/csne-minilm-retrieval-50k)
(private). Colab's local disk is ephemeral and would otherwise have lost it
on the next runtime recycle.

- The un-finetuned encoder already leads BM25 by ~0.16 MRR, the expected
  shape of the result: docstring queries and code bodies share little
  literal vocabulary, which handicaps lexical matching.
- BM25 is not a straw man — it splits snake_case identifiers, so
  `parse_json_file` is reachable from "parse json file". Its R@10 of 0.747
  indicates how much signal identifier names alone carry.
- Fine-tuning on a 50K-pair sample (repo-independent, not a corpus-order
  slice), one epoch, MultipleNegativesRankingLoss, batch 64, on a free Colab
  T4 (~11.6 min, 781 steps, train loss 0.229→0.217) adds **+0.107 MRR**
  (+15.5% relative) over the zero-shot encoder, and +0.267 over BM25.
  R@1 rises from 0.567 to 0.701 — the largest gain is getting the correct
  function to the very top of the list, not just into the top 10.
  The full 393K-pair / 3-epoch run (`finetune_minilm.yaml`) is the next
  experiment and is expected to improve further, though with diminishing
  returns and a real risk of overfitting the in-batch negatives.

### Discussion points to develop

- Whether docstring-as-query overstates real search performance, since the
  docstring is written by the same author as the code it describes.
- Sensitivity of every metric to `distractor_pool_size`.
- Corpus attrition: 412,178 raw train rows yield 393,427 after filtering
  (~4.5% dropped as boilerplate, non-English, over-long, or duplicate).
- Whether the 50K→393K jump mainly buys more in-batch negative diversity
  per epoch, more epochs of exposure, or both — worth an ablation once the
  full run lands.

## 6. Limitations & Future Work

- Language coverage is Python-first; other languages are out of scope
  initially.
- Dataset biases inherited from CodeSearchNet (e.g., repo/library
  popularity skew).
- Potential future extensions: MLX / on-device variants, privacy-focused
  local inference.

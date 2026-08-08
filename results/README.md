# Results

Experiment logs, committed to git so the report's numbers are traceable.

- `retrieval_results.csv` — one row per retrieval run: run name, config hash,
  git commit, MRR, Recall@{1,5,10}, NDCG@{1,5,10}, query count.
- `summarization_results.csv` — one row per summarization run: run name,
  backend, BLEU, BERTScore F1, optional LLM-judge score, example count.

Rows are appended, never overwritten. Bulky per-example dumps go in
`results/*/raw/`, which is gitignored.

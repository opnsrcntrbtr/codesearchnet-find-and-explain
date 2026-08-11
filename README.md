---
title: CodeSearchNet Find & Explain
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
thumbnail: https://huggingface.co/spaces/opnsrcntrbtrian/csne-find-and-explain/resolve/main/thumbnail.png
---

# CodeSearchNet Find & Explain

Search code repositories by natural language description, then get automatic
explanations of what each function does.

## How it works

1. **Retrieval** — A fine-tuned MiniLM-L6 encoder maps your query and the
   CodeSearchNet test corpus into a shared embedding space. Cosine similarity
   retrieves the top-k most relevant code snippets.

2. **Explanation** — Google's Flan-T5 model reads each retrieved function and
   generates a plain-English summary of what it does.

## Models

| Model | Description | MRR (test) |
|---|---|---|
| Full-corpus | Fine-tuned on 393K examples from the entire CodeSearchNet corpus | **0.815** |
| 50K | Fine-tuned on a 50K subset (lighter, faster) | 0.794 |

## Usage

Type a natural-language query describing what you're looking for, pick the
number of results, and choose between Retrieval-only or Find & Explain mode.

## Training

The retrieval encoder was fine-tuned from `sentence-transformers/all-MiniLM-L6-v2`
using triplet loss on the CodeSearchNet Python split. See `configs/finetune_minilm_full_1ep.yaml`
for the full training configuration.

## License

MIT

# CodeSearchNet Find & Explain

A demo of the **CodeSearchNet Find & Explain** pipeline: search code repositories by natural language description, then get automatic explanations of what each function does.

## Features

- **Retrieval**: Search code by natural language query using a fine-tuned MiniLM encoder
- **Find & Explain**: Retrieve top-k code snippets and generate natural-language explanations using a fine-tuned Flan-T5 model
- **Model Toggle**: Switch between the full-corpus model (MRR 0.815) and a lighter 50K model

## Models

| Model | Description |
|-------|-------------|
| Full-corpus | `opnsrcntrbtrian/csne-minilm-retrieval-full-1ep` - Fine-tuned on full CodeSearchNet corpus |
| 50K | `opnsrcntrbtrian/csne-minilm-retrieval-50k` - Smaller, faster inference |

## Usage (Local)

```bash
pip install gradio sentence-transformers torch transformers
python app.py --share
```

## Deployment

This is configured for Hugging Face Spaces. Deploy to HF Spaces by:
1. Creating a new Space with the `gradio` SDK
2. Uploading `app.py`, `requirements.txt`, and the `src/` directory
3. Setting the Python version to 3.11 or later

## Architecture

The pipeline consists of two stages:
1. **Retrieval**: A fine-tuned SentenceTransformer encodes queries and code snippets into a shared embedding space. Similarity search retrieves the top-k matching functions.
2. **Summarization**: A fine-tuned Flan-T5 model generates natural-language explanations of each retrieved function.

## Evaluation

- **Full-corpus retrieval**: MRR 0.815 (+2.1% vs 50K baseline)
- **Summarization**: BLEU 0.56, BERTScore F1 0.747

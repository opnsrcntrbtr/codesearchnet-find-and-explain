"""Gradio demo for CodeSearchNet Find & Explain — HF Space edition.

ZeroGPU-compatible: import spaces before torch, load models at module scope,
wrap inference in @spaces.GPU. Loads CodeSearchNet test data directly from
HF Hub (no local cache), builds an in-memory index on first use.

Deploy to Hugging Face Spaces with sentence-transformers, torch, transformers
in requirements.txt (gradio/spaces/huggingface_hub are preinstalled).
"""

# ZeroGPU rule: import spaces before any torch/CUDA-touching import
try:
    import spaces  # noqa: E402
except ImportError:
    spaces = None  # type: ignore[assignment]

# No-op decorator when spaces is unavailable (local dev)
_gpu = getattr(spaces, "GPU", lambda *a, **k: (lambda f: f)) if spaces else (lambda *a, **k: (lambda f: f))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets import load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRIEVAL_MODEL = "opnsrcntrbtrian/csne-minilm-retrieval-full-1ep"
SUMMARIZER_MODEL = "google/flan-t5-base"
DATASET_ID = "code-search-net/code_search_net"
DATASET_CONFIG = "python"

PROMPT_TEMPLATE = """You are explaining a Python function to a developer who just found it in a
code search and needs to decide whether it does what they want.

Function `{func_name}` from `{repo}` at `{path}`:

```python
{code}
```

Write 1-3 sentences describing what this function does. State its purpose and
its key behavior. Do not restate the signature, do not list parameters, and do
not mention that you are summarizing. If the function's behavior is unclear
from the body alone, say so rather than guessing."""

MODELS = {"full-corpus": RETRIEVAL_MODEL, "50K": "opnsrcntrbtrian/csne-minilm-retrieval-50k"}

# ---------------------------------------------------------------------------
# Lazy-loaded globals (loaded on first inference call)
# ---------------------------------------------------------------------------

_encoder = None
_summarizer = None
_index = None
_examples = None


def _load_dataset():
    """Load CodeSearchNet test split from HF Hub."""
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="test")
    examples = []
    for row in ds:
        url = row["func_code_url"]
        examples.append(
            {
                "id": url,
                "code": row["func_code_string"],
                "docstring": row.get("func_documentation_string", ""),
                "func_name": row["func_name"],
                "repo": row["repository_name"],
                "path": row.get("func_path_in_repository", ""),
                "language": "python",
            }
        )
    return examples


def _build_index():
    """Build the in-memory vector index from test data."""
    global _encoder, _index, _examples

    if _encoder is None:
        print(f"Loading retrieval model: {RETRIEVAL_MODEL}")
        _encoder = SentenceTransformer(RETRIEVAL_MODEL, device="cuda")

    if _examples is None:
        print("Loading CodeSearchNet test data from HF Hub...")
        _examples = _load_dataset()

    if _index is None:
        print("Building vector index...")
        code_strings = [f"{ex['func_name']} {ex['code']}".strip() for ex in _examples]
        embeddings = _encoder.encode(code_strings, batch_size=64, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        _index = {
            "examples": _examples,
            "embeddings": embeddings,
        }

    return _index


def _load_summarizer():
    """Load the flan-t5 summarizer."""
    global _summarizer
    if _summarizer is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Loading summarizer: {SUMMARIZER_MODEL}")
        _summarizer = {
            "tokenizer": AutoTokenizer.from_pretrained(SUMMARIZER_MODEL),
            "model": AutoModelForSeq2SeqLM.from_pretrained(
                SUMMARIZER_MODEL, torch_dtype=torch.bfloat16
            ).to("cuda"),
        }
    return _summarizer


def _encode_query(query: str) -> np.ndarray:
    """Encode a natural-language query to a vector."""
    return _encoder.encode(query, normalize_embeddings=True)


def _search_index(query_vec: np.ndarray, k: int) -> list[dict]:
    """Search the index for top-k similar code snippets."""
    embeddings = _index["embeddings"]
    # Cosine similarity via dot product (both are L2-normalized)
    scores = embeddings @ query_vec
    top_k = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_k, 1):
        ex = _index["examples"][idx]
        results.append(
            {
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "example": ex,
            }
        )
    return results


def _render_prompt(example: dict) -> str:
    """Fill the prompt template for one function."""
    return PROMPT_TEMPLATE.format(
        func_name=example["func_name"],
        repo=example["repo"],
        path=example["path"],
        code=example["code"],
    )


def _summarize_batch(examples: list[dict]) -> list[str]:
    """Summarize a batch of code examples using flan-t5."""
    summarizer = _load_summarizer()
    tokenizer = summarizer["tokenizer"]
    model = summarizer["model"]

    prompts = [_render_prompt(ex) for ex in examples]
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
        "cuda"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=2,
            length_penalty=1.0,
        )

    summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [s.strip() for s in summaries]


# ---------------------------------------------------------------------------
# ZeroGPU-decorated inference functions
# ---------------------------------------------------------------------------


@_gpu(duration=30)
def search_code(query: str, k: int, model_key: str):
    """Retrieve top-k code snippets for a natural language query."""
    if not query.strip():
        return []

    _build_index()
    model_name = MODELS.get(model_key, RETRIEVAL_MODEL)

    # Load and use the selected model
    encoder = SentenceTransformer(model_name, device="cuda")
    query_vec = _encode_query(query)

    # Rebuild index with selected model if needed
    global _index, _encoder
    if _encoder is not None and getattr(_encoder, "model_name_or_path", None) != model_name:
        code_strings = [f"{ex['func_name']} {ex['code']}".strip() for ex in _examples]
        embeddings = encoder.encode(code_strings, batch_size=64, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        _index = {"examples": _examples, "embeddings": embeddings}

    hits = _search_index(query_vec, k)

    results = []
    for hit in hits:
        ex = hit["example"]
        code_snippet = ex["code"][:500] + ("..." if len(ex["code"]) > 500 else "")
        results.append(
            {
                "Rank": hit["rank"],
                "Score": hit["score"],
                "Function": ex["func_name"],
                "Repo": ex["repo"],
                "Code": code_snippet,
            }
        )

    return results


@_gpu(duration=60)
def explain_code(query: str, k: int, model_key: str):
    """Retrieve top-k code snippets and summarize each one."""
    if not query.strip():
        return []

    _build_index()
    model_name = MODELS.get(model_key, RETRIEVAL_MODEL)

    encoder = SentenceTransformer(model_name, device="cuda")
    query_vec = _encode_query(query)

    # Rebuild index with selected model if needed
    global _index, _encoder
    if _encoder is not None and getattr(_encoder, "model_name_or_path", None) != model_name:
        code_strings = [f"{ex['func_name']} {ex['code']}".strip() for ex in _examples]
        embeddings = encoder.encode(code_strings, batch_size=64, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        _index = {"examples": _examples, "embeddings": embeddings}

    hits = _search_index(query_vec, k)
    examples_to_summarize = [hit["example"] for hit in hits]

    summaries = _summarize_batch(examples_to_summarize)

    results = []
    for hit, summary in zip(hits, summaries):
        ex = hit["example"]
        code_snippet = ex["code"][:500] + ("..." if len(ex["code"]) > 500 else "")
        results.append(
            {
                "Rank": hit["rank"],
                "Score": hit["score"],
                "Function": ex["func_name"],
                "Repo": ex["repo"],
                "Code": code_snippet,
                "Summary": summary,
            }
        )

    return results


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------


def build_demo():
    """Build the Gradio demo interface."""

    with gr.Blocks(title="CodeSearchNet Find & Explain") as demo:
        gr.Markdown(
            """
# CodeSearchNet Find & Explain

Search code repositories by natural language description, then get
automatic explanations of what each function does.

**Models:** Toggle between the full-corpus model (MRR 0.815) and
the 50K model (lighter, faster).

*Powered by a fine-tuned MiniLM-L6 retrieval encoder and Google's Flan-T5 summarizer.*
            """
        )

        with gr.Row():
            model_key = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="full-corpus",
                label="Retrieval Model",
            )

        with gr.Row():
            k = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="Number of results (k)",
            )

        query_input = gr.Textbox(
            label="Natural language query",
            placeholder="e.g. 'Parse a JSON file and return the data as a dictionary'",
            lines=2,
        )

        with gr.Tabs():
            # Tab 1: Retrieval only
            with gr.Tab("Retrieval"):
                search_btn = gr.Button("Search", variant="primary")
                search_output = gr.Dataframe(
                    headers=["Rank", "Score", "Function", "Repo", "Code"],
                    label="Retrieved Code Snippets",
                    wrap=True,
                )

            # Tab 2: Find & Explain (retrieval + summarization)
            with gr.Tab("Find & Explain"):
                explain_btn = gr.Button("Explain", variant="primary")
                explain_output = gr.Dataframe(
                    headers=["Rank", "Score", "Function", "Repo", "Code", "Summary"],
                    label="Retrieved Code with Explanations",
                    wrap=True,
                )

        # Wire up buttons
        search_btn.click(
            fn=search_code,
            inputs=[query_input, k, model_key],
            outputs=search_output,
        )

        explain_btn.click(
            fn=explain_code,
            inputs=[query_input, k, model_key],
            outputs=explain_output,
        )

        # Example queries
        gr.Examples(
            examples=[
                ["Parse a JSON file and return the data as a dictionary"],
                ["Calculate the Fibonacci sequence up to n terms"],
                ["Sort a list of dictionaries by a specific key value"],
                ["Read a CSV file and convert it to a list of dictionaries"],
                ["Reverse a linked list in Python"],
            ],
            inputs=query_input,
        )

    return demo


# ---------------------------------------------------------------------------
# HF Spaces entry point
# ---------------------------------------------------------------------------

demo = build_demo()
demo.launch(server_name="0.0.0.0", server_port=7860)

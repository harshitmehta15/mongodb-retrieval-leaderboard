# MongoDB AI Retrieval Challenge Widget

An interactive Jupyter widget (`MongoDBChallenge`) that turns a MongoDB Atlas retrieval pipeline into a hands-on, visual playground — and, optionally, into a scored competition.

Users walk through the four stages of a retrieval pipeline directly in a notebook:

1. **Chunking** — split source documents (fixed / recursive / markdown), tune chunk size and overlap, preview the resulting chunks.
2. **Embedding** — pick an embedding model, output dimensions and quantization dtype, embed the chunks and write them to an Atlas collection (vector index created automatically if missing).
3. **Retrieval** — run vector search or hybrid (vector + full-text) search, tune `top_k`, score threshold and hybrid penalties, and visualize the embedding space (UMAP projection).
4. **Reranking** — optionally rerank retrieved chunks with a Voyage AI reranker.

On top of that, two optional modes can be enabled:

- **Benchmark** — evaluate the current pipeline configuration against a golden dataset and get an **accuracy / latency / cost** score.
- **Scoreboard** — publish benchmark runs to a shared Atlas collection and see a live leaderboard.

Built with [AnyWidget](https://anywidget.dev/) + React, so it works in JupyterLab, Jupyter Notebook, VS Code and Google Colab.

---

## Prerequisites

- An [Atlas cluster](https://www.mongodb.com/docs/atlas/tutorial/create-new-cluster/) on MongoDB 6.0.11 / 7.0.2 or later, with your IP in the [access list](https://www.mongodb.com/docs/atlas/security/ip-access-list/).
- A database and a **dedicated collection** for the chunks. The widget writes and clears documents there — don't point it at a collection you care about.
- A notebook environment with widget support (JupyterLab, Jupyter Notebook, Colab, VS Code).
- An embedding model API key (the built-in cost model covers [Voyage AI](https://www.voyageai.com/) models) and, for reranking, a reranker model.

Vector and full-text search indexes are created by the widget when they don't exist yet. If you create them manually, the embedding field must be named `embedding`.

## Installation

```bash
pip install mongodb-ai-widget-challenge
```

Import path:

```python
from mongodb_ai_widget_challenge import MongoDBChallenge
```

---

## Configurations

The same class covers three levels of setup, and which tabs appear is decided purely by which arguments you pass:

| You pass | Tabs you get |
|---|---|
| `client`, `mongo_collection`, `index_name`, `loader`, `embedding_model` | Chunking · Embedding · Retrieval |
| …plus `reranker` | …with a working Reranking tab |
| …plus **all three** of `benchmark_dataset`, `benchmark_score_weights`, `constraints` | …plus **Benchmark** |
| …plus `benchmark_scoreboard` (all three fields filled) | …plus **Scoreboard** |

### 1a. Minimal playground

One embedding model, no reranker, no benchmark. The smallest thing that works:

```python
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_voyageai import VoyageAIEmbeddings
from mongodb_ai_widget_challenge import MongoDBChallenge

client = MongoClient(ATLAS_URI)
collection = client["documents"]["chunks"]
docs = PyPDFLoader("./mongodb_earnings_4th_quarter_2026.pdf").load()

MongoDBChallenge(
    client=client,
    mongo_collection=collection,
    index_name="vector_index",
    loader=docs,
    embedding_model=VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-4-lite"),
)
```

The Reranking tab is present but has no models to offer, and hybrid search needs `search_index` (below) to work.

### 1b. Full playground: model comparison + hybrid search + preset state

Pass **lists** to get dropdowns users can switch between, and preset the pipeline so the widget opens on a sensible configuration:

```python
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank

widget = MongoDBChallenge(
    client=client,
    mongo_collection=collection,
    index_name="vector_index",
    search_index="search_index",          # enables hybrid (vector + full-text) search
    loader=docs,
    embedding_model=[
        VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-4-lite"),
        VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-4"),
        VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-4-large"),
    ],
    reranker=[
        VoyageAIRerank(voyage_api_key=VOYAGE_KEY, model="rerank-2.5-lite", top_k=3),
        VoyageAIRerank(voyage_api_key=VOYAGE_KEY, model="rerank-2.5", top_k=3),
    ],
    # --- optional starting state (all still editable in the UI) ---
    split_strategy="Recursive",
    chunk_size=512,
    overlap_size=64,
    vector_index_dimensions=1024,
    embedding_output_dtype="float",
    search_sub_tab=2,                     # open on hybrid search
    top_k=10,
    reranking_enabled=True,
    rerank_top_k=5,
    rag_query="How much did Atlas revenue grow?",
)
widget
```

### 2. Playground + Benchmark

The **Benchmark** tab appears only when **all three** of these are passed:

- `benchmark_dataset`
- `benchmark_score_weights`
- `constraints`

Omit any one of them and the tab stays hidden.

```python
widget = MongoDBChallenge(
    client=client,
    mongo_collection=collection,
    index_name="vector_index",
    search_index="search_index",
    loader=docs,
    embedding_model=embedding_models,
    reranker=rerankers,
    benchmark_dataset="golden_dataset.json",     # dict, file path, or https URL
    benchmark_score_weights={
        "accuracy": 0.4,
        "latency": 0.2,
        "cost": 0.4,
    },
    constraints={
        "min_k": 5,
        "max_k": 15,
        "min_chunk_size": 125,
        "max_chunk_size": 1500,
    },
)
```

`benchmark_dataset` accepts three forms:

```python
benchmark_dataset="golden_dataset.json"                    # path, relative to the notebook's cwd
benchmark_dataset="https://example.com/golden.json"        # fetched over HTTP(S), 30s timeout
benchmark_dataset={"sources": [...], "seeds": [...], "examples": [...]}   # inline dict
```

Weights don't have to sum to 1 — they're normalized. These two are identical, and both mean "accuracy counts double":

```python
benchmark_score_weights={"accuracy": 0.5, "latency": 0.25, "cost": 0.25}
benchmark_score_weights={"accuracy": 2, "latency": 1, "cost": 1}
```

To take a dimension out of the score entirely, weight it `0`:

```python
benchmark_score_weights={"accuracy": 1, "latency": 0, "cost": 0}   # pure accuracy contest
```

`constraints` is what keeps a competition fair — the UI won't let users go outside these bounds. Use `None` for an open-ended `k`:

```python
constraints={"min_k": 1, "max_k": None, "min_chunk_size": 100, "max_chunk_size": 2000}
```

### 3. Playground + Benchmark + Scoreboard (full challenge)

The **Scoreboard** tab appears only when the Benchmark tab is enabled **and** `benchmark_scoreboard` is passed with a usable `mdb_uri`, `database` and `collection`.

```python
widget = MongoDBChallenge(
    ...,                                          # everything from level 2
    benchmark_scoreboard={
        "mdb_uri": CHALLENGE_URI,                 # cluster hosting the shared leaderboard
        "database": "documents",
        "collection": "scoreboard",
    },
)
```

Participants type a name and publish their run. The leaderboard is filtered by the exact score weights **and** constraints in use, so only runs from the same challenge setup appear on the same board — change either one and you effectively start a new board.

The scoreboard cluster can be a different cluster from the one being indexed; `mdb_uri` is a full connection string. Typically it's a shared cluster the organizer owns, while each participant indexes into their own.

A complete challenge, end to end:

```python
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from mongodb_ai_widget_challenge import MongoDBChallenge

# In Colab, read these from the Secrets panel rather than hardcoding them
client = MongoClient(ATLAS_URI)
collection = client["documents"]["chunks"]
docs = PyPDFLoader("https://investors.mongodb.com/node/14081/pdf").load()

MongoDBChallenge(
    client=client,
    mongo_collection=collection,
    index_name="vector_index",
    search_index="search_index",
    loader=docs,
    embedding_model=[
        VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-4-lite"),
        VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-4"),
        VoyageAIEmbeddings(voyage_api_key=VOYAGE_KEY, model="voyage-4-large"),
    ],
    reranker=[
        VoyageAIRerank(voyage_api_key=VOYAGE_KEY, model="rerank-2.5-lite", top_k=3),
        VoyageAIRerank(voyage_api_key=VOYAGE_KEY, model="rerank-2.5", top_k=3),
    ],
    benchmark_dataset="https://gist.githubusercontent.com/.../golden_dataset.json",
    benchmark_score_weights={"accuracy": 0.4, "latency": 0.2, "cost": 0.4},
    constraints={
        "min_k": 5,
        "max_k": 15,
        "min_chunk_size": 125,
        "max_chunk_size": 1500,
    },
    benchmark_scoreboard={
        "mdb_uri": CHALLENGE_URI,
        "database": "documents",
        "collection": "scoreboard",
    },
)
```

The official challenge notebook is distributed separately (Colab), not from this repo.

### 4. Workshop mode: attendees bring their own credentials

Set `override_settings_panel=True` to expose a panel where each attendee pastes their own model API key and MongoDB URI into the running widget. Useful when you ship one notebook to a room full of people:

```python
MongoDBChallenge(
    client=client,
    mongo_collection=collection,
    index_name="vector_index",
    loader=docs,
    embedding_model=VoyageAIEmbeddings(voyage_api_key=PLACEHOLDER_KEY, model="voyage-4-lite"),
    override_settings_panel=True,
)
```

This can be combined with any of the levels above.

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client` | `pymongo.MongoClient` | `None` | Client for the cluster holding the chunk collection. Used for index creation and connection checks. |
| `mongo_collection` | pymongo `Collection` | `None` | Target collection for embedded chunks. Contents are managed by the widget. |
| `index_name` | `str` | `None` | Atlas **Vector Search** index name. Created automatically if missing. |
| `search_index` | `str` | `"search_index"` | Atlas **full-text** search index name, used by hybrid search. (Legacy alias: `text_search_index`.) |
| `loader` | list of LangChain `Document`, or a loader | `None` | Source documents. Pass either `loader.load()` output or a LangChain loader. |
| `embedding_model` | embedding model, or list | `None` | One or several LangChain embedding models. A list becomes a dropdown so users can compare models; the first one is selected initially. |
| `reranker` | reranker, or list | `None` | One or several rerankers (e.g. `VoyageAIRerank`). A list becomes a dropdown. Reranking stays off until enabled in the UI. |
| `benchmark_dataset` | `dict` \| path `str` \| `https://` URL | `None` | Golden dataset (format below). Enables the Benchmark tab together with the two parameters below. |
| `benchmark_score_weights` | `dict` | `{accuracy: ⅓, latency: ⅓, cost: ⅓}` | Relative weights of the three score components. Normalized to sum to 1, so `{"accuracy": 2, "latency": 1, "cost": 1}` is valid too. |
| `constraints` | `dict` | `{min_k: None, max_k: None, min_chunk_size: 1, max_chunk_size: 2000}` | Bounds enforced in the UI, so every participant tunes within the same envelope. Keys: `min_k`, `max_k`, `min_chunk_size`, `max_chunk_size`; `min_k`/`max_k` may be `None` for unbounded. (Legacy alias: `benchmark_constraints`.) |
| `benchmark_scoreboard` | `dict` | `None` | `{"mdb_uri": ..., "database": ..., "collection": ...}`. All three must be non-empty to enable the Scoreboard tab. |
| `override_settings_panel` | `bool` | `False` | Shows an in-UI panel letting users override the model API key and MongoDB URI at runtime. Useful for workshops where attendees bring their own credentials. |

### Presetting the pipeline (optional)

Every UI control is a synced widget trait, and the constructor forwards unknown keyword arguments to the widget base class. So any of these can be passed to set the **starting state** of the UI — users can still change them in the widget afterwards.

| Keyword | Type | Default | Description |
|---|---|---|---|
| `split_strategy` | `str` | `"Fixed"` | Chunking strategy: `"Fixed"`, `"Recursive"` or `"Markdown"`. |
| `chunk_size` | `int` | `1024` | Characters per chunk. Clamped to the `constraints` chunk-size bounds. |
| `overlap_size` | `int` | `0` | Character overlap between consecutive chunks. |
| `current_step` | `int` | `1` | Tab shown first: 1 Chunking, 2 Embedding, 3 Retrieval, 4 Reranking. |
| `vector_index_dimensions` | `int` | `1024` | Embedding dimensions. One of `256`, `512`, `1024`, `2048`. Must match the vector index. |
| `embedding_output_dtype` | `str` | `"float"` | Quantization: `"float"`, `"int8"`, `"uint8"`, `"binary"`, `"ubinary"`. |
| `selected_embedding_index` | `int` | `0` | Which model from the `embedding_model` list is used for indexing. |
| `query_embedding_index` | `int` | `0` | Which model from the list is used to embed queries. |
| `search_sub_tab` | `int` | `1` | Retrieval mode: `1` = vector search, `2` = hybrid search. |
| `top_k` | `int` | `5` | Number of chunks retrieved. Clamped to the `constraints` `min_k`/`max_k`. |
| `score_threshold` | `float` \| `None` | `None` | Minimum similarity score for a result to be kept. `None` = no threshold. |
| `fulltext_penalty` | `int` | `50` | Hybrid search reciprocal-rank-fusion penalty for the full-text leg. |
| `vector_penalty` | `int` | `50` | Hybrid search RRF penalty for the vector leg. |
| `reranking_enabled` | `bool` | `False` | Start with reranking on. Requires a `reranker`. |
| `selected_reranker_index` | `int` | `0` | Which model from the `reranker` list is active. |
| `rerank_top_k` | `int` | `3` | Documents kept after reranking. |
| `rag_query` | `str` | `""` | Pre-fill the question box. |
| `scoreboard_entry_name` | `str` | `""` | Pre-fill the participant name used when publishing a score. |

### Deprecated / accepted for compatibility

| Keyword | Status |
|---|---|
| `text_search_index` | Old name for `search_index`; still honored, `search_index` wins if both are given. |
| `benchmark_constraints` | Old name for `constraints`; still honored and still counts toward enabling the Benchmark tab. |
| `llm` | Silently ignored. The widget no longer generates answers — it scores retrieval only. |

### Golden dataset format

`benchmark_dataset` must resolve to a JSON object with three lists:

```json
{
  "sources": [
    { "source_id": "doc.pdf::page_0", "title": "...", "text": "full page text" }
  ],
  "seeds": [
    { "seed_id": "doc.pdf::page_0::seed_ab12", "source_id": "doc.pdf::page_0",
      "title": "...", "text": "passage the answer comes from" }
  ],
  "examples": [
    { "qid": "5ad6d6dd8447",
      "question": "By what percentage did Atlas revenue grow?",
      "answer": "29%",
      "supporting_quote": "Atlas Revenue up 29% year-over-year ...",
      "relevant_source_id": "doc.pdf::page_0",
      "relevant_seed_id": "doc.pdf::page_0::seed_ab12",
      "seed_text": "..." }
  ]
}
```

`sources` are re-chunked with the user's current chunking settings; `examples` supply the questions and the ground-truth passages. `seeds` are the ground-truth passages a question's answer comes from, and are what recall and nDCG are measured against.

### How the score is computed

Each benchmark run produces three 0–100 sub-scores, combined into an overall score with `benchmark_score_weights`:

- **Accuracy** — `100 × (0.6 × mean recall + 0.4 × mean nDCG)` over the dataset questions. Reranked results are used when reranking is enabled, otherwise raw retrieval results.
- **Latency** — log-descending score over end-to-end query latency (best ≈ 300 ms, worst ≈ 2000 ms).
- **Cost** — log-descending score over estimated $ cost: embedding tokens + rerank tokens + LLM context tokens, priced with the built-in Voyage AI price table and a $2.50/M LLM context rate.

Published scoreboard documents look like:

```json
{ "name": "...", "overall_score": 78.4, "accuracy_score": 81.2,
  "latency_score": 74.0, "cost_score": 79.5,
  "score_weights": {"accuracy": 0.4, "latency": 0.2, "cost": 0.4},
  "constraints": {"min_k": 5, "max_k": 15, "min_chunk_size": 125, "max_chunk_size": 1500},
  "score_weights_key": "...", "constraints_key": "...",
  "published_at": "2026-08-04 12:00:00 UTC" }
```

---

## Development

```bash
git clone https://github.com/mongodb-developer/mongodb-ai-widgets.git
cd mongodb-ai-widgets

python -m venv newenv
source newenv/bin/activate      # Windows: newenv\Scripts\activate
pip install -e ".[dev]"
```

Frontend (React, bundled with esbuild into `index.js`, which is what ships in the wheel):

```bash
cd mongodb_ai_widget_challenge
npm install
npm run build          # one-off bundle
npm run build:watch    # rebuild on save while developing
```

Then run any notebook that builds a `MongoDBChallenge` and re-execute the widget cell to pick up frontend changes. Python changes require a kernel restart.

### Project layout

```
mongodb_ai_widget_challenge/     # the published package
├── rag_widget.py                # MongoDBChallenge: pipeline, benchmark, scoreboard
├── index.jsx                    # React source of the widget UI
├── index.js                     # esbuild bundle (committed, shipped in the wheel)
├── index.css                    # widget styles
├── package.json                 # frontend build
└── __init__.py                  # exports MongoDBChallenge
```

## Build

`build.sh` bundles the frontend and builds the sdist + wheel. It requires an activated virtual environment.

```bash
source newenv/bin/activate
./build.sh
```

It runs `npm install && npm run build` in `mongodb_ai_widget_challenge/`, removes stale `dist/mongodb_ai_widget_challenge-*` artifacts, then runs `python -m build`. Results land in `dist/`.

## Publish to PyPI

1. Bump `version` in `pyproject.toml` (PyPI rejects re-uploading an existing version).
2. Rebuild: `./build.sh`.
3. Publish:

   ```bash
   ./publish.sh
   ```

`publish.sh` reads the version from `pyproject.toml`, checks that the matching wheel and sdist exist in `dist/`, and uploads them with `twine`. Authenticate with an API token — username `__token__`, password `pypi-...` — at the prompt, via `~/.pypirc`, or via env vars:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxxxxx
./publish.sh
```

To dry-run against TestPyPI, add `--repository testpypi` to the `twine upload` line in `publish.sh`.

---

## License

[MIT](LICENSE)

## Acknowledgments

[AnyWidget](https://anywidget.dev/) · [LangChain](https://python.langchain.com/) · [langchain-mongodb](https://github.com/langchain-ai/langchain-mongodb) · [PyMongo](https://pymongo.readthedocs.io/) · [Voyage AI](https://www.voyageai.com/) · [MongoDB Atlas](https://www.mongodb.com/atlas)

## Additional Resources

- [GenAI Showcase](https://github.com/mongodb-developer/GenAI-Showcase)
- [AI Learning Hub](https://www.mongodb.com/resources/use-cases/artificial-intelligence)
- [GenAI Community Forum](https://www.mongodb.com/community/forums/c/generative-ai/162)

# rag_widget.py
# ------------------------------------------------
# Python: MongoDBRAGPlayground (updated with top_k and score_threshold)
# ------------------------------------------------
import pathlib
import json
import math
import hashlib
import time
import re
import os
import threading
import queue
from urllib.parse import urlparse
import anywidget
import traitlets
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import requests
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

try:
    import tiktoken
except ImportError:
    tiktoken = None

current_directory = pathlib.Path().resolve()

VOYAGE_EMBEDDING_PRICES_PER_MILLION = {
    "voyage-4-large": 0.12,
    "voyage-4": 0.06,
    "voyage-4-lite": 0.02,
    "voyage-context-3": 0.18,
    "voyage-context-4": 0.18,
    "voyage-3-large": 0.18,
    "voyage-3.5": 0.06,
    "voyage-3.5-lite": 0.02,
    "voyage-code-3": 0.18,
    "voyage-finance-2": 0.12,
    "voyage-law-2": 0.12,
    "voyage-code-2": 0.12,
    "voyage-multimodal-3": 0.12,
    "voyage-multimodal-3.5": 0.12,
    "voyage-3": 0.06,
    "voyage-3-lite": 0.02,
    "voyage-large-2-instruct": 0.12,
    "voyage-multilingual-2": 0.12,
    "voyage-large-2": 0.12,
    "voyage-02": 0.10,
    "voyage-2": 0.10,
    "voyage-lite-02-instruct": 0.10,
    "voyage-lite-01": 0.10,
    "voyage-lite-01-instruct": 0.10,
    "voyage-01": 0.10,
}

VOYAGE_RERANK_PRICES_PER_MILLION = {
    "rerank-2.5": 0.05,
    "rerank-2.5-lite": 0.02,
    "rerank-2": 0.05,
    "rerank-2-lite": 0.02,
    "rerank-lite-1": 0.02,
    "rerank-1": 0.05,
}

LLM_CONTEXT_PRICE_PER_MILLION = 2.50
LATENCY_SCORE_BEST_MS = 300.0
LATENCY_SCORE_WORST_MS = 2000.0
SOFT_COST_SCORE_POWER = 1.0
LLM_CONTEXT_MIN_TOKENS = 500
LLM_CONTEXT_MAX_DOCUMENTS = 50
DEFAULT_BENCHMARK_SCORE_WEIGHTS = {
    "accuracy": 1.0 / 3.0,
    "latency": 1.0 / 3.0,
    "cost": 1.0 / 3.0,
}
DEFAULT_BENCHMARK_CONSTRAINTS = {
    "min_k": None,
    "max_k": None,
    "min_chunk_size": 1,
    "max_chunk_size": 2000,
}
DEFAULT_BENCHMARK_SCOREBOARD = {
    "mdb_uri": "",
    "database": "",
    "collection": "",
}
ALLOWED_VECTOR_INDEX_DIMENSIONS = [256, 512, 1024, 2048]
DEFAULT_VECTOR_INDEX_DIMENSIONS = 1024
ALLOWED_EMBEDDING_OUTPUT_DTYPES = ["float", "int8", "uint8", "binary", "ubinary"]
DEFAULT_EMBEDDING_OUTPUT_DTYPE = "float"
AUTO_INDEX_TIMEOUT_SECONDS = 90
CONFIG_CHECK_MONGODB_TIMEOUT_SECONDS = 5
CONFIG_CHECK_EMBEDDING_TIMEOUT_SECONDS = 8


class DimensionAwareEmbeddings:
    """Apply Voyage embedding options and normalize vectors for Atlas."""

    def __init__(self, base_embedding, dimensions, output_dtype=DEFAULT_EMBEDDING_OUTPUT_DTYPE):
        self._base_embedding = base_embedding
        self._dimensions = int(dimensions)
        self._output_dtype = (
            output_dtype
            if output_dtype in ALLOWED_EMBEDDING_OUTPUT_DTYPES
            else DEFAULT_EMBEDDING_OUTPUT_DTYPE
        )

    def _output_dtype_arg(self):
        return self._output_dtype or DEFAULT_EMBEDDING_OUTPUT_DTYPE

    def _configured_base_embedding(self):
        model = self._base_embedding
        fields = getattr(model, "model_fields", None) or {}
        updates = {}
        if "output_dimension" in fields:
            updates["output_dimension"] = self._dimensions
        if "output_dtype" in fields:
            updates["output_dtype"] = self._output_dtype_arg()

        if updates and hasattr(model, "model_copy"):
            try:
                return model.model_copy(update=updates)
            except Exception:
                return model
        return model

    def _can_embed_voyage_directly(self):
        client = getattr(self._base_embedding, "_client", None)
        return (
            client is not None
            and hasattr(client, "embed")
            and hasattr(self._base_embedding, "model")
        )

    def _voyage_batches(self, texts):
        build_batches = getattr(self._base_embedding, "_build_batches", None)
        if callable(build_batches):
            yield from build_batches(texts)
            return
        yield texts, len(texts)

    def _embed_voyage_regular(self, texts, input_type):
        embeddings = []
        client = self._base_embedding._client
        model = getattr(self._base_embedding, "model")
        truncation = getattr(self._base_embedding, "truncation", True)
        for batch, _ in self._voyage_batches(texts):
            result = client.embed(
                batch,
                model=model,
                input_type=input_type,
                truncation=truncation,
                output_dimension=self._dimensions,
                output_dtype=self._output_dtype_arg(),
            )
            embeddings.extend(result.embeddings)
        return embeddings

    def _embed_voyage_context(self, texts, input_type):
        embeddings = []
        client = self._base_embedding._client
        model = getattr(self._base_embedding, "model")
        for batch, _ in self._voyage_batches(texts):
            result = client.contextualized_embed(
                inputs=[batch],
                model=model,
                input_type=input_type,
                output_dimension=self._dimensions,
                output_dtype=self._output_dtype_arg(),
            )
            embeddings.extend(result.results[0].embeddings)
        return embeddings

    def _embed_voyage_documents(self, texts):
        is_context_model = getattr(self._base_embedding, "_is_context_model", None)
        if callable(is_context_model) and is_context_model():
            return self._embed_voyage_context(texts, "document")
        return self._embed_voyage_regular(texts, "document")

    def _embed_voyage_query(self, text):
        is_context_model = getattr(self._base_embedding, "_is_context_model", None)
        if callable(is_context_model) and is_context_model():
            result = self._embed_voyage_context([text], "query")
        else:
            result = self._embed_voyage_regular([text], "query")
        return result[0]

    def _normalize_vector(self, vector):
        if isinstance(vector, (bytes, bytearray)):
            vector = list(vector)
        values = [float(value) for value in (vector or [])]
        if len(values) > self._dimensions:
            return values[: self._dimensions]
        if len(values) < self._dimensions:
            return values + [0.0] * (self._dimensions - len(values))
        return values

    def embed_query(self, text):
        if self._can_embed_voyage_directly():
            return self._normalize_vector(self._embed_voyage_query(text))
        return self._normalize_vector(self._configured_base_embedding().embed_query(text))

    def embed_documents(self, texts):
        if self._can_embed_voyage_directly():
            return [
                self._normalize_vector(vector)
                for vector in self._embed_voyage_documents(texts)
            ]
        return [
            self._normalize_vector(vector)
            for vector in self._configured_base_embedding().embed_documents(texts)
        ]

    def __getattr__(self, name):
        return getattr(self._base_embedding, name)

class MongoDBChallenge(anywidget.AnyWidget):
    _esm = str(pathlib.Path(__file__).parent / "index.js")
    _css = str(pathlib.Path(__file__).parent / "index.css")

    # Which step is selected in the UI
    # (1=Chunking, 2=Embedding, 3=Retrieval, 4=Reranking)
    current_step = traitlets.Int(1).tag(sync=True)

    # Chunking controls
    split_strategy = traitlets.Unicode("Fixed").tag(sync=True)
    chunk_size = traitlets.Int(1024).tag(sync=True)
    overlap_size = traitlets.Int(0).tag(sync=True)
    current_doc_index = traitlets.Int(0).tag(sync=True)
    document_preview = traitlets.Unicode("").tag(sync=True)
    chunks_table = traitlets.List(traitlets.Dict()).tag(sync=True)

    # Step 2 controls
    selected_index = traitlets.Unicode("").tag(sync=True)
    embeddings_table = traitlets.List(traitlets.Dict()).tag(sync=True)
    mongo_docs_table = traitlets.List(traitlets.Dict()).tag(sync=True)
    indexing_status = traitlets.Dict(default_value={}).tag(sync=True)
    embedding_ready = traitlets.Bool(False).tag(sync=True)
    loaded_in_mongo = traitlets.Bool(False).tag(sync=True)
    vector_indexes = traitlets.List(traitlets.Unicode()).tag(sync=True)
    vector_index_dimension_options = traitlets.List(
        traitlets.Int(), default_value=list(ALLOWED_VECTOR_INDEX_DIMENSIONS)
    ).tag(sync=True)
    vector_index_dimensions = traitlets.Int(DEFAULT_VECTOR_INDEX_DIMENSIONS).tag(sync=True)
    embedding_output_dtype_options = traitlets.List(
        traitlets.Unicode(), default_value=list(ALLOWED_EMBEDDING_OUTPUT_DTYPES)
    ).tag(sync=True)
    embedding_output_dtype = traitlets.Unicode(DEFAULT_EMBEDDING_OUTPUT_DTYPE).tag(sync=True)

    # Display name for the embedding model (for backward compatibility)
    embedding_model = traitlets.Unicode("").tag(sync=True)

    # List of available embedding model names for dropdown
    embedding_models = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)

    # Index of the currently selected embedding model (for indexing)
    selected_embedding_index = traitlets.Int(0).tag(sync=True)

    # Index of the currently selected embedding model for query time
    query_embedding_index = traitlets.Int(0).tag(sync=True)

    # DB and collection names, inferred from mongo_collection
    mongo_db_name = traitlets.Unicode("").tag(sync=True)
    mongo_collection_name = traitlets.Unicode("").tag(sync=True)

    # Query controls
    search_sub_tab = traitlets.Int(1).tag(sync=True)  # 1=Vector Search, 2=Hybrid Search
    rag_query = traitlets.Unicode("").tag(sync=True)  # The user's question
    rag_results = traitlets.List(traitlets.Dict()).tag(sync=True)  # Vector search results
    query_embedding = traitlets.List(traitlets.Float()).tag(sync=True)  # Query embedding vector
    all_doc_embeddings = traitlets.List(traitlets.Dict()).tag(sync=True)  # All document embeddings for visualization
    retrieval_status = traitlets.Dict(default_value={}).tag(sync=True)

    # RAG TRAITLETS for k and threshold
    top_k = traitlets.Int(5).tag(sync=True)
    score_threshold = traitlets.Union(
        [traitlets.Float(), traitlets.Float(allow_none=True), traitlets.Unicode(allow_none=True)],
        default_value=None,
        allow_none=True
    ).tag(sync=True)
    fulltext_penalty = traitlets.Int(50).tag(sync=True)
    vector_penalty = traitlets.Int(50).tag(sync=True)

    # Reranking controls
    reranking_enabled = traitlets.Bool(False).tag(sync=True)
    reranker_models = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    selected_reranker_index = traitlets.Int(0).tag(sync=True)
    rerank_top_k = traitlets.Int(3).tag(sync=True)
    rerank_results = traitlets.List(traitlets.Dict()).tag(sync=True)

    # Benchmark controls and results
    benchmark_dataset_name = traitlets.Unicode("").tag(sync=True)
    benchmark_dataset_loaded = traitlets.Bool(False).tag(sync=True)
    benchmark_summary = traitlets.Dict(default_value={}).tag(sync=True)
    benchmark_results = traitlets.List(traitlets.Dict()).tag(sync=True)
    benchmark_running = traitlets.Bool(False).tag(sync=True)
    benchmark_status = traitlets.Dict(default_value={}).tag(sync=True)
    configuration_status = traitlets.Dict(default_value={}).tag(sync=True)
    benchmark_selected_qid = traitlets.Unicode("").tag(sync=True)
    benchmark_score_weights = traitlets.Dict(
        default_value=dict(DEFAULT_BENCHMARK_SCORE_WEIGHTS)
    ).tag(sync=True)
    constraints = traitlets.Dict(
        default_value=dict(DEFAULT_BENCHMARK_CONSTRAINTS)
    ).tag(sync=True)
    benchmark_tab_visible = traitlets.Bool(False).tag(sync=True)
    scoreboard_tab_visible = traitlets.Bool(False).tag(sync=True)
    scoreboard_entries = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    scoreboard_loading = traitlets.Bool(False).tag(sync=True)
    scoreboard_available = traitlets.Bool(False).tag(sync=True)
    scoreboard_entry_name = traitlets.Unicode("").tag(sync=True)
    override_settings_panel = traitlets.Bool(False).tag(sync=True)
    override_model_api_key = traitlets.Unicode("").tag(sync=True)
    override_mongodb_uri = traitlets.Unicode("").tag(sync=True)
    overrides_status = traitlets.Unicode("").tag(sync=True)

    # Command and error
    command = traitlets.Unicode("").tag(sync=True)
    error = traitlets.Unicode("").tag(sync=True)

    def set_error(self, message):
        """Set an error message and ensure it's sent to the frontend"""
        if message:
            self.error = message
            self.send({"type": "update_error", "error": message})

    def clear_error(self):
        """Clear any error messages"""
        self.error = ""
        self.send({"type": "update_error", "error": ""})

    def set_overrides_status(self, message):
        self.overrides_status = message or ""

    def test_error(self, message="This is a test error message"):
        """Function to test the error display system"""
        self.set_error(message)

    def __init__(
        self,
        client=None,
        loader=None,
        embedding_model=None,
        reranker=None,
        benchmark_dataset=None,
        benchmark_score_weights=None,
        constraints=None,
        benchmark_scoreboard=None,
        search_index=None,
        mongo_collection=None,
        index_name=None,
        override_settings_panel=False,
        **kwargs,
    ):
        """
        loader: optional doc loader
        embedding_model: single embedding model or list of embedding models for vectorstore
        reranker: single reranker model or list of reranker models
        benchmark_dataset: dict or JSON path following the golden dataset format
        search_index: optional Atlas Search index name used for hybrid full-text search
        mongo_collection: pymongo collection for storing embedded docs
        index_name: name of the MongoDB Atlas Search index to use
        override_settings_panel: show the override settings button/panel when True; hidden by default
        """
        # Keep the actual Python embedding object separate
        self.client = client
        self._mongo_client = client
        self.loader = loader
        self.mongo_collection = mongo_collection
        legacy_text_search_index = kwargs.pop("text_search_index", None)
        self._text_search_index = search_index or legacy_text_search_index or "search_index"
        # Backward compatibility: ignore deprecated llm kwarg if passed.
        kwargs.pop("llm", None)

        # Handle both single embedding model and list of models
        if isinstance(embedding_model, list):
            self._embedding_models = embedding_model  # internal Python objects
            self._embedding_model = embedding_model[0] if embedding_model else None
        else:
            self._embedding_models = [embedding_model] if embedding_model else []
            self._embedding_model = embedding_model  # internal Python object

        if isinstance(reranker, list):
            self._rerankers = reranker
            self._reranker = reranker[0] if reranker else None
        else:
            self._rerankers = [reranker] if reranker else []
            self._reranker = reranker

        self._benchmark_dataset_raw = benchmark_dataset
        self._benchmark_dataset = {"sources": [], "seeds": [], "examples": []}
        self._benchmark_sources_by_id = {}
        self._benchmark_seeds_by_id = {}
        self._token_encoding = None
        self._benchmark_score_weights = self._normalize_benchmark_score_weights(
            benchmark_score_weights
        )
        legacy_constraints = kwargs.pop("benchmark_constraints", None)
        has_explicit_benchmark_constraints = (
            constraints is not None or legacy_constraints is not None
        )
        self._constraints = self._normalize_constraints(
            constraints if constraints is not None else legacy_constraints
        )
        self._benchmark_scoreboard = self._normalize_benchmark_scoreboard(
            benchmark_scoreboard
        )
        self._benchmark_tab_visible = (
            benchmark_dataset is not None
            and benchmark_score_weights is not None
            and has_explicit_benchmark_constraints
        )
        self._scoreboard_tab_visible = (
            self._benchmark_tab_visible
            and benchmark_scoreboard is not None
        )
        self._scoreboard_client = None
        self._scoreboard_collection = None
        self._mongo_database_name = ""
        self._mongo_collection_name = ""

        # Store the index name for later use
        self._index_name = index_name

        try:
            if hasattr(loader, "load"):
                docs = loader.load()
            elif isinstance(loader, list):
                docs = loader
            else:
                docs = []
            self.loaded_pages = []
            self.loaded_page_metadata = []
            for doc in docs:
                if hasattr(doc, "page_content"):
                    self.loaded_pages.append(doc.page_content)
                    metadata = getattr(doc, "metadata", {}) or {}
                    self.loaded_page_metadata.append(metadata if isinstance(metadata, dict) else {})
                else:
                    self.loaded_pages.append(str(doc))
                    self.loaded_page_metadata.append({})
        except Exception as e:
            self.loaded_pages = []
            self.loaded_page_metadata = []
            # Don't set error yet, wait until super().__init__ is called

        self.chunks_by_page = []

        # Initialize parent class first so traitlets are properly set up
        super().__init__(**kwargs)

        # Now set traitlet values after initialization
        self.benchmark_score_weights = dict(self._benchmark_score_weights)
        self.constraints = dict(self._constraints)
        self.benchmark_tab_visible = bool(self._benchmark_tab_visible)
        self.scoreboard_tab_visible = bool(self._scoreboard_tab_visible)
        self.scoreboard_available = self._has_scoreboard_config()
        self.override_settings_panel = bool(override_settings_panel)
        self._apply_constraints()

        # Infer DB and collection name from the provided pymongo collection
        if self.mongo_collection is not None:
            try:
                self._sync_mongo_namespace_from_collection()
            except Exception:
                # Leave them as empty strings if something goes wrong
                self.mongo_db_name = ""
                self.mongo_collection_name = ""

        # Human-readable model names for the embedding model UI
        model_display_names = []
        for model in self._embedding_models:
            display_name = ""
            try:
                if isinstance(model, str):
                    display_name = model
                elif hasattr(model, "model"):
                    display_name = str(getattr(model, "model"))
                elif hasattr(model, "model_name"):
                    display_name = str(getattr(model, "model_name"))
                elif model is not None:
                    display_name = model.__class__.__name__
            except Exception as e:
                display_name = f"Unknown Model {len(model_display_names) + 1}"

            # Ensure we have a valid string
            if not display_name or not isinstance(display_name, str):
                display_name = f"Model {len(model_display_names) + 1}"

            model_display_names.append(display_name)

        # Always ensure we have at least an empty list, never None
        self.embedding_models = model_display_names if model_display_names else []

        # Set the current embedding_model for backward compatibility
        # This should be a string, not the list
        if model_display_names:
            self.embedding_model = str(model_display_names[0])
        else:
            self.embedding_model = ""

        reranker_display_names = []
        for model in self._rerankers:
            display_name = ""
            try:
                if isinstance(model, str):
                    display_name = model
                elif hasattr(model, "model"):
                    display_name = str(getattr(model, "model"))
                elif hasattr(model, "model_name"):
                    display_name = str(getattr(model, "model_name"))
                elif model is not None:
                    display_name = model.__class__.__name__
            except Exception:
                display_name = f"Reranker {len(reranker_display_names) + 1}"

            if not display_name or not isinstance(display_name, str):
                display_name = f"Reranker {len(reranker_display_names) + 1}"

            reranker_display_names.append(display_name)

        self.reranker_models = reranker_display_names if reranker_display_names else []

        if self._index_name:
            self.selected_index = self._index_name

        self._create_chunks_for_all_pages()
        self._update_highlighted_preview()
        self._build_embeddings_table()
        self.mongo_docs_table = []
        self._reset_indexing_state()
        self._load_benchmark_dataset()
        self._reset_benchmark_status()
        self._reset_configuration_status()
        self._reset_retrieval_status()
        # Run the preflight inline: state changes pushed from a background thread
        # before the frontend comm is connected never reach the browser, which
        # left the UI stuck on the first "Checking MongoDB..." message.
        self._run_configuration_preflight()

    # -----------------
    # Step 1: CHUNKING
    # -----------------
    def _create_chunks_for_all_pages(self):
        self.chunks_by_page = []
        table_accumulator = []

        if not self.loaded_pages:
            self.chunks_table = []
            return

        splitter = self._get_text_splitter(add_start_index=True)
        for p_idx, page_text in enumerate(self.loaded_pages):
            page_metadata = dict(
                (
                self.loaded_page_metadata[p_idx]
                if p_idx < len(self.loaded_page_metadata)
                else {}
                )
                or {}
            )
            docs = splitter.create_documents([page_text], metadatas=[page_metadata])
            for c_idx, doc in enumerate(docs):
                chunk_id = self._make_chunk_id(
                    metadata=doc.metadata,
                    page_index=p_idx,
                    chunk_index=c_idx,
                    start_offset=doc.metadata["start_index"],
                    chunk_text=doc.page_content,
                )
                doc.metadata["chunk_id"] = chunk_id
                doc.metadata["page_index"] = p_idx
                doc.metadata["chunk_index"] = c_idx
                table_accumulator.append(
                    {
                        "chunk_id": chunk_id,
                        "page_index": p_idx,
                        "chunk_index": c_idx,
                        "chunk_text": doc.page_content,
                        "start_offset": doc.metadata["start_index"],
                        "end_offset": doc.metadata["start_index"]
                        + len(doc.page_content),
                        "metadata": doc.metadata,
                    }
                )
        self.chunks_table = table_accumulator

    def _get_text_splitter(self, add_start_index=False):
        if self.split_strategy == "Recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap_size,
                add_start_index=add_start_index,
            )
        elif self.split_strategy == "Markdown":
            return MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap_size,
                add_start_index=add_start_index,
            )
        else:
            return CharacterTextSplitter(
                separator="",
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap_size,
                add_start_index=add_start_index,
            )

    def _make_chunk_id(self, metadata, page_index, chunk_index, start_offset, chunk_text):
        source_key = (
            metadata.get("source_id")
            or metadata.get("source")
            or metadata.get("title")
            or f"page-{page_index}"
        )
        digest = hashlib.sha1(
            f"{source_key}|{page_index}|{chunk_index}|{start_offset}|{len(chunk_text)}".encode("utf-8")
        ).hexdigest()[:12]
        return f"chunk::{digest}"

    def _build_highlighted_html(self, page_text, chunks_info):
        if not page_text:
            return page_text

        coverage = [[] for _ in range(len(page_text))]
        for info in chunks_info:
            start = info.get("start_offset", 0)
            end = info.get("end_offset", 0)
            c_idx = info["chunk_index"]
            for i in range(start, min(end, len(page_text))):
                coverage[i].append(c_idx)

        # New palette: brighter, lighter fills so they pop on dark background
        # but still look fine in light mode.
        colors = [
            "rgba(0, 237, 100, 0.40)",   # bright green
            "rgba(0, 214, 255, 0.40)",   # cyan
            "rgba(255, 214, 10, 0.40)",  # yellow
            "rgba(255, 159, 243, 0.40)", # pink
            "rgba(156, 220, 254, 0.40)", # light blue
        ]
        # Overlaps: slightly stronger accent
        overlap_color = "rgba(255, 99, 132, 0.55)"

        def html_escape(txt):
            return txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        def style_for(indices):
            if len(indices) > 1:
                return f'style="background-color:{overlap_color}"'
            elif len(indices) == 1:
                idx_mod = indices[0] % len(colors)
                return f'style="background-color:{colors[idx_mod]}"'
            else:
                return ""

        output = []
        if len(page_text) == 0:
            return ""

        span_buffer = page_text[0]
        last_cov = coverage[0]

        for i in range(1, len(page_text)):
            current_cov = coverage[i]
            if len(current_cov) != len(last_cov) or set(current_cov) != set(last_cov):
                output.append(f'<span {style_for(last_cov)}>{html_escape(span_buffer)}</span>')
                span_buffer = page_text[i]
                last_cov = current_cov
            else:
                span_buffer += page_text[i]

        if span_buffer:
            output.append(f'<span {style_for(last_cov)}>{html_escape(span_buffer)}</span>')

        return "".join(output)

    def _update_highlighted_preview(self):
        if 0 <= self.current_doc_index < len(self.loaded_pages):
            page_text = self.loaded_pages[self.current_doc_index]
            page_chunks_info = [
                row
                for row in self.chunks_table
                if row["page_index"] == self.current_doc_index
            ]
            self.document_preview = self._build_highlighted_html(
                page_text, page_chunks_info
            )
        else:
            self.document_preview = (
                f"No page found at index {self.current_doc_index}"
            )

    @traitlets.observe("current_doc_index")
    def _on_page_change(self, change):
        self._update_highlighted_preview()

    @traitlets.observe("constraints")
    def _on_constraints_change(self, change):
        self._constraints = self._normalize_constraints(change["new"])
        if self.constraints != self._constraints:
            self.constraints = dict(self._constraints)
            return
        self._apply_constraints()
        self.load_scoreboard()

    @traitlets.observe("benchmark_score_weights")
    def _on_benchmark_score_weights_change(self, change):
        self._benchmark_score_weights = self._normalize_benchmark_score_weights(change["new"])
        if self.benchmark_score_weights != self._benchmark_score_weights:
            self.benchmark_score_weights = dict(self._benchmark_score_weights)
            return
        self.load_scoreboard()

    @traitlets.observe("reranking_enabled", "top_k", "rerank_top_k")
    def _on_constrainted_knobs_change(self, change):
        self._apply_constraints()

    @traitlets.observe("chunk_size", "split_strategy", "overlap_size")
    def _on_chunk_settings_change(self, change):
        min_chunk_size, max_chunk_size = self._constraint_chunk_size_bounds()
        if self.chunk_size < min_chunk_size:
            self.chunk_size = min_chunk_size
            return
        if self.chunk_size > max_chunk_size:
            self.chunk_size = max_chunk_size
            return
        self._create_chunks_for_all_pages()
        self._update_highlighted_preview()
        self._build_embeddings_table()

    @traitlets.observe("selected_embedding_index")
    def _on_embedding_model_change(self, change):
        """Update the active embedding model when user selects a different one"""
        new_index = change["new"]
        if 0 <= new_index < len(self._embedding_models):
            self._embedding_model = self._embedding_models[new_index]
            self.embedding_model = self.embedding_models[new_index]
            # Reset the loaded state since we're using a different model
            self.loaded_in_mongo = False
            self.mongo_docs_table = []
            self._reset_indexing_state()

    @traitlets.observe("selected_reranker_index")
    def _on_reranker_change(self, change):
        new_index = change["new"]
        if 0 <= new_index < len(self._rerankers):
            self._reranker = self._rerankers[new_index]

    # -----------------
    # Step 2: EMBEDDING
    # -----------------
    def _load_vector_indexes(self):
        if self.mongo_collection is None:
            self.vector_indexes = []
            return
        try:
            all_indexes = list(self.mongo_collection.list_search_indexes())
            self.vector_indexes = [idx.get("name", "") for idx in all_indexes]
            if not self.vector_indexes:
                self.vector_indexes = []
        except Exception as e:
            if "command not found" in str(e).lower():
                idx_info = self.mongo_collection.index_information()
                self.vector_indexes = list(idx_info.keys())
            else:
                self.set_error(f"Error fetching search indexes: {e}")
                self.vector_indexes = []

    @traitlets.observe("vector_index_dimensions")
    def _on_vector_index_dimensions_change(self, change):
        new_dimensions = change["new"]
        if new_dimensions not in ALLOWED_VECTOR_INDEX_DIMENSIONS:
            self.vector_index_dimensions = DEFAULT_VECTOR_INDEX_DIMENSIONS
            return
        self.loaded_in_mongo = False
        self.mongo_docs_table = []
        self._reset_indexing_state()

    @traitlets.observe("embedding_output_dtype")
    def _on_embedding_output_dtype_change(self, change):
        new_output_dtype = change["new"]
        if new_output_dtype not in ALLOWED_EMBEDDING_OUTPUT_DTYPES:
            self.embedding_output_dtype = DEFAULT_EMBEDDING_OUTPUT_DTYPE
            return
        self.loaded_in_mongo = False
        self.mongo_docs_table = []
        self._reset_indexing_state()

    def _build_vector_store(
        self,
        embedding_model,
        *,
        auto_create_index=False,
        dimensions=None,
        auto_index_timeout=AUTO_INDEX_TIMEOUT_SECONDS,
    ):
        store_kwargs = {
            "collection": self.mongo_collection,
            "embedding": embedding_model,
            "index_name": self.selected_index,
            "relevance_score_fn": "cosine",
        }
        if auto_create_index:
            store_kwargs["auto_create_index"] = True
            store_kwargs["dimensions"] = (
                int(dimensions)
                if dimensions is not None
                else int(self.vector_index_dimensions or DEFAULT_VECTOR_INDEX_DIMENSIONS)
            )
            store_kwargs["auto_index_timeout"] = int(auto_index_timeout)
        return MongoDBAtlasVectorSearch(**store_kwargs)

    def _get_target_vector_dimensions(self):
        dimensions = int(self.vector_index_dimensions or DEFAULT_VECTOR_INDEX_DIMENSIONS)
        if dimensions not in ALLOWED_VECTOR_INDEX_DIMENSIONS:
            return DEFAULT_VECTOR_INDEX_DIMENSIONS
        return dimensions

    def _get_dimension_aware_embedding_model(self, embedding_model):
        if embedding_model is None:
            return None
        return DimensionAwareEmbeddings(
            embedding_model,
            self._get_target_vector_dimensions(),
            self.embedding_output_dtype,
        )

    def _ensure_vector_search_index(self):
        if self.mongo_collection is None:
            self.set_error("No MongoDB collection provided.")
            return None
        if self._embedding_model is None:
            self.set_error("No embedding model provided.")
            return None
        if not self.selected_index:
            self.set_error("No vector index was specified.")
            return None

        dimensions = self._get_target_vector_dimensions()
        if dimensions not in ALLOWED_VECTOR_INDEX_DIMENSIONS:
            self.set_error(
                "Invalid vector index dimension. Choose one of 256, 512, 1024, or 2048."
            )
            return None

        try:
            vector_store = self._build_vector_store(
                self._get_dimension_aware_embedding_model(self._embedding_model),
                auto_create_index=True,
                dimensions=dimensions,
                auto_index_timeout=AUTO_INDEX_TIMEOUT_SECONDS,
            )
            existing_indexes = {
                idx.get("name", "")
                for idx in self.mongo_collection.list_search_indexes()
            }
            if self.selected_index in existing_indexes:
                vector_store.create_vector_search_index(
                    dimensions=dimensions,
                    update=True,
                    wait_until_complete=AUTO_INDEX_TIMEOUT_SECONDS,
                )
            self._load_vector_indexes()
            return vector_store
        except Exception as e:
            self.set_error(f"Vector index creation/update error: {e}")
            return None

    def _wait_for_search_index_ready(
        self, index_name, timeout_seconds=AUTO_INDEX_TIMEOUT_SECONDS, collection=None
    ):
        collection = collection if collection is not None else self.mongo_collection
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                index_docs = list(collection.list_search_indexes(index_name))
            except TypeError:
                index_docs = [
                    idx for idx in collection.list_search_indexes()
                    if idx.get("name") == index_name
                ]
            except Exception:
                index_docs = []

            if index_docs:
                index_doc = index_docs[0] or {}
                status = str(index_doc.get("status", "")).upper()
                queryable = index_doc.get("queryable")
                if queryable is True or status in {"READY", "ACTIVE"}:
                    return

            time.sleep(2)

        raise TimeoutError(
            f"Search index '{index_name}' was not ready within {timeout_seconds} seconds."
        )

    def _ensure_hybrid_search_index(self, collection=None):
        collection = collection if collection is not None else self.mongo_collection
        if collection is None:
            self.set_error("No MongoDB collection provided.")
            return False

        index_definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "text": {"type": "string"},
                },
            },
        }

        try:
            existing_indexes = {
                idx.get("name", "")
                for idx in collection.list_search_indexes()
            }
            if self._text_search_index in existing_indexes:
                collection.update_search_index(
                    self._text_search_index,
                    index_definition,
                )
            else:
                collection.create_search_index(
                    SearchIndexModel(
                        definition=index_definition,
                        name=self._text_search_index,
                    )
                )
            self._wait_for_search_index_ready(
                self._text_search_index,
                timeout_seconds=AUTO_INDEX_TIMEOUT_SECONDS,
                collection=collection,
            )
            return True
        except Exception as e:
            self.set_error(f"Hybrid search index creation/update error: {e}")
            return False

    def _hybrid_search_index_exists(self, collection=None):
        collection = collection if collection is not None else self.mongo_collection
        try:
            return any(
                idx.get("name") == self._text_search_index
                for idx in collection.list_search_indexes()
            )
        except Exception:
            return False

    def _ensure_hybrid_search_index_available(self, collection=None):
        """Create the full-text index on demand if it is not there yet."""
        collection = collection if collection is not None else self.mongo_collection
        if collection is None:
            self.set_error("No MongoDB collection provided.")
            return False
        if self._hybrid_search_index_exists(collection):
            return True
        return self._ensure_hybrid_search_index(collection)

    def _build_embeddings_table(self):
        data = []
        for row in self.chunks_table:
            data.append(
                {
                    "chunk_id": row["chunk_id"],
                    "chunk_text": row["chunk_text"],
                    "page_index": row["page_index"],
                    "chunk_index": row["chunk_index"],
                    "start_offset": row.get("start_offset", 0),
                    "end_offset": row.get("end_offset", 0),
                    "metadata": row.get("metadata", {}),
                }
            )
        self.embeddings_table = data
        self.embedding_ready = False
        self.loaded_in_mongo = False

    def _sync_mongo_namespace_from_collection(self):
        if self.mongo_collection is None:
            self._mongo_database_name = ""
            self._mongo_collection_name = ""
            self.mongo_db_name = ""
            self.mongo_collection_name = ""
            return

        self._mongo_database_name = self.mongo_collection.database.name
        self._mongo_collection_name = self.mongo_collection.name
        self.mongo_db_name = self._mongo_database_name
        self.mongo_collection_name = self._mongo_collection_name
        if self._mongo_client is None:
            self._mongo_client = self.mongo_collection.database.client

    def _set_api_key_on_target(self, target, api_key, visited=None):
        if target is None:
            return False
        if visited is None:
            visited = set()
        target_id = id(target)
        if target_id in visited:
            return False
        visited.add(target_id)

        updated = False
        for attr_name in ("api_key", "openai_api_key", "voyage_api_key", "voyageai_api_key"):
            if hasattr(target, attr_name):
                try:
                    setattr(target, attr_name, api_key)
                    updated = True
                except Exception:
                    pass

        for child_attr in ("client", "_client", "root_client", "async_client", "_async_client"):
            child = getattr(target, child_attr, None)
            if child is not None:
                updated = self._set_api_key_on_target(child, api_key, visited) or updated

        return updated

    def _apply_model_api_key_override(self, api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["VOYAGE_API_KEY"] = api_key
        os.environ["MONGODB_ATLAS_LOCAL_VOYAGE_API_KEY"] = api_key

        updated_targets = 0
        for model in list(self._embedding_models) + list(self._rerankers):
            if self._set_api_key_on_target(model, api_key):
                updated_targets += 1
        return updated_targets

    def _apply_mongodb_uri_override(self, mongodb_uri):
        if not self._mongo_database_name or not self._mongo_collection_name:
            raise ValueError(
                "Cannot override the MongoDB URI because the current database and collection names are unknown."
            )

        mongo_client = MongoClient(mongodb_uri)
        mongo_client.admin.command("ping")
        self._mongo_client = mongo_client
        self.client = mongo_client
        self.mongo_collection = mongo_client[self._mongo_database_name][self._mongo_collection_name]
        self._sync_mongo_namespace_from_collection()
        self.mongo_docs_table = []
        self.rag_results = []
        self.rerank_results = []
        self.query_embedding = []
        self.all_doc_embeddings = []
        self.loaded_in_mongo = False
        self._load_vector_indexes()

    def apply_overrides(self):
        self.clear_error()
        messages = []

        api_key = (self.override_model_api_key or "").strip()
        mongodb_uri = (self.override_mongodb_uri or "").strip()

        if api_key:
            updated_targets = self._apply_model_api_key_override(api_key)
            if updated_targets:
                messages.append(f"Updated model credentials on {updated_targets} configured model objects.")
            else:
                messages.append("Stored the model API key override in environment variables for this widget session.")

        if mongodb_uri:
            self._apply_mongodb_uri_override(mongodb_uri)
            messages.append(
                f"Connected to {self.mongo_db_name}.{self.mongo_collection_name} with the override MongoDB URI."
            )

        if not messages:
            messages.append("No overrides were applied because both fields are empty.")

        self.set_overrides_status(" ".join(messages))
        self.mongo_docs_table = []
        self._reset_indexing_state()

    def _reset_indexing_state(self):
        self.indexing_status = {
            "phase": "idle",
            "message": "",
            "purged": False,
            "chunks_indexed": 0,
            "documents_stored": 0,
            "total_chunks": 0,
            "total_documents": 0,
            "progress": 0.0,
        }

    def _update_indexing_status(
        self,
        *,
        phase,
        message="",
        purged=False,
        chunks_indexed=0,
        documents_stored=0,
        total_chunks=0,
        total_documents=0,
    ):
        total = max(int(total_chunks or 0), int(total_documents or 0), 1)
        completed = max(int(chunks_indexed or 0), int(documents_stored or 0))
        self.indexing_status = {
            "phase": phase,
            "message": message,
            "purged": bool(purged),
            "chunks_indexed": int(chunks_indexed or 0),
            "documents_stored": int(documents_stored or 0),
            "total_chunks": int(total_chunks or 0),
            "total_documents": int(total_documents or 0),
            "progress": max(0.0, min(1.0, completed / total)),
        }

    def _build_mongo_doc_preview(self, mongo_doc):
        text_value = str(mongo_doc.get("text", "") or "")
        embedding = mongo_doc.get("embedding", []) or []
        if not isinstance(embedding, list):
            embedding = list(embedding)
        embedding_preview = embedding[:5]
        if len(embedding) > 5:
            embedding_preview = embedding_preview + ["..."]
        extra_fields = {}
        for key, value in mongo_doc.items():
            if key in {"_id", "text", "embedding"}:
                continue
            if isinstance(value, (dict, list)):
                try:
                    extra_fields[key] = json.dumps(value, ensure_ascii=True)
                except Exception:
                    extra_fields[key] = str(value)
            else:
                extra_fields[key] = "" if value is None else str(value)
        return {
            "_id": str(mongo_doc.get("_id", "")),
            "text": text_value[:60] + "..." if len(text_value) > 60 else text_value,
            "embedding": embedding_preview,
            "extra_fields": extra_fields,
        }

    def _reset_benchmark_status(self):
        self.benchmark_status = {
            "phase": "idle",
            "message": "",
            "completed": 0,
            "total": 0,
            "progress": 0.0,
            "primary_label": "tasks",
            "secondary_label": "",
        }

    def _reset_configuration_status(self):
        self.configuration_status = {
            "phase": "pending",
            "message": "Validating configuration...",
            "completed": 0,
            "total": 2,
            "progress": 0.0,
            "primary_label": "checks",
            "secondary_label": "",
        }

    def _reset_retrieval_status(self):
        self.retrieval_status = {
            "phase": "idle",
            "message": "",
            "completed": 0,
            "total": 0,
            "progress": 0.0,
            "primary_label": "stages",
            "secondary_label": "",
        }

    def _update_retrieval_status(
        self,
        *,
        phase,
        message="",
        completed=0,
        total=3,
        secondary_label="",
    ):
        total_value = max(int(total or 0), 1)
        completed_value = max(0, min(int(completed or 0), total_value))
        self.retrieval_status = {
            "phase": phase,
            "message": message,
            "completed": completed_value,
            "total": int(total or 0),
            "progress": max(0.0, min(1.0, completed_value / total_value)),
            "primary_label": "stages",
            "secondary_label": secondary_label,
        }

    def _update_configuration_status(
        self,
        *,
        phase,
        message="",
        completed=0,
        total=2,
        secondary_label="",
    ):
        total_value = max(int(total or 0), 1)
        completed_value = max(0, min(int(completed or 0), total_value))
        self.configuration_status = {
            "phase": phase,
            "message": message,
            "completed": completed_value,
            "total": int(total or 0),
            "progress": max(0.0, min(1.0, completed_value / total_value)),
            "primary_label": "checks",
            "secondary_label": secondary_label,
        }

    def _run_with_timeout(self, func, timeout_seconds, timeout_message):
        result_queue = queue.Queue(maxsize=1)

        def target():
            try:
                result_queue.put(("result", func()))
            except Exception as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            state, payload = result_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            raise TimeoutError(timeout_message)
        if state == "error":
            raise payload
        return payload

    def _check_mongodb_connection(self):
        mongo_client = None
        if self.mongo_collection is not None:
            mongo_client = self.mongo_collection.database.client
        elif self.client is not None and hasattr(self.client, "admin"):
            mongo_client = self.client

        if mongo_client is None:
            raise ValueError("No MongoDB collection or client was provided.")

        mongo_client.admin.command("ping")
        return True

    def _check_embedding_api(self):
        if self._embedding_model is None:
            raise ValueError("No embedding model was provided.")

        probe_embedding_model = self._get_dimension_aware_embedding_model(self._embedding_model)
        vector = probe_embedding_model.embed_query("configuration test")
        if vector is None or not hasattr(vector, "__len__") or len(vector) == 0:
            raise ValueError("Embedding API returned an empty vector.")
        return True

    def _run_configuration_preflight(self):
        try:
            self._update_configuration_status(
                phase="running",
                message="Checking MongoDB network access and credentials...",
                completed=0,
                total=2,
                secondary_label="MongoDB connectivity",
            )
            self._run_with_timeout(
                self._check_mongodb_connection,
                CONFIG_CHECK_MONGODB_TIMEOUT_SECONDS,
                "Timed out while trying to reach MongoDB. Check network access, IP allowlist, and the URI.",
            )

            self._update_configuration_status(
                phase="running",
                message="Checking the embedding API with a minimal test request...",
                completed=1,
                total=2,
                secondary_label="Embedding API validation",
            )
            self._run_with_timeout(
                self._check_embedding_api,
                CONFIG_CHECK_EMBEDDING_TIMEOUT_SECONDS,
                "Timed out while trying to reach the embedding API. Check the API key, endpoint, and network access.",
            )

            self._load_vector_indexes()
            self.load_scoreboard()
            self._update_configuration_status(
                phase="completed",
                message="Configuration checks passed. MongoDB and the embedding API are ready.",
                completed=2,
                total=2,
                secondary_label="Ready",
            )
        except Exception as e:
            self._update_configuration_status(
                phase="error",
                message=str(e),
                completed=0,
                total=2,
                secondary_label="Configuration failed",
            )
            self.set_error(f"Configuration Error: {e}")

    def _start_configuration_preflight(self):
        thread = threading.Thread(target=self._run_configuration_preflight, daemon=True)
        thread.start()

    def _update_benchmark_status(
        self,
        *,
        phase,
        message="",
        completed=0,
        total=0,
        primary_label="tasks",
        secondary_label="",
    ):
        total_value = max(int(total or 0), 1)
        completed_value = max(0, min(int(completed or 0), total_value))
        self.benchmark_status = {
            "phase": phase,
            "message": message,
            "completed": completed_value,
            "total": int(total or 0),
            "progress": max(0.0, min(1.0, completed_value / total_value)),
            "primary_label": primary_label,
            "secondary_label": secondary_label,
        }

    @traitlets.observe("command")
    def _on_command(self, change):
        cmd = change["new"]
        if cmd == "load_into_mongo":
            self.load_into_mongo()
        elif cmd == "vector_search":
            self.run_vector_search()
        elif cmd == "hybrid_search":
            self.run_hybrid_search()
        elif cmd == "rerank_results":
            self.run_reranking()
        elif cmd == "run_benchmark":
            self.run_benchmark()
        elif cmd == "run_configuration_check":
            self._run_configuration_preflight()
        elif cmd == "load_scoreboard":
            self.load_scoreboard()
        elif cmd == "publish_scoreboard":
            self.publish_scoreboard()
        elif cmd == "explain_benchmark_task":
            self.explain_benchmark_task()
        elif cmd == "apply_overrides":
            try:
                self.apply_overrides()
            except Exception as e:
                self.set_overrides_status("")
                self.set_error(f"Override Error: {e}")
        self.command = ""

    def load_into_mongo(self):
        self.clear_error()
        self.mongo_docs_table = []
        self.loaded_in_mongo = False
        self.embedding_ready = False

        total_chunks = len(self.chunks_table or [])
        self._update_indexing_status(
            phase="preparing",
            message="Creating or updating the MongoDB vector index (this can take a moment)...",
            total_chunks=total_chunks,
            total_documents=total_chunks,
        )

        vector_store = self._ensure_vector_search_index()
        if vector_store is None:
            self._update_indexing_status(
                phase="error",
                message="Could not prepare the vector index.",
                total_chunks=total_chunks,
                total_documents=total_chunks,
            )
            return
        # Always create the full-text index: the user may switch to the hybrid
        # tab after loading, and the index would otherwise never exist.
        if not self._ensure_hybrid_search_index():
            self._update_indexing_status(
                phase="error",
                message="Could not prepare the hybrid search index.",
                total_chunks=total_chunks,
                total_documents=total_chunks,
            )
            return

        try:
            self._update_indexing_status(
                phase="purging",
                message="Purging existing MongoDB documents...",
                total_chunks=total_chunks,
                total_documents=total_chunks,
            )
            self.mongo_collection.delete_many({})
            self._update_indexing_status(
                phase="purged",
                message="Collection purged. Starting indexing...",
                purged=True,
                total_chunks=total_chunks,
                total_documents=total_chunks,
            )

            texts = [r["chunk_text"] for r in self.chunks_table]
            metadatas = [
                row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
                for row in self.chunks_table
            ]

            if not texts:
                self.embedding_ready = True
                self.loaded_in_mongo = True
                self._update_indexing_status(
                    phase="completed",
                    message="No chunks to index.",
                    purged=True,
                    total_chunks=0,
                    total_documents=0,
                )
                return

            embedding_model = self._get_dimension_aware_embedding_model(self._embedding_model)
            batch_size = 20
            preview_rows = []
            stored_documents = 0
            indexed_chunks = 0

            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_metadatas = metadatas[start : start + batch_size]
                batch_embeddings = embedding_model.embed_documents(batch_texts)

                mongo_batch = []
                for text, metadata, embedding in zip(batch_texts, batch_metadatas, batch_embeddings):
                    mongo_batch.append(
                        {
                            "text": text,
                            "embedding": [float(value) for value in (embedding or [])],
                            **dict(metadata or {}),
                        }
                    )

                insert_result = self.mongo_collection.insert_many(mongo_batch)
                for inserted_id, mongo_doc in zip(insert_result.inserted_ids, mongo_batch):
                    preview_rows.append(
                        self._build_mongo_doc_preview({**mongo_doc, "_id": inserted_id})
                    )

                stored_documents += len(mongo_batch)
                indexed_chunks += len(batch_texts)
                self.mongo_docs_table = preview_rows
                self._update_indexing_status(
                    phase="indexing",
                    message=(
                        f"Indexing {indexed_chunks} / {total_chunks} chunks and "
                        f"storing {stored_documents} / {total_chunks} MongoDB documents..."
                    ),
                    purged=True,
                    chunks_indexed=indexed_chunks,
                    documents_stored=stored_documents,
                    total_chunks=total_chunks,
                    total_documents=total_chunks,
                )

            self.embedding_ready = True
            self.loaded_in_mongo = True
            self._update_indexing_status(
                phase="completed",
                message=f"Indexed {total_chunks} chunks and stored {stored_documents} MongoDB documents.",
                purged=True,
                chunks_indexed=indexed_chunks,
                documents_stored=stored_documents,
                total_chunks=total_chunks,
                total_documents=total_chunks,
            )
        except Exception as e:
            self.mongo_docs_table = []
            self._update_indexing_status(
                phase="error",
                message="Indexing failed before documents could be stored.",
                total_chunks=total_chunks,
                total_documents=total_chunks,
            )
            self.set_error(f"MongoDB Load Error: {e}")

    def _get_query_embedding_model(self):
        query_emb_index = self.query_embedding_index
        if 0 <= query_emb_index < len(self._embedding_models):
            query_embedding_model = self._embedding_models[query_emb_index]
            query_model_name = (
                self.embedding_models[query_emb_index]
                if query_emb_index < len(self.embedding_models)
                else "Unknown"
            )
        else:
            query_embedding_model = self._embedding_model
            query_model_name = self.embedding_model
        return query_embedding_model, query_model_name, query_emb_index

    def _run_vector_query(self):
        # Clear old state for this single run
        self.rag_results = []
        self.query_embedding = []
        self.all_doc_embeddings = []
        self.rerank_results = []
        self._reset_retrieval_status()
        self.clear_error()

        # Basic validation
        if not self.selected_index:
            self.set_error("No vector index was specified.")
            return
        if self._embedding_model is None:
            self.set_error("No embedding model provided.")
            return

        query = (self.rag_query or "").strip()
        if not query:
            self.set_error("No query provided.")
            return

        try:
            self._update_retrieval_status(
                phase="running",
                message="Generating query embedding...",
                completed=0,
                total=3,
                secondary_label="embedding query",
            )
            query_embedding_model, _, _ = self._get_query_embedding_model()
            query_embedding_model = self._get_dimension_aware_embedding_model(query_embedding_model)

            # Build vector store with the query embedding model
            vector_store = self._build_vector_store(query_embedding_model)

            # Generate query embedding using the selected query embedding model
            query_emb = query_embedding_model.embed_query(query)
            self.query_embedding = [float(x) for x in query_emb]
            self._update_retrieval_status(
                phase="running",
                message="Searching MongoDB for the most relevant chunks...",
                completed=1,
                total=3,
                secondary_label="vector retrieval",
            )

            # k (fallback to 5 if bad)
            k = self.top_k if isinstance(self.top_k, int) and self.top_k > 0 else 5
            # Optional threshold
            threshold_val = None
            if self.score_threshold not in (None, ""):
                try:
                    threshold_val = float(self.score_threshold)
                except (ValueError, TypeError):
                    threshold_val = None

            # One retrieval - get more than k to allow threshold filtering
            # Get a larger set first, then filter by threshold
            retrieval_k = max(k * 3, 20)  # Get 3x k or at least 20 documents
            large_search_args = {"k": retrieval_k}

            all_scored_docs = vector_store.similarity_search_with_score(
                query, **large_search_args
            )

            # Filter by threshold if provided (this gives us ALL docs above threshold)
            if threshold_val is not None:
                docs_above_threshold = [
                    (doc, score)
                    for (doc, score) in all_scored_docs
                    if score >= threshold_val
                ]
            else:
                docs_above_threshold = all_scored_docs

            # Now take top K from the threshold-filtered docs for RAG context
            docs_with_scores = docs_above_threshold[:k]

            # Create a mapping from text to embedding for all docs above threshold
            text_to_embedding = {}
            all_embeddings_for_viz = []

            # Process ALL documents above threshold for visualization
            for idx, (doc, score) in enumerate(docs_above_threshold):
                text = doc.page_content
                chunk_id = doc.metadata.get("chunk_id")
                # Fetch the embedding from MongoDB
                mongo_doc = self.mongo_collection.find_one(
                    {"chunk_id": chunk_id} if chunk_id else {"text": text},
                    {"_id": 1, "text": 1, "embedding": 1}
                )
                if mongo_doc:
                    embedding = mongo_doc.get("embedding", [])
                    if embedding:
                        text_to_embedding[chunk_id or text] = embedding
                        display_text = text[:50] + "..." if len(text) > 50 else text
                        all_embeddings_for_viz.append({
                            "id": chunk_id or str(mongo_doc.get("_id", idx)),
                            "text": display_text,
                            "embedding": embedding,
                            "score": float(score)
                        })

            self.all_doc_embeddings = all_embeddings_for_viz

            # Update UI results with embeddings for retrieved docs
            results_with_embeddings = []
            for rank, (doc, score) in enumerate(docs_with_scores, start=1):
                # Try to find the embedding by matching the text content
                chunk_id = doc.metadata.get("chunk_id")
                embedding = text_to_embedding.get(chunk_id or doc.page_content, [])

                # Format embedding with sample (first 5 values) like in Indexing section
                embedding_sample = (
                    embedding[:5] + ["..."]
                    if len(embedding) > 5
                    else embedding
                )

                result = {
                    "result_id": chunk_id or f"search-{rank}",
                    "rank": rank,
                    "score": float(score),
                    "text": doc.page_content,  # Changed from "content" to "text"
                    "metadata": doc.metadata,
                    "embedding": embedding_sample  # Use sampled version
                }
                results_with_embeddings.append(result)

            self.rag_results = results_with_embeddings
            self._update_retrieval_status(
                phase="completed",
                message=f"Search complete. Retrieved {len(results_with_embeddings)} results.",
                completed=3,
                total=3,
                secondary_label="results ready",
            )

        except Exception as e:
            self._update_retrieval_status(
                phase="error",
                message="Search failed before results were ready.",
                completed=0,
                total=3,
                secondary_label="search failed",
            )
            self.set_error(f"Vector Search Error: {e}")

    def _run_hybrid_query(self):
        self.rag_results = []
        self.query_embedding = []
        self.all_doc_embeddings = []
        self.rerank_results = []
        self._reset_retrieval_status()
        self.clear_error()

        if not self.selected_index:
            self.set_error("No vector index was specified.")
            return
        if self._embedding_model is None:
            self.set_error("No embedding model provided.")
            return

        query = (self.rag_query or "").strip()
        if not query:
            self.set_error("No query provided.")
            return

        try:
            self._update_retrieval_status(
                phase="running",
                message="Preparing the full-text search index...",
                completed=0,
                total=3,
                secondary_label="text index",
            )
            if not self._ensure_hybrid_search_index_available():
                self._update_retrieval_status(
                    phase="error",
                    message="Could not prepare the full-text search index used by hybrid search.",
                    completed=0,
                    total=3,
                    secondary_label="search failed",
                )
                return

            self._update_retrieval_status(
                phase="running",
                message="Generating query embedding...",
                completed=0,
                total=3,
                secondary_label="embedding query",
            )
            query_embedding_model, _, _ = self._get_query_embedding_model()
            query_embedding_model = self._get_dimension_aware_embedding_model(query_embedding_model)

            vector_store = self._build_vector_store(query_embedding_model)

            query_emb = query_embedding_model.embed_query(query)
            self.query_embedding = [float(x) for x in query_emb]
            self._update_retrieval_status(
                phase="running",
                message="Running hybrid search across vector and text indexes...",
                completed=1,
                total=3,
                secondary_label="hybrid retrieval",
            )

            k = self.top_k if isinstance(self.top_k, int) and self.top_k > 0 else 5
            fulltext_penalty = int(self.fulltext_penalty) if isinstance(self.fulltext_penalty, int) else 50
            vector_penalty = int(self.vector_penalty) if isinstance(self.vector_penalty, int) else 50

            retriever = MongoDBAtlasHybridSearchRetriever(
                vectorstore=vector_store,
                search_index_name=self._text_search_index,
                fulltext_penalty=fulltext_penalty,
                vector_penalty=vector_penalty,
                top_k=k,
            )

            hybrid_docs = retriever.invoke(query)

            results_with_scores = []
            all_embeddings_for_viz = []
            for idx, doc in enumerate(hybrid_docs):
                fulltext_score = float(doc.metadata.get("fulltext_score", 0.0))
                vector_score = float(doc.metadata.get("vector_score", 0.0))
                total_score = fulltext_score + vector_score

                text = doc.page_content
                chunk_id = doc.metadata.get("chunk_id")
                mongo_doc = self.mongo_collection.find_one(
                    {"chunk_id": chunk_id} if chunk_id else {"text": text},
                    {"_id": 1, "text": 1, "embedding": 1, "source": 1}
                )

                embedding = mongo_doc.get("embedding", []) if mongo_doc else []
                if embedding:
                    display_text = text[:50] + "..." if len(text) > 50 else text
                    all_embeddings_for_viz.append({
                        "id": chunk_id or str(mongo_doc.get("_id", idx)),
                        "text": display_text,
                        "embedding": embedding,
                        "score": total_score,
                    })

                embedding_sample = (
                    embedding[:5] + ["..."]
                    if len(embedding) > 5
                    else embedding
                )

                results_with_scores.append({
                    "result_id": chunk_id or f"search-{idx + 1}",
                    "rank": idx + 1,
                    "score": total_score,
                    "vector_score": vector_score,
                    "search_score": fulltext_score,
                    "total_score": total_score,
                    "text": text,
                    "metadata": doc.metadata,
                    "embedding": embedding_sample,
                })

            self.all_doc_embeddings = all_embeddings_for_viz
            self.rag_results = results_with_scores
            self._update_retrieval_status(
                phase="completed",
                message=f"Hybrid search complete. Retrieved {len(results_with_scores)} results.",
                completed=3,
                total=3,
                secondary_label="results ready",
            )

        except Exception as e:
            self._update_retrieval_status(
                phase="error",
                message="Hybrid search failed before results were ready.",
                completed=0,
                total=3,
                secondary_label="search failed",
            )
            self.set_error(f"Hybrid Search Error: {e}")

    def _run_reranking(self):
        self.rerank_results = []
        self.clear_error()

        if not self.reranking_enabled:
            self.set_error("Enable reranking to run this step.")
            return
        if not self.rag_results:
            self.set_error("Run retrieval first so there are chunks to rerank.")
            return
        if self._reranker is None:
            self.set_error("No reranker provided.")
            return

        query = (self.rag_query or "").strip()
        if not query:
            self.set_error("No query provided.")
            return

        rerank_k = self.rerank_top_k if isinstance(self.rerank_top_k, int) and self.rerank_top_k > 0 else 3

        try:
            candidate_docs = []
            for idx, row in enumerate(self.rag_results, start=1):
                metadata = dict(row.get("metadata", {}) or {})
                metadata["_rerank_id"] = row.get("result_id", f"search-{idx}")
                metadata["_source_rank"] = row.get("rank", idx)
                metadata["_source_score"] = row.get("score")
                if "vector_score" in row:
                    metadata["_vector_score"] = row.get("vector_score")
                if "search_score" in row:
                    metadata["_search_score"] = row.get("search_score")
                candidate_docs.append(
                    Document(
                        page_content=row.get("text", ""),
                        metadata=metadata,
                    )
                )

            if hasattr(self._reranker, "top_k"):
                try:
                    self._reranker.top_k = min(rerank_k, len(candidate_docs))
                except Exception:
                    pass

            compressed_docs = self._reranker.compress_documents(candidate_docs, query)

            reranked_rows = []
            for idx, doc in enumerate(compressed_docs[:rerank_k], start=1):
                metadata = dict(getattr(doc, "metadata", {}) or {})
                reranked_rows.append(
                    {
                        "result_id": metadata.get("_rerank_id", f"search-{idx}"),
                        "rank": idx,
                        "original_rank": metadata.get("_source_rank"),
                        "source_score": metadata.get("_source_score"),
                        "vector_score": metadata.get("_vector_score"),
                        "search_score": metadata.get("_search_score"),
                        "relevance_score": metadata.get("relevance_score"),
                        "text": getattr(doc, "page_content", ""),
                        "metadata": metadata,
                    }
                )

            self.rerank_results = reranked_rows

        except Exception as e:
            self.set_error(f"Reranking Error: {e}")

    def _load_benchmark_dataset(self):
        dataset = self._benchmark_dataset_raw
        if dataset is None:
            self.benchmark_dataset_name = ""
            self.benchmark_dataset_loaded = False
            return

        try:
            if isinstance(dataset, (str, pathlib.Path)):
                dataset_str = str(dataset).strip()
                normalized_url = None
                if dataset_str.startswith(("http://", "https://")):
                    normalized_url = dataset_str
                elif dataset_str.startswith("http:/") and not dataset_str.startswith("http://"):
                    normalized_url = dataset_str.replace("http:/", "http://", 1)
                elif dataset_str.startswith("https:/") and not dataset_str.startswith("https://"):
                    normalized_url = dataset_str.replace("https:/", "https://", 1)

                if normalized_url:
                    response = requests.get(normalized_url, timeout=30)
                    response.raise_for_status()
                    payload = json.loads(response.text)
                    parsed_url = urlparse(normalized_url)
                    self.benchmark_dataset_name = pathlib.Path(parsed_url.path).name or parsed_url.netloc
                else:
                    path = pathlib.Path(dataset_str).expanduser()
                    if not path.is_absolute():
                        path = pathlib.Path.cwd() / path
                    payload = json.loads(path.read_text())
                    self.benchmark_dataset_name = path.name
            elif isinstance(dataset, dict):
                payload = dataset
                self.benchmark_dataset_name = "benchmark_dataset"
            else:
                raise ValueError("benchmark_dataset must be a dict, a JSON file path, or an HTTP/HTTPS URL")

            sources = payload.get("sources", [])
            seeds = payload.get("seeds", [])
            examples = payload.get("examples", [])
            if not isinstance(sources, list) or not isinstance(seeds, list) or not isinstance(examples, list):
                raise ValueError("benchmark dataset must contain list fields: sources, seeds, examples")

            self._benchmark_dataset = payload
            self._benchmark_sources_by_id = {
                row.get("source_id"): row for row in sources if row.get("source_id")
            }
            self._benchmark_seeds_by_id = {
                row.get("seed_id"): row for row in seeds if row.get("seed_id")
            }
            self.benchmark_dataset_loaded = True
        except Exception as e:
            self.benchmark_dataset_name = ""
            self.benchmark_dataset_loaded = False
            self.set_error(f"Benchmark Dataset Error: {e}")

    def _build_chunk_rows_for_documents(self, documents):
        rows = []
        splitter = self._get_text_splitter(add_start_index=True)
        for doc_index, document in enumerate(documents):
            text = document.get("text", "") or ""
            metadata = {
                "source_id": document.get("source_id", f"source-{doc_index}"),
                "source": document.get("source_id", f"source-{doc_index}"),
                "title": document.get("title", f"Document {doc_index + 1}"),
            }
            doc_parts = splitter.create_documents([text], metadatas=[metadata])
            for chunk_index, doc_part in enumerate(doc_parts):
                chunk_id = self._make_chunk_id(
                    metadata=doc_part.metadata,
                    page_index=doc_index,
                    chunk_index=chunk_index,
                    start_offset=doc_part.metadata["start_index"],
                    chunk_text=doc_part.page_content,
                )
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "source_id": metadata["source_id"],
                        "chunk_index": chunk_index,
                        "text": doc_part.page_content,
                        "start_offset": doc_part.metadata["start_index"],
                        "end_offset": doc_part.metadata["start_index"] + len(doc_part.page_content),
                        "metadata": {
                            **doc_part.metadata,
                            "chunk_id": chunk_id,
                            "source_id": metadata["source_id"],
                            "chunk_index": chunk_index,
                        },
                    }
                )
        return rows

    def _build_benchmark_relevance(self, chunk_rows):
        relevant_by_seed = {}
        chunks_by_source = {}
        for row in chunk_rows:
            chunks_by_source.setdefault(row["source_id"], []).append(row)

        examples_by_seed_id = {}
        for example in self._benchmark_dataset.get("examples", []):
            seed_id = example.get("relevant_seed_id")
            if seed_id and seed_id not in examples_by_seed_id:
                examples_by_seed_id[seed_id] = example

        for seed_id, seed in self._benchmark_seeds_by_id.items():
            source_id = seed.get("source_id")
            source_row = self._benchmark_sources_by_id.get(source_id)
            source_text = (source_row or {}).get("text", "")
            seed_text = seed.get("text", "")
            example = examples_by_seed_id.get(seed_id, {})
            supporting_quote = (example.get("supporting_quote", "") or "").strip()
            answer = (example.get("answer", "") or "").strip()
            if not source_text or not seed_text:
                relevant_by_seed[seed_id] = set()
                continue

            start = source_text.find(seed_text)
            if start < 0:
                relevant_by_seed[seed_id] = set()
                continue
            end = start + len(seed_text)

            overlapping_rows = []
            for row in chunks_by_source.get(source_id, []):
                overlap_start = max(row["start_offset"], start)
                overlap_end = min(row["end_offset"], end)
                overlap_len = max(0, overlap_end - overlap_start)
                if overlap_len > 0:
                    overlapping_rows.append((overlap_len, row))

            if not overlapping_rows:
                relevant_by_seed[seed_id] = set()
                continue

            quote_matches = [
                row for _, row in overlapping_rows
                if supporting_quote and supporting_quote in row.get("text", "")
            ]
            if quote_matches:
                quote_matches.sort(key=lambda row: (row["start_offset"], row["chunk_index"]))
                relevant_by_seed[seed_id] = {quote_matches[0]["chunk_id"]}
                continue

            answer_matches = [
                row for _, row in overlapping_rows
                if answer and answer.lower() in row.get("text", "").lower()
            ]
            if answer_matches:
                answer_matches.sort(key=lambda row: (row["start_offset"], row["chunk_index"]))
                relevant_by_seed[seed_id] = {answer_matches[0]["chunk_id"]}
                continue

            # Fallback: pick one canonical gold chunk by highest overlap with the seed span.
            # Tie-break toward the earlier chunk in the document.
            overlapping_rows.sort(
                key=lambda item: (-item[0], item[1]["start_offset"], item[1]["chunk_index"])
            )
            best_row = overlapping_rows[0][1]
            relevant_by_seed[seed_id] = {best_row["chunk_id"]}

        return relevant_by_seed

    def _load_benchmark_documents_into_mongo(self, chunk_rows):
        if self.mongo_collection is None:
            self.set_error("No MongoDB collection provided for benchmark.")
            return None
        if not self.selected_index:
            self.set_error("No vector index was specified for benchmark.")
            return None
        if self._embedding_model is None:
            self.set_error("No embedding model provided for benchmark.")
            return None
        if not chunk_rows:
            self.set_error("Benchmark dataset produced no chunks to index.")
            return None

        benchmark_collection = self.mongo_collection
        benchmark_collection.delete_many({})
        vector_store = MongoDBAtlasVectorSearch(
            collection=benchmark_collection,
            embedding=self._get_dimension_aware_embedding_model(self._embedding_model),
            index_name=self.selected_index,
            relevance_score_fn="cosine",
        )
        vector_store.add_texts(
            [row["text"] for row in chunk_rows],
            metadatas=[row.get("metadata", {}) for row in chunk_rows],
        )
        return benchmark_collection

    def _wait_for_benchmark_documents_searchable(self, benchmark_collection, probe_query, expected_min_results=1, timeout_seconds=30):
        query_embedding_model, _, _ = self._get_query_embedding_model()
        vector_store = MongoDBAtlasVectorSearch(
            collection=benchmark_collection,
            embedding=self._get_dimension_aware_embedding_model(query_embedding_model),
            index_name=self.selected_index,
            relevance_score_fn="cosine",
        )

        deadline = time.time() + timeout_seconds
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                probe_results = vector_store.similarity_search_with_score(probe_query, k=max(expected_min_results, 1))
                if len(probe_results) >= expected_min_results:
                    return
            except Exception:
                pass
            time.sleep(2)

        raise TimeoutError(
            f"Benchmark documents were not searchable in MongoDB within {timeout_seconds} seconds."
        )

    def _benchmark_retrieve(self, query, benchmark_collection):
        query_embedding_model, _, _ = self._get_query_embedding_model()
        top_k = self.top_k if isinstance(self.top_k, int) and self.top_k > 0 else 5
        hybrid_enabled = int(self.search_sub_tab or 1) == 2
        vector_store = MongoDBAtlasVectorSearch(
            collection=benchmark_collection,
            embedding=self._get_dimension_aware_embedding_model(query_embedding_model),
            index_name=self.selected_index,
            relevance_score_fn="cosine",
        )

        if hybrid_enabled:
            if not self._ensure_hybrid_search_index_available(benchmark_collection):
                raise RuntimeError(
                    "Could not prepare the full-text search index "
                    f"'{self._text_search_index}' on the benchmark collection."
                )
            fulltext_penalty = int(self.fulltext_penalty) if isinstance(self.fulltext_penalty, int) else 50
            vector_penalty = int(self.vector_penalty) if isinstance(self.vector_penalty, int) else 50
            retriever = MongoDBAtlasHybridSearchRetriever(
                vectorstore=vector_store,
                search_index_name=self._text_search_index,
                fulltext_penalty=fulltext_penalty,
                vector_penalty=vector_penalty,
                top_k=top_k,
            )
            hybrid_docs = retriever.invoke(query)
            retrieval_rows = []
            for rank, doc in enumerate(hybrid_docs, start=1):
                vector_score = float(doc.metadata.get("vector_score", 0.0))
                fulltext_score = float(doc.metadata.get("fulltext_score", 0.0))
                retrieval_rows.append(
                    {
                        "result_id": doc.metadata.get("chunk_id", f"search-{rank}"),
                        "rank": rank,
                        "score": vector_score + fulltext_score,
                        "vector_score": vector_score,
                        "search_score": fulltext_score,
                        "total_score": vector_score + fulltext_score,
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )
            return retrieval_rows

        threshold_val = None
        if self.score_threshold not in (None, ""):
            try:
                threshold_val = float(self.score_threshold)
            except (ValueError, TypeError):
                threshold_val = None

        retrieval_k = max(top_k * 3, 20)
        all_scored_docs = vector_store.similarity_search_with_score(query, k=retrieval_k)
        if threshold_val is not None:
            all_scored_docs = [
                (doc, score) for (doc, score) in all_scored_docs if score >= threshold_val
            ]

        retrieval_rows = []
        for rank, (doc, score) in enumerate(all_scored_docs[:top_k], start=1):
            retrieval_rows.append(
                {
                    "result_id": doc.metadata.get("chunk_id", f"search-{rank}"),
                    "rank": rank,
                    "score": float(score),
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return retrieval_rows

    def _benchmark_rerank(self, query, retrieval_rows):
        if not self.reranking_enabled or self._reranker is None or not retrieval_rows:
            return []

        rerank_k = self.rerank_top_k if isinstance(self.rerank_top_k, int) and self.rerank_top_k > 0 else 3
        candidate_docs = []
        for row in retrieval_rows:
            metadata = dict(row.get("metadata", {}) or {})
            metadata["_rerank_id"] = row["result_id"]
            metadata["_source_rank"] = row["rank"]
            metadata["_source_score"] = row["score"]
            metadata["_vector_score"] = row.get("vector_score")
            metadata["_search_score"] = row.get("search_score")
            candidate_docs.append(Document(page_content=row["text"], metadata=metadata))

        if hasattr(self._reranker, "top_k"):
            try:
                self._reranker.top_k = min(rerank_k, len(candidate_docs))
            except Exception:
                pass

        compressed_docs = self._reranker.compress_documents(candidate_docs, query)
        reranked_rows = []
        for rank, doc in enumerate(compressed_docs[:rerank_k], start=1):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            reranked_rows.append(
                {
                    "result_id": metadata.get("_rerank_id", f"rerank-{rank}"),
                    "rank": rank,
                    "original_rank": metadata.get("_source_rank"),
                    "source_score": metadata.get("_source_score"),
                    "vector_score": metadata.get("_vector_score"),
                    "search_score": metadata.get("_search_score"),
                    "relevance_score": metadata.get("relevance_score"),
                    "text": getattr(doc, "page_content", ""),
                    "metadata": metadata,
                }
            )
        return reranked_rows

    def _score_ranked_results(self, ranked_rows, relevant_chunk_ids, supporting_quote):
        if not ranked_rows:
            return {"recall": 0.0, "ndcg": 0.0, "first_relevant_rank": None}

        supporting_quote = (supporting_quote or "").strip()
        graded_rels = []
        hits = 0
        first_relevant_rank = None
        for rank, row in enumerate(ranked_rows, start=1):
            chunk_id = row.get("result_id")
            text = row.get("text", "")
            if chunk_id in relevant_chunk_ids:
                hits += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank
                gain = 3.0 if supporting_quote and supporting_quote in text else 2.0
            else:
                gain = 0.0
            graded_rels.append(gain)

        relevant_total = max(len(relevant_chunk_ids), 1)
        recall = hits / relevant_total

        def dcg(values):
            total = 0.0
            for idx, rel in enumerate(values, start=1):
                if rel > 0:
                    total += (2 ** rel - 1) / math.log2(idx + 1)
            return total

        ideal_values = sorted(
            [3.0] * min(len(relevant_chunk_ids), len(ranked_rows)),
            reverse=True,
        )
        ndcg = dcg(graded_rels) / dcg(ideal_values) if ideal_values and dcg(ideal_values) > 0 else 0.0

        return {
            "recall": float(recall),
            "ndcg": float(ndcg),
            "first_relevant_rank": first_relevant_rank,
        }

    def _mean_metric(self, rows, key):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        return float(sum(values) / len(values)) if values else 0.0

    def _get_token_encoding(self):
        if self._token_encoding is not None:
            return self._token_encoding
        if tiktoken is None:
            return None
        try:
            self._token_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._token_encoding = None
        return self._token_encoding

    def _count_text_tokens(self, text):
        text = text or ""
        encoding = self._get_token_encoding()
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception:
                pass
        # Fallback approximation when tiktoken is unavailable.
        coarse_units = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return max(1, math.ceil(len(text) / 4), len(coarse_units)) if text else 0

    def _count_rows_tokens(self, rows):
        return int(sum(self._count_text_tokens(row.get("text", "")) for row in (rows or [])))

    def _count_rerank_request_tokens(self, query, retrieval_rows):
        return int(self._count_text_tokens(query) + self._count_rows_tokens(retrieval_rows))

    def _normalize_model_name(self, model_name):
        normalized = str(model_name or "").strip().lower()
        if not normalized:
            return ""
        if "/" in normalized:
            normalized = normalized.split("/")[-1]
        normalized = normalized.replace("_", "-")
        if "voyage-4-large" in normalized:
            return "voyage-4-large"
        if "voyage-4-lite" in normalized:
            return "voyage-4-lite"
        if "voyage-4" in normalized:
            return "voyage-4"
        if "voyage-context-3" in normalized:
            return "voyage-context-3"
        if "voyage-3.5-lite" in normalized:
            return "voyage-3.5-lite"
        if "voyage-3.5" in normalized:
            return "voyage-3.5"
        if "voyage-3-large" in normalized:
            return "voyage-3-large"
        if "voyage-3-lite" in normalized:
            return "voyage-3-lite"
        if "voyage-3" in normalized:
            return "voyage-3"
        if "voyage-code-3" in normalized:
            return "voyage-code-3"
        if "rerank-2.5-lite" in normalized:
            return "rerank-2.5-lite"
        if "rerank-2.5" in normalized:
            return "rerank-2.5"
        if "rerank-2-lite" in normalized:
            return "rerank-2-lite"
        if "rerank-2" in normalized:
            return "rerank-2"
        if "rerank-lite-1" in normalized:
            return "rerank-lite-1"
        if "rerank-1" in normalized:
            return "rerank-1"
        return normalized

    def _selected_embedding_model_name(self):
        index = int(self.selected_embedding_index or 0)
        if 0 <= index < len(self.embedding_models):
            return self.embedding_models[index]
        return self.embedding_model or ""

    def _selected_reranker_model_name(self):
        index = int(self.selected_reranker_index or 0)
        if 0 <= index < len(self.reranker_models):
            return self.reranker_models[index]
        return ""

    def _lookup_voyage_embedding_price(self, model_name):
        normalized = self._normalize_model_name(model_name)
        return VOYAGE_EMBEDDING_PRICES_PER_MILLION.get(normalized)

    def _lookup_voyage_rerank_price(self, model_name):
        normalized = self._normalize_model_name(model_name)
        return VOYAGE_RERANK_PRICES_PER_MILLION.get(normalized)

    def _log_descending_score(self, value, best, worst):
        if value is None:
            return None
        value = float(value)
        best = float(best)
        worst = float(worst)
        if value <= best:
            return 100.0
        if value >= worst:
            return 0.0
        if best <= 0 or worst <= best:
            return None
        ratio = math.log(value / best) / math.log(worst / best)
        return max(0.0, min(100.0, 100.0 * (1.0 - ratio)))

    def _soft_log_descending_score(self, value, best, worst, power=SOFT_COST_SCORE_POWER):
        if value is None:
            return None
        value = float(value)
        best = float(best)
        worst = float(worst)
        if value <= best:
            return 100.0
        if value >= worst:
            return 0.0
        if best <= 0 or worst <= best:
            return None
        ratio = math.log(value / best) / math.log(worst / best)
        softened_ratio = ratio ** float(power)
        return max(0.0, min(100.0, 100.0 * (1.0 - softened_ratio)))

    def _safe_score_average(self, *values):
        scored_values = [float(value) for value in values if isinstance(value, (int, float))]
        if not scored_values:
            return None
        return sum(scored_values) / len(scored_values)

    def _normalize_benchmark_score_weights(self, weights):
        merged = dict(DEFAULT_BENCHMARK_SCORE_WEIGHTS)
        if isinstance(weights, dict):
            for key in DEFAULT_BENCHMARK_SCORE_WEIGHTS:
                value = weights.get(key)
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                merged[key] = max(0.0, numeric_value)

        total = sum(merged.values())
        if total <= 0:
            return dict(DEFAULT_BENCHMARK_SCORE_WEIGHTS)
        return {key: value / total for key, value in merged.items()}

    def _weighted_score_average(self, scores, weights):
        weighted_total = 0.0
        weight_total = 0.0
        for key, score in scores.items():
            if not isinstance(score, (int, float)):
                continue
            weight = float(weights.get(key, 0.0))
            if weight <= 0:
                continue
            weighted_total += float(score) * weight
            weight_total += weight
        if weight_total <= 0:
            return None
        return weighted_total / weight_total

    def _normalize_constraints(self, constraints):
        normalized = dict(DEFAULT_BENCHMARK_CONSTRAINTS)
        if isinstance(constraints, dict):
            for key in ("min_k", "max_k"):
                value = constraints.get(key)
                try:
                    numeric_value = int(value) if value is not None else None
                except (TypeError, ValueError):
                    numeric_value = None
                normalized[key] = max(1, numeric_value) if numeric_value is not None else None

            for key in ("min_chunk_size", "max_chunk_size"):
                value = constraints.get(key, normalized[key])
                try:
                    normalized[key] = max(1, int(value))
                except (TypeError, ValueError):
                    pass

        if normalized["min_k"] is not None and normalized["max_k"] is not None and normalized["max_k"] < normalized["min_k"]:
            normalized["max_k"] = normalized["min_k"]
        if normalized["max_chunk_size"] < normalized["min_chunk_size"]:
            normalized["max_chunk_size"] = normalized["min_chunk_size"]
        return normalized

    def _normalize_benchmark_scoreboard(self, scoreboard):
        normalized = dict(DEFAULT_BENCHMARK_SCOREBOARD)
        if isinstance(scoreboard, dict):
            for key in normalized:
                value = scoreboard.get(key)
                normalized[key] = str(value).strip() if value is not None else ""
        return normalized

    def _has_scoreboard_config(self):
        config = getattr(self, "_benchmark_scoreboard", None) or {}
        return bool(
            config.get("mdb_uri")
            and config.get("database")
            and config.get("collection")
        )

    def _canonicalize_score_weights(self, weights):
        normalized = self._normalize_benchmark_score_weights(weights)
        rounded = {
            key: round(float(normalized.get(key, 0.0)), 6)
            for key in DEFAULT_BENCHMARK_SCORE_WEIGHTS
        }
        key = json.dumps(rounded, sort_keys=True, separators=(",", ":"))
        return rounded, key

    def _canonicalize_constraints(self, constraints):
        normalized = self._normalize_constraints(constraints)
        key = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return normalized, key

    def _constraint_k_bounds(self):
        constraints = getattr(self, "constraints", None) or self._constraints
        if isinstance(constraints, dict):
            min_k = constraints.get("min_k")
            max_k = constraints.get("max_k")
            if not isinstance(min_k, int) or min_k <= 0:
                min_k = None
            if not isinstance(max_k, int) or max_k <= 0:
                max_k = None
            return min_k, max_k
        return None, None

    def _constraint_chunk_size_bounds(self):
        constraints = getattr(self, "constraints", None) or self._constraints
        if not isinstance(constraints, dict):
            constraints = DEFAULT_BENCHMARK_CONSTRAINTS
        min_chunk_size = max(1, int(constraints.get("min_chunk_size", 1)))
        max_chunk_size = max(min_chunk_size, int(constraints.get("max_chunk_size", 2000)))
        return min_chunk_size, max_chunk_size

    def _apply_constraints(self):
        min_k, max_k = self._constraint_k_bounds()
        min_chunk_size, max_chunk_size = self._constraint_chunk_size_bounds()

        if self.chunk_size < min_chunk_size:
            self.chunk_size = min_chunk_size
            return
        if self.chunk_size > max_chunk_size:
            self.chunk_size = max_chunk_size
            return

        if min_k is not None and max_k is not None and min_k == max_k:
            if self.top_k != min_k:
                self.top_k = min_k
                return
            if bool(self.reranking_enabled) and self.rerank_top_k != min_k:
                self.rerank_top_k = min_k
                return
            return

        effective_top_k_min = min_k if min_k is not None else 1
        effective_top_k_max = max_k if max_k is not None else None
        clamped_top_k = max(effective_top_k_min, int(self.top_k or effective_top_k_min))
        if effective_top_k_max is not None:
            clamped_top_k = min(clamped_top_k, effective_top_k_max)
        if clamped_top_k != self.top_k:
            self.top_k = clamped_top_k
            return

        if bool(self.reranking_enabled):
            effective_rerank_min = min_k if min_k is not None else 1
            effective_rerank_max = self.top_k
            if max_k is not None:
                effective_rerank_max = min(effective_rerank_max, max_k)
            effective_rerank_max = max(effective_rerank_min, effective_rerank_max)

            clamped_rerank_top_k = max(
                effective_rerank_min,
                int(self.rerank_top_k or effective_rerank_min),
            )
            clamped_rerank_top_k = min(clamped_rerank_top_k, effective_rerank_max)
            if clamped_rerank_top_k != self.rerank_top_k:
                self.rerank_top_k = clamped_rerank_top_k
                return

    def _get_scoreboard_collection(self):
        if not self._has_scoreboard_config():
            return None
        if self._scoreboard_collection is not None:
            return self._scoreboard_collection
        config = self._benchmark_scoreboard
        if self._scoreboard_client is None:
            self._scoreboard_client = MongoClient(config["mdb_uri"])
        self._scoreboard_collection = self._scoreboard_client[config["database"]][config["collection"]]
        return self._scoreboard_collection

    def _current_scoreboard_filter(self):
        weights, weights_key = self._canonicalize_score_weights(self.benchmark_score_weights)
        constraints, constraints_key = self._canonicalize_constraints(self.constraints)
        return (
            {
                "score_weights_key": weights_key,
                "constraints_key": constraints_key,
            },
            weights,
            constraints,
        )

    def load_scoreboard(self):
        self.scoreboard_available = self._has_scoreboard_config()
        if not self.scoreboard_available:
            self.scoreboard_entries = []
            return
        self.scoreboard_loading = True
        try:
            collection = self._get_scoreboard_collection()
            filter_query, _, _ = self._current_scoreboard_filter()
            docs = list(
                collection.find(
                    filter_query,
                    {
                        "_id": 0,
                        "name": 1,
                        "overall_score": 1,
                        "accuracy_score": 1,
                        "latency_score": 1,
                        "cost_score": 1,
                        "published_at": 1,
                    },
                ).sort("overall_score", -1)
            )
            scoreboard_entries = []
            for rank, doc in enumerate(docs, start=1):
                scoreboard_entries.append(
                    {
                        "rank": rank,
                        "name": str(doc.get("name") or "Anonymous"),
                        "overall_score": doc.get("overall_score"),
                        "accuracy_score": doc.get("accuracy_score"),
                        "latency_score": doc.get("latency_score"),
                        "cost_score": doc.get("cost_score"),
                        "published_at": doc.get("published_at"),
                    }
                )
            self.scoreboard_entries = scoreboard_entries
        except Exception as e:
            self.scoreboard_entries = []
            self.set_error(f"Scoreboard Error: {e}")
        finally:
            self.scoreboard_loading = False

    def publish_scoreboard(self):
        if not self._has_scoreboard_config():
            self.set_error("No benchmark scoreboard configuration provided.")
            return
        summary = self.benchmark_summary if isinstance(self.benchmark_summary, dict) else {}
        if not summary or not isinstance(summary.get("overall_score"), (int, float)):
            self.set_error("Run a benchmark before publishing to the scoreboard.")
            return
        name = (self.scoreboard_entry_name or "").strip()
        if not name:
            self.set_error("Please enter your name before publishing to the scoreboard.")
            return

        self.scoreboard_loading = True
        try:
            collection = self._get_scoreboard_collection()
            _, normalized_weights, normalized_constraints = self._current_scoreboard_filter()
            weights_key = json.dumps(normalized_weights, sort_keys=True, separators=(",", ":"))
            constraints_key = json.dumps(normalized_constraints, sort_keys=True, separators=(",", ":"))
            document = {
                "name": name,
                "overall_score": float(summary.get("overall_score")),
                "accuracy_score": float(summary.get("accuracy_score")),
                "latency_score": float(summary.get("latency_score")),
                "cost_score": float(summary.get("cost_score")),
                "score_weights": normalized_weights,
                "constraints": normalized_constraints,
                "score_weights_key": weights_key,
                "constraints_key": constraints_key,
                "published_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }
            collection.insert_one(document)
            self.load_scoreboard()
        except Exception as e:
            self.set_error(f"Scoreboard Publish Error: {e}")
        finally:
            self.scoreboard_loading = False

    def _estimate_benchmark_costs(
        self,
        embedding_total_token_count,
        rerank_total_token_count,
        llm_context_total_token_count,
        llm_context_min_token_count,
        llm_context_max_token_count,
    ):
        embedding_model_name = self._selected_embedding_model_name()
        reranker_model_name = self._selected_reranker_model_name()
        embedding_price = self._lookup_voyage_embedding_price(embedding_model_name)
        rerank_price = self._lookup_voyage_rerank_price(reranker_model_name)

        embedding_cost = (
            float(embedding_total_token_count) / 1_000_000.0 * embedding_price
            if embedding_price is not None
            else None
        )
        rerank_cost = (
            float(rerank_total_token_count) / 1_000_000.0 * rerank_price
            if rerank_price is not None and rerank_total_token_count > 0
            else 0.0
        )
        llm_context_cost = (
            float(llm_context_total_token_count) / 1_000_000.0 * LLM_CONTEXT_PRICE_PER_MILLION
        )
        min_llm_context_cost = (
            float(llm_context_min_token_count) / 1_000_000.0 * LLM_CONTEXT_PRICE_PER_MILLION
        )
        max_llm_context_cost = (
            float(llm_context_max_token_count) / 1_000_000.0 * LLM_CONTEXT_PRICE_PER_MILLION
        )

        total_cost = None if embedding_cost is None else embedding_cost + rerank_cost + llm_context_cost

        cheapest_embedding_price = min(VOYAGE_EMBEDDING_PRICES_PER_MILLION.values())
        priciest_embedding_price = max(VOYAGE_EMBEDDING_PRICES_PER_MILLION.values())
        cheapest_rerank_price = min(VOYAGE_RERANK_PRICES_PER_MILLION.values()) if rerank_total_token_count > 0 else 0.0
        priciest_rerank_price = max(VOYAGE_RERANK_PRICES_PER_MILLION.values()) if rerank_total_token_count > 0 else 0.0

        cheapest_total_cost = (
            float(embedding_total_token_count) / 1_000_000.0 * cheapest_embedding_price
            + float(rerank_total_token_count) / 1_000_000.0 * cheapest_rerank_price
            + min_llm_context_cost
        )
        priciest_total_cost = (
            float(embedding_total_token_count) / 1_000_000.0 * priciest_embedding_price
            + float(rerank_total_token_count) / 1_000_000.0 * priciest_rerank_price
            + max_llm_context_cost
        )

        if total_cost is None or priciest_total_cost <= cheapest_total_cost:
            cost_score = 100.0 if total_cost is not None else None
        else:
            cost_score = self._soft_log_descending_score(
                max(total_cost, cheapest_total_cost),
                cheapest_total_cost,
                priciest_total_cost,
            )

        return {
            "embedding_model_name": embedding_model_name,
            "reranker_model_name": reranker_model_name,
            "embedding_price_per_million": embedding_price,
            "rerank_price_per_million": rerank_price,
            "llm_context_price_per_million": LLM_CONTEXT_PRICE_PER_MILLION,
            "embedding_cost_usd": embedding_cost,
            "rerank_cost_usd": rerank_cost,
            "llm_context_cost_usd": llm_context_cost,
            "min_llm_context_cost_usd": min_llm_context_cost,
            "max_llm_context_cost_usd": max_llm_context_cost,
            "total_cost_usd": total_cost,
            "cheapest_total_cost_usd": cheapest_total_cost,
            "priciest_total_cost_usd": priciest_total_cost,
            "cost_score": cost_score,
        }

    def _run_benchmark(self):
        self.benchmark_results = []
        self.benchmark_summary = {}
        self._reset_benchmark_status()
        self.clear_error()

        if not self.benchmark_dataset_loaded:
            self.set_error("No benchmark dataset provided.")
            return
        if self._embedding_model is None:
            self.set_error("No embedding model provided.")
            return
        if self.mongo_collection is None:
            self.set_error("No MongoDB collection provided for benchmark.")
            return
        if not self.selected_index:
            self.set_error("No vector index was specified for benchmark.")
            return

        self.benchmark_running = True
        benchmark_collection = None
        try:
            example_count = len(self._benchmark_dataset.get("examples", []) or [])
            self._update_benchmark_status(
                phase="preparing",
                message="Preparing benchmark corpus and evaluation plan...",
                completed=0,
                total=example_count,
                primary_label="queries",
                secondary_label="setup in progress",
            )
            chunk_rows = self._build_chunk_rows_for_documents(self._benchmark_dataset.get("sources", []))
            embedding_corpus_token_count = self._count_rows_tokens(chunk_rows)
            self._update_benchmark_status(
                phase="loading",
                message=(
                    f"Loading benchmark corpus into MongoDB ({len(chunk_rows)} chunks from "
                    f"{len(self._benchmark_dataset.get('sources', []) or [])} source documents)..."
                ),
                completed=0,
                total=example_count,
                primary_label="queries",
                secondary_label=f"{len(chunk_rows)} chunks prepared",
            )
            benchmark_collection = self._load_benchmark_documents_into_mongo(chunk_rows)
            if benchmark_collection is None:
                self._update_benchmark_status(
                    phase="error",
                    message="Benchmark corpus could not be loaded into MongoDB.",
                    completed=0,
                    total=example_count,
                    primary_label="queries",
                )
                return
            probe_query = (
                (self._benchmark_dataset.get("examples", [{}])[0] or {}).get("question")
                or "benchmark probe"
            )
            self._update_benchmark_status(
                phase="waiting",
                message="Waiting for benchmark documents to become searchable in MongoDB (this can take a moment)...",
                completed=0,
                total=example_count,
                primary_label="queries",
                secondary_label="checking search readiness",
            )
            self._wait_for_benchmark_documents_searchable(
                benchmark_collection,
                probe_query=probe_query,
                expected_min_results=1,
                timeout_seconds=30,
            )
            chunk_rows_by_id = {row["chunk_id"]: row for row in chunk_rows}
            relevance_by_seed = self._build_benchmark_relevance(chunk_rows)
            reranking_enabled = bool(self.reranking_enabled and self._reranker is not None)

            per_query_results = []
            for query_index, example in enumerate(self._benchmark_dataset.get("examples", []), start=1):
                query = example.get("question", "")
                relevant_seed_id = example.get("relevant_seed_id")
                relevant_chunk_ids = relevance_by_seed.get(relevant_seed_id, set())
                embedding_model_token_count = self._count_text_tokens(query)
                query_preview = query[:90] + "..." if len(query) > 90 else query
                self._update_benchmark_status(
                    phase="running",
                    message=f"Running benchmark query {query_index} / {example_count}: {query_preview}",
                    completed=query_index - 1,
                    total=example_count,
                    primary_label="queries",
                    secondary_label="retrieval and scoring in progress",
                )

                task_started_at = time.perf_counter()
                retrieval_started_at = time.perf_counter()
                retrieval_rows = self._benchmark_retrieve(query, benchmark_collection)
                retrieval_latency_ms = (time.perf_counter() - retrieval_started_at) * 1000.0
                if reranking_enabled:
                    rerank_started_at = time.perf_counter()
                    rerank_rows = self._benchmark_rerank(query, retrieval_rows)
                    rerank_latency_ms = (time.perf_counter() - rerank_started_at) * 1000.0
                else:
                    rerank_rows = []
                    rerank_latency_ms = None
                total_latency_ms = (time.perf_counter() - task_started_at) * 1000.0
                retrieval_token_count = self._count_rows_tokens(retrieval_rows)
                rerank_token_count = self._count_rows_tokens(rerank_rows)
                rerank_model_token_count = self._count_rerank_request_tokens(query, retrieval_rows) if rerank_rows else 0

                retrieval_metrics = self._score_ranked_results(
                    retrieval_rows,
                    relevant_chunk_ids,
                    example.get("supporting_quote", ""),
                )
                rerank_metrics = self._score_ranked_results(
                    rerank_rows,
                    relevant_chunk_ids,
                    example.get("supporting_quote", ""),
                ) if rerank_rows else {"recall": None, "ndcg": None, "first_relevant_rank": None}
                expected_chunk_ids = sorted(relevant_chunk_ids)
                expected_chunks = [
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_rows_by_id.get(chunk_id, {}).get("text", ""),
                    }
                    for chunk_id in expected_chunk_ids
                ]

                per_query_results.append(
                    {
                        "qid": example.get("qid", ""),
                        "question": query,
                        "relevant_seed_id": relevant_seed_id,
                        "relevant_chunk_count": len(relevant_chunk_ids),
                        "retrieval_latency_ms": float(retrieval_latency_ms),
                        "rerank_latency_ms": float(rerank_latency_ms) if rerank_latency_ms is not None else None,
                        "total_latency_ms": float(total_latency_ms),
                        "embedding_model_token_count": int(embedding_model_token_count),
                        "rerank_model_token_count": int(rerank_model_token_count),
                        "retrieval_token_count": int(retrieval_token_count),
                        "rerank_token_count": int(rerank_token_count),
                        "retrieval_recall": retrieval_metrics["recall"],
                        "retrieval_ndcg": retrieval_metrics["ndcg"],
                        "retrieval_first_relevant_rank": retrieval_metrics["first_relevant_rank"],
                        "rerank_recall": rerank_metrics["recall"],
                        "rerank_ndcg": rerank_metrics["ndcg"],
                        "rerank_first_relevant_rank": rerank_metrics["first_relevant_rank"],
                        "expected_chunk_ids": expected_chunk_ids,
                        "expected_chunks": expected_chunks,
                        "retrieval_results": retrieval_rows,
                        "rerank_results": rerank_rows,
                    }
                )
                self._update_benchmark_status(
                    phase="running",
                    message=f"Completed benchmark query {query_index} / {example_count}: {query_preview}",
                    completed=query_index,
                    total=example_count,
                    primary_label="queries",
                    secondary_label="results and metrics updated",
                )

            self.benchmark_results = per_query_results
            retrieval_mean_recall = self._mean_metric(per_query_results, "retrieval_recall")
            retrieval_mean_ndcg = self._mean_metric(per_query_results, "retrieval_ndcg")
            rerank_mean_recall = self._mean_metric(per_query_results, "rerank_recall")
            rerank_mean_ndcg = self._mean_metric(per_query_results, "rerank_ndcg")
            effective_recall = rerank_mean_recall if reranking_enabled else retrieval_mean_recall
            effective_ndcg = rerank_mean_ndcg if reranking_enabled else retrieval_mean_ndcg
            accuracy_score = 100.0 * ((0.6 * effective_recall) + (0.4 * effective_ndcg))
            total_mean_latency_ms = self._mean_metric(per_query_results, "total_latency_ms")
            latency_score = self._log_descending_score(
                total_mean_latency_ms,
                LATENCY_SCORE_BEST_MS,
                LATENCY_SCORE_WORST_MS,
            )
            retrieval_total_token_count = int(sum(int(row.get("retrieval_token_count", 0)) for row in per_query_results))
            reranked_context_total_token_count = int(sum(int(row.get("rerank_token_count", 0)) for row in per_query_results))
            llm_context_total_token_count = (
                reranked_context_total_token_count if reranking_enabled and reranked_context_total_token_count > 0
                else retrieval_total_token_count
            )
            llm_context_mean_token_count = (
                self._mean_metric(per_query_results, "rerank_token_count")
                if reranking_enabled and reranked_context_total_token_count > 0
                else self._mean_metric(per_query_results, "retrieval_token_count")
            )
            llm_context_min_token_count = int(LLM_CONTEXT_MIN_TOKENS)
            llm_context_max_token_count = int(
                LLM_CONTEXT_MAX_DOCUMENTS * max(1, self.chunk_size)
            )
            embedding_total_token_count = int(
                sum(int(row.get("embedding_model_token_count", 0)) for row in per_query_results)
            )
            embedding_mean_token_count = self._mean_metric(per_query_results, "embedding_model_token_count")
            rerank_total_token_count = int(sum(int(row.get("rerank_model_token_count", 0)) for row in per_query_results))
            rerank_mean_token_count = self._mean_metric(per_query_results, "rerank_model_token_count")
            cost_metrics = self._estimate_benchmark_costs(
                embedding_total_token_count=embedding_mean_token_count,
                rerank_total_token_count=rerank_mean_token_count,
                llm_context_total_token_count=llm_context_mean_token_count,
                llm_context_min_token_count=llm_context_min_token_count,
                llm_context_max_token_count=llm_context_max_token_count,
            )
            effective_score_weights = self._normalize_benchmark_score_weights(
                self.benchmark_score_weights
            )
            effective_constraints = self._normalize_constraints(self.constraints)
            _, score_weights_key = self._canonicalize_score_weights(effective_score_weights)
            _, constraints_key = self._canonicalize_constraints(effective_constraints)
            overall_score = self._weighted_score_average(
                {
                    "accuracy": accuracy_score,
                    "latency": latency_score,
                    "cost": cost_metrics["cost_score"],
                },
                effective_score_weights,
            )
            self.benchmark_summary = {
                "query_count": len(per_query_results),
                "retrieval_k": self.top_k if isinstance(self.top_k, int) and self.top_k > 0 else 5,
                "rerank_k": self.rerank_top_k if isinstance(self.rerank_top_k, int) and self.rerank_top_k > 0 else 3,
                "embedding_corpus_token_count": int(embedding_corpus_token_count),
                "embedding_total_token_count": embedding_total_token_count,
                "retrieval_total_token_count": retrieval_total_token_count,
                "reranked_context_total_token_count": reranked_context_total_token_count,
                "llm_context_total_token_count": llm_context_total_token_count,
                "llm_context_mean_token_count": llm_context_mean_token_count,
                "llm_context_min_token_count": llm_context_min_token_count,
                "llm_context_max_token_count": llm_context_max_token_count,
                "rerank_total_token_count": rerank_total_token_count,
                "embedding_mean_token_count": embedding_mean_token_count,
                "rerank_model_mean_token_count": rerank_mean_token_count,
                "retrieval_mean_latency_ms": self._mean_metric(per_query_results, "retrieval_latency_ms"),
                "rerank_mean_latency_ms": self._mean_metric(per_query_results, "rerank_latency_ms"),
                "total_mean_latency_ms": total_mean_latency_ms,
                "retrieval_mean_token_count": self._mean_metric(per_query_results, "retrieval_token_count"),
                "rerank_mean_token_count": self._mean_metric(per_query_results, "rerank_token_count"),
                "token_count_method": "tiktoken_cl100k_base" if self._get_token_encoding() is not None else "estimated_chars_div_4",
                "retrieval_mean_recall": retrieval_mean_recall,
                "retrieval_mean_ndcg": retrieval_mean_ndcg,
                "rerank_mean_recall": rerank_mean_recall,
                "rerank_mean_ndcg": rerank_mean_ndcg,
                "effective_mean_recall": effective_recall,
                "effective_mean_ndcg": effective_ndcg,
                "accuracy_score": accuracy_score,
                "latency_score": latency_score,
                "cost_score": cost_metrics["cost_score"],
                "overall_score": overall_score,
                "score_weights": effective_score_weights,
                "score_weights_key": score_weights_key,
                "constraints": effective_constraints,
                "constraints_key": constraints_key,
                "embedding_cost_usd": cost_metrics["embedding_cost_usd"],
                "rerank_cost_usd": cost_metrics["rerank_cost_usd"],
                "llm_context_cost_usd": cost_metrics["llm_context_cost_usd"],
                "min_llm_context_cost_usd": cost_metrics["min_llm_context_cost_usd"],
                "max_llm_context_cost_usd": cost_metrics["max_llm_context_cost_usd"],
                "total_cost_usd": cost_metrics["total_cost_usd"],
                "cheapest_total_cost_usd": cost_metrics["cheapest_total_cost_usd"],
                "priciest_total_cost_usd": cost_metrics["priciest_total_cost_usd"],
                "embedding_price_per_million": cost_metrics["embedding_price_per_million"],
                "rerank_price_per_million": cost_metrics["rerank_price_per_million"],
                "llm_context_price_per_million": cost_metrics["llm_context_price_per_million"],
                "embedding_model_name": cost_metrics["embedding_model_name"],
                "reranker_model_name": cost_metrics["reranker_model_name"],
                "hybrid_enabled": int(self.search_sub_tab or 1) == 2,
                "reranking_enabled": reranking_enabled,
                "benchmark_dataset_name": self.benchmark_dataset_name,
            }
            self._update_benchmark_status(
                phase="completed",
                message=f"Benchmark complete. Evaluated {len(per_query_results)} queries and refreshed the summary.",
                completed=len(per_query_results),
                total=example_count,
                primary_label="queries",
                secondary_label="summary ready",
            )
        except Exception as e:
            self._update_benchmark_status(
                phase="error",
                message="Benchmark run failed before completion.",
                completed=0,
                total=len(self._benchmark_dataset.get("examples", []) or []),
                primary_label="queries",
            )
            self.set_error(f"Benchmark Error: {e}")
        finally:
            self.benchmark_running = False

    # -----------------
    # Step 3: Vector Search
    # -----------------
    def run_vector_search(self):
        self._run_vector_query()

    def run_hybrid_search(self):
        self._run_hybrid_query()

    def run_reranking(self):
        self._run_reranking()

    def run_benchmark(self):
        self._run_benchmark()

    def explain_benchmark_task(self):
        qid = (self.benchmark_selected_qid or "").strip()
        if not qid:
            self.set_error("No benchmark task selected.")
            return

        result = next((row for row in self.benchmark_results if row.get("qid") == qid), None)
        if result is None:
            self.set_error(f"No benchmark result found for qid '{qid}'.")
            return

        return

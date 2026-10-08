// index.jsx (front-end for MongoDBRAGPlayground using React + @anywidget/react)
import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import {
  autoUpdate,
  flip,
  offset,
  shift,
  useDismiss,
  useFloating,
  useFocus,
  useHover,
  useInteractions,
  useRole,
} from "@floating-ui/react";
import { UMAP } from "umap-js";

/** * @typedef {Object} ChunkRow 
 * @property {number} page_index 
 * @property {number} chunk_index 
 * @property {string} chunk_text 
 * @property {number} [start_offset] 
 * @property {number} [end_offset] 
 */
/** * @typedef {Object} MongoDoc 
 * @property {string} _id 
 * @property {string} text 
 * @property {Array<any>} embedding 
 */
/** * @typedef {Object} RagResult 
 * @property {number} score 
 * @property {string} content 
 * @property {Object} [metadata] 
 */
function TwoColumnLayout({ left, right, align = "stretch" }) {
  return (
    <div
      className="two-column-layout"
      style={{
        display: "flex",
        flexWrap: "nowrap", // Prevent wrapping to keep heights synced
        gap: "2rem",
        alignItems: align,
        width: "100%",
        // Height is now handled by the parent flex-grow
      }}
    >
      <div style={{ flex: "1", minWidth: 0, height: "100%" }}>{left}</div>
      <div style={{ flex: "1", minWidth: 0, height: "100%" }}>{right}</div>
    </div>
  );
}

function StepsNav({ currentStep, onChange, benchmarkTabVisible, scoreboardTabVisible }) {
  const steps = [
    { label: "Chunk →", value: 1 },
    { label: "Embed →", value: 2 },
    { label: "Search →", value: 3 },
    { label: "Rerank →", value: 4 },
    benchmarkTabVisible ? { label: "Benchmark", value: 5 } : null,
    scoreboardTabVisible ? { label: "Scoreboard", value: 6, alignRight: true } : null,
  ].filter(Boolean);
  return (
    <div className="steps-nav">
      {steps.map((step) => (
        <button
          key={step.value}
          className={
            "step-button" + (currentStep === step.value ? " active-step" : "")
          }
          style={step.alignRight ? { marginLeft: "auto" } : undefined}
          onClick={() => onChange(step.value)}
          type="button"
        >
          {step.label}
        </button>
      ))}
    </div>
  );
}

function ErrorBox() {
  const model = useModel();
  const [errorTrait] = useModelState("error");
  const [error, setError] = React.useState(errorTrait || "");
  React.useEffect(() => {
    setError(errorTrait || "");
  }, [errorTrait]);
  React.useEffect(() => {
    function handleCustom(msg) {
      if (msg && msg.type === "update_error") {
        setError(msg.error || "");
      }
    }
    model.on("msg:custom", handleCustom);
    return () => model.off("msg:custom", handleCustom);
  }, [model]);
  if (!error) return null;
  return (
    <div
      id="error-alert"
      className="alert alert-danger"
      style={{ display: "flex" }}
    >
      {error}
    </div>
  );
}

function getConstraintKBounds(constraints) {
  const minK = typeof constraints?.min_k === "number" && constraints.min_k > 0 ? constraints.min_k : null;
  const maxK = typeof constraints?.max_k === "number" && constraints.max_k > 0 ? constraints.max_k : null;
  return { minK, maxK };
}

function clampInteger(value, minValue, maxValue) {
  const fallback = minValue ?? 1;
  const numericValue = Number(value);
  let nextValue = Number.isFinite(numericValue) ? Math.trunc(numericValue) : fallback;
  if (minValue != null) {
    nextValue = Math.max(minValue, nextValue);
  }
  if (maxValue != null) {
    nextValue = Math.min(maxValue, nextValue);
  }
  return nextValue;
}

function HelpTooltip({ message, ariaLabel = "More information" }) {
  const [open, setOpen] = React.useState(false);
  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange: setOpen,
    placement: "right",
    whileElementsMounted: autoUpdate,
    middleware: [
      offset(8),
      flip({
        fallbackPlacements: ["left", "bottom", "top"],
      }),
      shift({ padding: 8 }),
    ],
  });

  const hover = useHover(context, { move: false });
  const focus = useFocus(context);
  const dismiss = useDismiss(context);
  const role = useRole(context, { role: "tooltip" });
  const { getReferenceProps, getFloatingProps } = useInteractions([
    hover,
    focus,
    dismiss,
    role,
  ]);

  return (
    <span className="info-tooltip">
      <button
        ref={refs.setReference}
        className="info-tooltip-trigger"
        type="button"
        aria-label={ariaLabel}
        {...getReferenceProps()}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          fill="none"
          viewBox="0 0 16 16"
          role="img"
          aria-hidden="true"
        >
          <path
            fill="currentColor"
            fillRule="evenodd"
            d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14M6.932 5.612C7.054 5.298 7.425 5 7.942 5 8.615 5 9 5.478 9 5.875c0 .162-.057.33-.172.476a1 1 0 0 1-.242.216l-.016.01a1 1 0 0 1-.098.054c-.59.286-1.53.967-1.53 2.119V9a1 1 0 0 0 2 0c0-.035.011-.12.138-.27a2.7 2.7 0 0 1 .587-.48 3 3 0 0 0 .726-.656A2.74 2.74 0 0 0 11 5.875C11 4.201 9.54 3 7.941 3c-1.275 0-2.43.745-2.873 1.888a1 1 0 1 0 1.864.724M8 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2"
            clipRule="evenodd"
          />
        </svg>
      </button>
      {open ? (
        <span
          ref={refs.setFloating}
          className="info-tooltip-bubble"
          style={floatingStyles}
          {...getFloatingProps()}
        >
          {message}
        </span>
      ) : null}
    </span>
  );
}

function FieldLabel({ label, help, htmlFor }) {
  return (
    <label className="field-label" htmlFor={htmlFor}>
      <span>{label}</span>
      {help ? <HelpTooltip message={help} ariaLabel={`${label} help`} /> : null}
    </label>
  );
}

function SettingsIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M19.14 12.94c.04-.31.06-.62.06-.94s-.02-.63-.06-.94l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.39.96a7.03 7.03 0 0 0-1.63-.94l-.36-2.54a.5.5 0 0 0-.5-.42h-3.84a.5.5 0 0 0-.5.42l-.36 2.54c-.58.23-1.13.54-1.63.94l-2.39-.96a.5.5 0 0 0-.6.22L2.7 8.84a.5.5 0 0 0 .12.64l2.03 1.58c-.04.31-.06.62-.06.94s.02.63.06.94L2.82 14.52a.5.5 0 0 0-.12.64l1.92 3.32a.5.5 0 0 0 .6.22l2.39-.96c.5.4 1.05.71 1.63.94l.36 2.54a.5.5 0 0 0 .5.42h3.84a.5.5 0 0 0 .5-.42l.36-2.54c.58-.23 1.13-.54 1.63-.94l2.39.96a.5.5 0 0 0 .6-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58ZM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7Z" />
    </svg>
  );
}

function ProgressStatusPanel({
  status,
  defaultTitle = "Status",
  completeTitle = "Complete",
  errorTitle = "Failed",
  defaultMessage = "Working...",
}) {
  const phase = status?.phase || "idle";
  if (phase === "idle") return null;

  const totalChunks = Number(status?.total_chunks || 0);
  const totalDocuments = Number(status?.total_documents || 0);
  const chunksIndexed = Number(status?.chunks_indexed || 0);
  const documentsStored = Number(status?.documents_stored || 0);
  const completed = Number(status?.completed || 0);
  const total = Number(status?.total || 0);
  const primaryLabel = status?.primary_label || "items";
  const secondaryLabel = status?.secondary_label || "";
  const progress = Math.max(0, Math.min(1, Number(status?.progress || 0)));
  const percent = Math.round(progress * 100);
  const isError = phase === "error";
  const isComplete = phase === "completed";
  const title = isError
    ? errorTitle
    : isComplete
      ? completeTitle
      : defaultTitle;
  const hasIndexingCounts = totalChunks > 0 || totalDocuments > 0;
  const hasGenericCounts = total > 0;

  return (
    <div
      className={`indexing-progress-panel${isError ? " is-error" : ""}${isComplete ? " is-complete" : ""}`}
    >
      <div className="indexing-progress-header">
        <div>
          <div className="indexing-progress-title">{title}</div>
          {hasIndexingCounts && (
            <div className="indexing-progress-subtitle">
              {chunksIndexed} / {totalChunks || totalDocuments} chunks indexed
              {" · "}
              {documentsStored} / {totalDocuments || totalChunks} documents stored
            </div>
          )}
          {!hasIndexingCounts && hasGenericCounts && (
            <div className="indexing-progress-subtitle">
              {completed} / {total} {primaryLabel}
              {secondaryLabel ? ` · ${secondaryLabel}` : ""}
            </div>
          )}
        </div>
        <div className="indexing-progress-percent">{percent}%</div>
      </div>
      <div className="indexing-progress-message">
        {status?.message || defaultMessage}
      </div>
      <div className="indexing-progress-bar-wrap" aria-hidden="true">
        <div
          className="indexing-progress-bar"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function getConfigurationBlockReason(configurationStatus) {
  const phase = configurationStatus?.phase || "idle";
  if (phase === "completed") return "";
  if (phase === "error") {
    return configurationStatus?.message || "Configuration check failed.";
  }
  if (phase === "running" || phase === "pending") {
    return configurationStatus?.message || "Checking MongoDB and embedding API configuration...";
  }
  return "";
}

// Step 1: Chunking 
function ChunkingStep() {
  const [splitStrategy, setSplitStrategy] = useModelState("split_strategy");
  const [chunkSize, setChunkSize] = useModelState("chunk_size");
  const [overlapSize, setOverlapSize] = useModelState("overlap_size");
  const [constraints] = useModelState("constraints");
  const [currentDocIndex, setCurrentDocIndex] =
    useModelState("current_doc_index");
  const [documentPreview] = useModelState("document_preview");
  const [chunksTable] = useModelState("chunks_table");
  const minChunkSize =
    typeof constraints?.min_chunk_size === "number"
      ? constraints.min_chunk_size
      : 1;
  const maxChunkSize =
    typeof constraints?.max_chunk_size === "number"
      ? constraints.max_chunk_size
      : 2000;
  const filteredChunks = React.useMemo(
    () =>
      (chunksTable || []).filter(
        (row) => row.page_index === Number(currentDocIndex || 0),
      ),
    [chunksTable, currentDocIndex],
  );
  const handlePrev = () => {
    const idx = Number(currentDocIndex || 0);
    if (idx > 0) setCurrentDocIndex(idx - 1);
  };
  const handleNext = () => {
    const idx = Number(currentDocIndex || 0);
    setCurrentDocIndex(idx + 1);
  };
  return (
    <div id="chunking-section">
      <div id="settings-section">
        {/* Page nav first, on same row as others */}
        <div className="control-group">
          <FieldLabel
            label="Page Navigation"
            help="Move between source pages to inspect how the current chunking settings split the document."
          />
          {/* Removed inline style, now relying on .control-group > div:has(button) */}
          <div>
            <button
              className="action-button"
              type="button"
              onClick={handlePrev}
            >
              Previous Page
            </button>
            <button
              className="action-button"
              type="button"
              onClick={handleNext}
            >
              Next Page
            </button>
          </div>
        </div>
        <div className="control-group">
          <FieldLabel
            label="Split Strategy"
            help="Choose how the document is broken into chunks: fixed characters, recursive boundaries, or markdown-aware sections."
          />
          <select
            value={splitStrategy || "Fixed"}
            onChange={(e) => setSplitStrategy(e.target.value)}
          >
            <option value="Fixed">Fixed</option>
            <option value="Recursive">Recursive</option>
            <option value="Markdown">Markdown</option>
          </select>
        </div>
        <div className="control-group">
          <FieldLabel
            label="Chunk Size"
            help="Set the target amount of text in each chunk before embeddings are generated."
          />
          <div className="range-control">
            <input
              type="range"
              min={minChunkSize}
              max={Math.max(minChunkSize, maxChunkSize)}
              value={chunkSize ?? 512}
              onChange={(e) => {
                const nextValue = Number(e.target.value);
                setChunkSize(
                  Math.min(
                    Math.max(nextValue, minChunkSize),
                    Math.max(minChunkSize, maxChunkSize),
                  ),
                );
              }}
            />
            <span className="range-value">{chunkSize ?? 512}</span>
          </div>
        </div>
        <div className="control-group">
          <FieldLabel
            label="Overlap Size"
            help="Keep a shared slice of text between neighboring chunks to preserve context across chunk boundaries."
          />
          <div className="range-control">
            <input
              type="range"
              min="0"
              max="500"
              value={overlapSize ?? 0}
              onChange={(e) => setOverlapSize(Number(e.target.value))}
            />
            <span className="range-value">{overlapSize ?? 0}</span>
          </div>
        </div>
      </div>
      <TwoColumnLayout
        left={
          <div className="column-content">
            <h4 className="section-title">Source documents</h4>
            <div
              id="document-view"
              dangerouslySetInnerHTML={{
                __html:
                  documentPreview || "Loading document preview...",
              }}
            />
          </div>
        }
        right={
          <div className="column-content">
            <h4 className="section-title">Chunks</h4>
            <div id="chunks-table-container">
              {!filteredChunks.length ? (
                <p>No chunk data for this page.</p>
              ) : (
                <div className="chunk-card-container">
                  {filteredChunks.map((row) => (
                    <div
                      className="chunk-card"
                      key={`${row.page_index}-${row.chunk_index}`}
                    >
                      <span className="chunk-tag">
                        {" "}
                        Page {row.page_index}, Chunk {row.chunk_index}{" "}
                      </span>
                      <div className="chunk-text">
                        {row.chunk_text}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        }
      />
    </div>
  );
}

// Step 2: Embedding and indexing 
function EmbeddingStep() {
  const [embeddingsTable] = useModelState("embeddings_table");
  const [mongoDocsTable] = useModelState("mongo_docs_table");
  const [indexingStatus] = useModelState("indexing_status");
  const [currentDocIndex, setCurrentDocIndex] =
    useModelState("current_doc_index");
  const [, setCommand] = useModelState("command");
  const [embeddingModels] = useModelState("embedding_models");
  const [selectedEmbeddingIndex, setSelectedEmbeddingIndex] = useModelState("selected_embedding_index");
  const [selectedIndex] = useModelState("selected_index");
  const [vectorIndexDimensions, setVectorIndexDimensions] = useModelState("vector_index_dimensions");
  const [vectorIndexDimensionOptions] = useModelState("vector_index_dimension_options");
  const [embeddingOutputDtype, setEmbeddingOutputDtype] = useModelState("embedding_output_dtype");
  const [embeddingOutputDtypeOptions] = useModelState("embedding_output_dtype_options");
  const [configurationStatus] = useModelState("configuration_status");
  const [error] = useModelState("error");
  const [isLoading, setIsLoading] = React.useState(false);
  const indexingPhase = indexingStatus?.phase || "idle";

  // Names fed from Python side
  const [mongoDbName] = useModelState("mongo_db_name");
  const [mongoCollectionName] = useModelState("mongo_collection_name");

  // Ensure embeddingModels is always an array
  const safeEmbeddingModels = React.useMemo(() => {
    return Array.isArray(embeddingModels) ? embeddingModels : [];
  }, [embeddingModels]);
  const safeVectorIndexDimensionOptions = React.useMemo(() => {
    return Array.isArray(vectorIndexDimensionOptions)
      ? vectorIndexDimensionOptions
      : [256, 512, 1024, 2048];
  }, [vectorIndexDimensionOptions]);
  const safeEmbeddingOutputDtypeOptions = React.useMemo(() => {
    return Array.isArray(embeddingOutputDtypeOptions)
      ? embeddingOutputDtypeOptions
      : ["float", "int8", "uint8", "binary", "ubinary"];
  }, [embeddingOutputDtypeOptions]);
  const hasEmbeddingModels = safeEmbeddingModels.length > 0;
  const isConfigurationReady = configurationStatus?.phase === "completed";
  const configurationBlockReason = getConfigurationBlockReason(configurationStatus);

  const filteredChunks = React.useMemo(
    () =>
      (embeddingsTable || []).filter(
        (row) => row.page_index === Number(currentDocIndex || 0),
      ),
    [embeddingsTable, currentDocIndex],
  );

  const handlePrev = () => {
    const idx = Number(currentDocIndex || 0);
    if (idx > 0) setCurrentDocIndex(idx - 1);
  };
  const handleNext = () => {
    const idx = Number(currentDocIndex || 0);
    setCurrentDocIndex(idx + 1);
  };

  const handleLoadIntoMongo = () => {
    setIsLoading(true);
    setCommand("load_into_mongo");
  };

  React.useEffect(() => {
    const phase = indexingStatus?.phase || "idle";
    if (phase === "preparing" || phase === "purging" || phase === "purged" || phase === "indexing") {
      setIsLoading(true);
      return;
    }
    if (phase === "completed" || phase === "error" || error) {
      setIsLoading(false);
    }
  }, [indexingStatus, error]);

  const docs = mongoDocsTable || [];
  const visibleDocs =
    isLoading && ["idle", "preparing", "purging", "purged"].includes(indexingPhase)
      ? []
      : docs;
  const destinationLabel =
    mongoDbName && mongoCollectionName
      ? `${mongoDbName}.${mongoCollectionName}`
      : "database.collection";

  return (
    <div id="embedding-section">
      <div id="settings-section">
        {/* Page nav first in the row */}
        <div className="control-group">
          <FieldLabel
            label="Page Navigation"
            help="Move between pages to compare chunk text with the MongoDB documents created from it."
          />
          {/* Removed inline style, now relying on .control-group > div:has(button) */}
          <div>
            <button
              className="action-button"
              type="button"
              onClick={handlePrev}
            >
              Previous Page
            </button>
            <button
              className="action-button"
              type="button"
              onClick={handleNext}
            >
              Next Page
            </button>
          </div>
        </div>
        {/* Embedding model */}
        <div className="control-group">
          <FieldLabel
            label="Embedding Model"
            help="Pick the embedding model used to turn chunks into vectors before they are indexed in MongoDB."
          />
          <select
            value={selectedEmbeddingIndex ?? 0}
            onChange={(e) => setSelectedEmbeddingIndex(Number(e.target.value))}
            disabled={safeEmbeddingModels.length === 0}
          >
            {safeEmbeddingModels.length > 0 ? (
              safeEmbeddingModels.map((model, idx) => (
                <option key={idx} value={idx}>
                  {model}
                </option>
              ))
            ) : (
              <option value={0}>Loading models...</option>
            )}
          </select>
        </div>
        <div className="control-group embedding-index-name-group">
          <FieldLabel
            label="Vector Index Name"
            help="Shows the Atlas vector search index the playground will use for indexing and retrieval."
          />
          <input
            type="text"
            readOnly
            disabled
            value={selectedIndex || ""}
            placeholder="Provided by the widget"
          />
        </div>
        {/* Destination collection + button */}
        <div className="control-group embedding-destination-group">
          <FieldLabel
            label="Destination Collection"
            help="This is the MongoDB collection that will receive the embedded chunks when you load them."
          />
          <div>
            <input
              type="text"
              readOnly
              disabled
              value={destinationLabel}
            />
            <button
              id="loadButton"
              className="action-button"
              type="button"
              onClick={handleLoadIntoMongo}
              disabled={isLoading || !hasEmbeddingModels}
            >
              {isLoading ? (
                <>
                  Loading...
                </>
              ) : (
                "Load into MongoDB"
              )}
            </button>
          </div>
        </div>
        <div className="control-group embedding-dimensions-group">
          <FieldLabel
            label="Vector Dimensions"
            help="Choose the vector size used for the Atlas index and embeddings stored in MongoDB. Voyage supports smaller output dimensions thanks to Matryoshka Representation Learning (MRL)."
          />
          <select
            value={vectorIndexDimensions ?? 1024}
            onChange={(e) => setVectorIndexDimensions(Number(e.target.value))}
          >
            {safeVectorIndexDimensionOptions.map((dimension) => (
              <option key={dimension} value={dimension}>
                {dimension}
              </option>
            ))}
          </select>
        </div>
        <div className="control-group embedding-output-dtype-group">
          <FieldLabel
            label="Embedding Data Type"
            help="Choose how Voyage returns each vector value. float keeps full precision; int8, uint8, binary, and ubinary use quantized outputs to reduce storage and transfer size. Quantized output data types are enabled by Quantization-Aware Training (QAT)."
          />
          <select
            value={embeddingOutputDtype || "float"}
            onChange={(e) => setEmbeddingOutputDtype(e.target.value)}
          >
            {safeEmbeddingOutputDtypeOptions.map((outputDtype) => (
              <option key={outputDtype} value={outputDtype}>
                {outputDtype}
              </option>
            ))}
          </select>
        </div>
      </div>
      <TwoColumnLayout
        left={
          <div className="column-content">
            <h4 className="section-title">Chunks</h4>
            {!filteredChunks.length ? (
              <p>No chunks to display for this page.</p>
            ) : (
              <div className="chunk-card-container">
                {filteredChunks.map((row) => (
                  <div
                    className="chunk-card"
                    key={`${row.page_index}-${row.chunk_index}`}
                  >
                    <span className="chunk-tag">
                      {" "}
                      Page {row.page_index}, Chunk {row.chunk_index}{" "}
                    </span>
                    <div className="chunk-text">
                      {row.chunk_text}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        }
        right={
          <div className="column-content">
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <h4 className="section-title">
                Documents in MongoDB
              </h4>
            </div>
            <div id="doc-list-container">
              <ProgressStatusPanel
                status={indexingStatus}
                defaultTitle="Indexing status"
                completeTitle="Indexing complete"
                errorTitle="Indexing failed"
                defaultMessage="Preparing indexing..."
              />
              {!visibleDocs.length ? (
                <p>No documents loaded yet.</p>
              ) : (
                <div className="doc-card-container">
                  {visibleDocs.map((doc) => {
                    const embedding = Array.isArray(doc.embedding)
                      ? doc.embedding
                      : [];
                    const extraFields =
                      doc.extra_fields && typeof doc.extra_fields === "object"
                        ? Object.entries(doc.extra_fields)
                        : [];
                    const embPreview =
                      embedding.length > 5
                        ? [...embedding.slice(0, 5), "..."]
                        : embedding;
                    return (
                      <div className="doc-card" key={doc._id}>
                        <div>
                          <span className="doc-key">
                            _id:
                          </span>
                          <span className="doc-value doc-value-id">
                            {" "}
                            {doc._id}{" "}
                          </span>
                        </div>
                        <div>
                          <span className="doc-key">
                            text:
                          </span>
                          <span className="doc-value doc-value-text">
                            {" "}
                            {doc.text}{" "}
                          </span>
                        </div>
                        <div>
                          <span className="doc-key">
                            embedding:
                          </span>
                          <span className="doc-value doc-value-embedding">
                            {" "}
                            [{embPreview.join(", ")}]{" "}
                          </span>
                        </div>
                        {extraFields.map(([key, value]) => (
                          <div key={`${doc._id}-${key}`}>
                            <span className="doc-key">
                              {key}:
                            </span>
                            <span className="doc-value doc-value-text">
                              {" "}
                              {String(value)}{" "}
                            </span>
                          </div>
                        ))}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        }
      />
    </div>
  );
}

// UMAP Visualization Component with Plotly
// Shows all documents above threshold, with top K highlighted in green
function UMAPVisualization({ queryEmbedding, allDocEmbeddings, retrievedDocEmbeddings, topK }) {
  const plotContainerRef = React.useRef(null);
  const plotIdRef = React.useRef(`umap_plot_${Math.random().toString(36).substr(2, 9)}`);
  const [isLoading, setIsLoading] = React.useState(true);
  const isInitializedRef = React.useRef(false); // Track if plot has been created

  // Memoize data to avoid recomputation on unrelated changes
  const visualizationData = React.useMemo(() => {
    if (!queryEmbedding || queryEmbedding.length === 0) {
      return null;
    }

    // Use allDocEmbeddings which contains ALL docs above threshold (from backend)
    const docsToVisualize = allDocEmbeddings.length > 0 ? allDocEmbeddings : [];

    if (docsToVisualize.length === 0) {
      return null;
    }

    // Prepare data: chunks + query (query last for consistency with Python example)
    const X = [...docsToVisualize.map(d => d.embedding), queryEmbedding];
    const chunks = docsToVisualize.map(d => d.text || d.content || "");
    const docIds = docsToVisualize.map(d => d.id || d.result_id || d.text || d.content || "");

    // Use scores from backend if available, otherwise calculate
    const sims = docsToVisualize.map((doc) => {
      // Use pre-calculated score if available
      if (doc.score !== undefined) {
        return doc.score;
      }

      // Fallback: calculate cosine similarity
      const embedding = doc.embedding;
      if (!embedding || embedding.length !== queryEmbedding.length) return 0;

      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < embedding.length; i++) {
        dot += embedding[i] * queryEmbedding[i];
        na += embedding[i] * embedding[i];
        nb += queryEmbedding[i] * queryEmbedding[i];
      }
      const denom = Math.sqrt(na) * Math.sqrt(nb) + 1e-12;
      return dot / denom;
    });

    // Create a set of top K document texts for highlighting
    const topKIds = new Set(
      retrievedDocEmbeddings.slice(0, topK).map(d => d.id || d.result_id || d.text || d.content)
    );

    return { X, chunks, sims, nChunks: chunks.length, topKIds, docIds };
  }, [queryEmbedding, allDocEmbeddings, retrievedDocEmbeddings, topK]);

  React.useEffect(() => {
    // If already initialized, don't re-render
    if (isInitializedRef.current) {
      console.log("UMAPVisualization: Already initialized, skipping re-render");
      return;
    }

    console.log("UMAPVisualization effect triggered", {
      hasContainer: !!plotContainerRef.current,
      hasVisualizationData: !!visualizationData,
      topK
    });

    if (!plotContainerRef.current || !visualizationData) {
      console.log("UMAPVisualization: Missing requirements, not rendering");
      setIsLoading(false);
      return;
    }

    console.log("UMAPVisualization: Starting initialization with", visualizationData.nChunks, "documents");
    setIsLoading(true);

    const { X, chunks, sims, nChunks, topKIds, docIds } = visualizationData;

    // Configuration
    const n_neighbors = 10;
    const min_dist = 0.05;

    const plotId = plotIdRef.current;

    // Load dependencies and create visualization
    const initVisualization = async () => {
      try {
        // Load Plotly if not already loaded
        if (typeof window.Plotly === 'undefined') {
          console.log("UMAPVisualization: Loading Plotly from CDN");
          const script = document.createElement('script');
          script.src = 'https://cdn.plot.ly/plotly-2.30.0.min.js';
          script.async = true;
          document.head.appendChild(script);

          await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = reject;
          });
          console.log("UMAPVisualization: Plotly loaded");
        }

        const N = X.length;
        const queryIdx = nChunks;

        const cosineDistance = (a, b) => {
          let dot = 0, na = 0, nb = 0;
          for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
          }
          const denom = Math.sqrt(na) * Math.sqrt(nb) + 1e-12;
          return 1 - (dot / denom);
        };

        // UMAP-JS requirement: nNeighbors < N
        const requestedNN = n_neighbors;
        const nn = Math.max(2, Math.min(requestedNN, N - 1));

        // Create a seeded RNG for deterministic UMAP results
        // Using a simple Linear Congruential Generator (LCG)
        let seed = 42; // Fixed seed for reproducibility
        const seededRandom = () => {
          seed = (seed * 9301 + 49297) % 233280;
          return seed / 233280;
        };

        console.log("UMAPVisualization: Computing UMAP projection with", N, "points (seeded RNG)");
        const umap = new UMAP({
          nNeighbors: nn,
          minDist: min_dist,
          nComponents: 2,
          distanceFn: cosineDistance,
          random: seededRandom,
        });

        const Y = umap.fit(X);
        console.log("UMAPVisualization: UMAP projection complete");

        const chunkXY = Y.slice(0, nChunks);
        const queryXY = Y[queryIdx];

        // Identify which chunks are in top K (green points with edges)
        const topKIndices = [];
        docIds.forEach((docId, i) => {
          if (topKIds.has(docId)) {
            topKIndices.push(i);
          }
        });

        // Split chunks into top K (green) and rest (gray)
        const topKChunks = {
          x: [],
          y: [],
          text: [],
          indices: []
        };
        const otherChunks = {
          x: [],
          y: [],
          text: []
        };

        chunkXY.forEach(([x, y], i) => {
          const isTopK = topKIndices.includes(i);
          if (isTopK) {
            topKChunks.x.push(x);
            topKChunks.y.push(y);
            topKChunks.text.push(`#${i} sim=${sims[i].toFixed(3)}<br>${chunks[i]}`);
            topKChunks.indices.push(i);
          } else {
            otherChunks.x.push(x);
            otherChunks.y.push(y);
            otherChunks.text.push(`#${i} sim=${sims[i].toFixed(3)}<br>${chunks[i]}`);
          }
        });

        // Trace for non-top-K chunks (blue)
        const otherChunksTrace = {
          type: "scatter",
          mode: "markers",
          name: "Other Chunks",
          x: otherChunks.x,
          y: otherChunks.y,
          text: otherChunks.text,
          hoverinfo: "text",
          marker: { size: 8, color: "#3b82f6", line: { width: 1, color: "white" }, opacity: 0.7 },
        };

        // Trace for top K chunks (green)
        const topKChunksTrace = {
          type: "scatter",
          mode: "markers",
          name: "Top K Retrieved",
          x: topKChunks.x,
          y: topKChunks.y,
          text: topKChunks.text,
          hoverinfo: "text",
          marker: { size: 11, color: "#00ed64", line: { width: 1, color: "white" }, opacity: 0.85 },
        };

        const queryTrace = {
          type: "scatter",
          mode: "markers",
          name: "Query",
          x: [queryXY[0]],
          y: [queryXY[1]],
          text: [`Query Embedding`],
          hoverinfo: "text",
          marker: { size: 18, color: "#ff6b6b", line: { width: 1.5, color: "white" } },
        };

        // Draw edges from query to top K chunks
        const edgeTraces = topKChunks.indices.map(i => ({
          type: "scatter",
          mode: "lines",
          showlegend: false,
          x: [queryXY[0], chunkXY[i][0]],
          y: [queryXY[1], chunkXY[i][1]],
          line: { width: 2, color: "#00ed64", dash: "dot" },
          hoverinfo: "skip",
          opacity: 0.7,
        }));

        const layout = {
          title: {
            text: "UMAP visualization of the embedding vector space",
            font: { size: 14, family: "'Euclid Circular A', 'Helvetica Neue', Helvetica, Arial, sans-serif" },
            x: 0.5,
            xanchor: "center",
          },
          margin: { l: 40, r: 20, t: 40, b: 40 },
          xaxis: { zeroline: false, gridcolor: "rgba(0,0,0,0.08)", showticklabels: false },
          yaxis: { zeroline: false, gridcolor: "rgba(0,0,0,0.08)", showticklabels: false },
          legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.15 },
          height: 300,
        };

        const plotDiv = document.getElementById(plotId);
        if (plotDiv && window.Plotly) {
          console.log("UMAPVisualization: Creating plot with Plotly");
          // Order: edges first (background), then other chunks, then top K chunks, then query (foreground)
          const traces = [...edgeTraces, otherChunksTrace, topKChunksTrace, queryTrace];
          await window.Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
          console.log("UMAPVisualization: Plot created successfully");
          isInitializedRef.current = true; // Mark as initialized
          setIsLoading(false);
        } else {
          console.error("UMAPVisualization: Plot div or Plotly not found");
          setIsLoading(false);
        }
      } catch (error) {
        console.error("Error creating UMAP visualization:", error);
        setIsLoading(false);
        const plotDiv = document.getElementById(plotId);
        if (plotDiv) {
          plotDiv.innerHTML = `<div style="padding: 20px; text-align: center; color: #721c24; background: #f8d7da; border-radius: 8px;">
            Error loading visualization: ${error.message}
          </div>`;
        }
      }
    };

    initVisualization();

  }, [visualizationData, topK]);

  return (
    <div className="umap-visualization" style={{ marginBottom: "1rem", width: "100%", position: "relative", height: "300px" }}>
      {isLoading && (
        <div style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          padding: "20px",
          background: "rgba(255, 255, 255, 0.9)",
          borderRadius: "8px",
          zIndex: 10
        }}>
          Loading visualization...
        </div>
      )}
      <div
        id={plotIdRef.current}
        ref={plotContainerRef}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}

function HybridSearchBreakdown({ results }) {
  const rows = (results || []).slice(0, 10);

  if (!rows.length) {
    return <p>No hybrid results yet.</p>;
  }

  const maxScore = rows.reduce((maxValue, row) => {
    const total = Number(row.total_score ?? row.score ?? 0);
    return Math.max(maxValue, total);
  }, 0) || 1;

  return (
    <div className="hybrid-breakdown">
      {rows.map((row, idx) => {
        const vectorScore = Number(row.vector_score ?? 0);
        const searchScore = Number(row.search_score ?? 0);
        const totalScore = Number(row.total_score ?? row.score ?? 0);
        const totalWidth = Math.max(8, (totalScore / maxScore) * 100);
        const vectorWidth = totalScore > 0 ? (vectorScore / totalScore) * 100 : 0;
        const searchWidth = totalScore > 0 ? (searchScore / totalScore) * 100 : 0;

        return (
          <div className="hybrid-breakdown-row" key={row.result_id || idx}>
            <div className="hybrid-breakdown-header">
              <span className="hybrid-breakdown-rank">#{idx + 1}</span>
              <span className="hybrid-breakdown-score">Total {totalScore.toFixed(4)}</span>
            </div>
            <div className="hybrid-breakdown-bar-wrap">
              <div
                className="hybrid-breakdown-bar"
                style={{ width: `${totalWidth}%` }}
              >
                <div
                  className="hybrid-breakdown-bar-vector"
                  style={{ width: `${vectorWidth}%` }}
                  title={`Vector score: ${vectorScore.toFixed(4)}`}
                />
                <div
                  className="hybrid-breakdown-bar-search"
                  style={{ width: `${searchWidth}%` }}
                  title={`Search score: ${searchScore.toFixed(4)}`}
                />
              </div>
            </div>
            <div className="hybrid-breakdown-meta">
              <span>Vector {vectorScore.toFixed(4)}</span>
              <span>Search {searchScore.toFixed(4)}</span>
            </div>
          </div>
        );
      })}
      <div className="hybrid-breakdown-legend">
        <span className="hybrid-breakdown-legend-item">
          <span className="hybrid-breakdown-legend-swatch hybrid-breakdown-legend-vector" />
          Vector score
        </span>
        <span className="hybrid-breakdown-legend-item">
          <span className="hybrid-breakdown-legend-swatch hybrid-breakdown-legend-search" />
          Text score
        </span>
      </div>
    </div>
  );
}

function RetrievalVisualizationOnly() {
  const [ragResults] = useModelState("rag_results");
  const [queryEmbedding] = useModelState("query_embedding");
  const [allDocEmbeddings] = useModelState("all_doc_embeddings");
  const [topK] = useModelState("top_k");
  const [searchSubTab] = useModelState("search_sub_tab");

  const results = ragResults || [];
  const allDocs = allDocEmbeddings || [];
  const hybridEnabled = Number(searchSubTab || 1) === 2;

  const retrievedDocEmbeddings = results
    .filter(r => r.embedding && r.embedding.length > 0)
    .map(r => ({
      id: r.result_id,
      result_id: r.result_id,
      embedding: r.embedding,
      content: r.text,
      score: r.score,
      text: r.text?.substring(0, 50) + "..." || "Document"
    }));

  const showVisualization = queryEmbedding &&
    queryEmbedding.length > 0 &&
    allDocs.length > 0;

  if (!results.length) {
    return <p>No visualization yet. Run a query first.</p>;
  }

  if (hybridEnabled) {
    return <HybridSearchBreakdown results={results} />;
  }

  if (!showVisualization) {
    return <p>Visualization unavailable for the current results.</p>;
  }

  return (
    <UMAPVisualization
      queryEmbedding={queryEmbedding}
      allDocEmbeddings={allDocs}
      retrievedDocEmbeddings={retrievedDocEmbeddings}
      topK={topK ?? 5}
    />
  );
}

function RetrievedDocumentsList() {
  const [ragResults] = useModelState("rag_results");
  const results = ragResults || [];

  if (!results.length) {
    return (
      <div id="doc-list-container">
        <p>No retrieved documents yet.</p>
      </div>
    );
  }

  return (
    <div id="doc-list-container">
      <div className="doc-card-container">
        {results.map((r, idx) => {
          const score =
            typeof r.score === "number"
              ? r.score.toFixed(4)
              : String(r.score ?? "");
          const vectorScore =
            typeof r.vector_score === "number"
              ? r.vector_score.toFixed(4)
              : null;
          const searchScore =
            typeof r.search_score === "number"
              ? r.search_score.toFixed(4)
              : null;
          const totalScore =
            typeof r.total_score === "number"
              ? r.total_score.toFixed(4)
              : null;
          const docId =
            (r.metadata && (r.metadata._id || r.metadata.id)) ||
            `Doc ${idx + 1}`;
          // Format embedding preview like in Indexing section
          const embedding = Array.isArray(r.embedding) ? r.embedding : [];
          const embPreview = embedding.length > 0
            ? JSON.stringify(embedding)
            : "N/A";
          return (
            <div
              className="doc-card"
              style={{ position: "relative" }}
              key={idx}
            >
              <span
                className="doc-tag"
                style={{
                  position: "absolute",
                  top: "0.5rem",
                  right: "0.5rem",
                }}
              >
                {vectorScore !== null && searchScore !== null && totalScore !== null
                  ? `V: ${vectorScore} | S: ${searchScore} | T: ${totalScore}`
                  : `Score: ${score}`}
              </span>
              <div>
                <span className="doc-key">_id:</span>
                <span className="doc-value doc-value-id">
                  {docId}
                </span>
              </div>
              <div>
                <span className="doc-key">text:</span>
                <span className="doc-value doc-value-text">
                  {" "}
                  {r.text}{" "}
                </span>
              </div>
              <div>
                <span className="doc-key">embedding:</span>
                <span className="doc-value doc-value-embedding">
                  {embPreview}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RerankCard({ item, variant = "source", registerRef, onMouseEnter, onMouseLeave }) {
  const scoreLabel =
    variant === "reranked"
      ? item.relevance_score ?? item.score ?? null
      : item.total_score ?? item.score ?? null;
  const badgePrefix = variant === "reranked" ? "Rerank" : "Search";

  return (
    <div
      className={`chunk-card rerank-card ${variant === "reranked" ? "rerank-card-target" : "rerank-card-source"}`}
      ref={(node) => registerRef(item.result_id, node)}
      data-result-id={item.result_id}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="rerank-badge-row">
        <span className="chunk-tag">
          {badgePrefix} #{item.rank ?? "?"}
        </span>
        {scoreLabel !== null && scoreLabel !== undefined && (
          <span className="rerank-score-pill">
            {variant === "reranked" ? "Relevance" : "Score"}: {Number(scoreLabel).toFixed(4)}
          </span>
        )}
      </div>
      <div className="rerank-meta-row">
        {item.original_rank ? (
          <span className="rerank-rank-label">
            From search #{item.original_rank}
          </span>
        ) : (
          <span className="rerank-rank-label">
            Original rank #{item.rank ?? "?"}
          </span>
        )}
      </div>
      <div className="chunk-text">{item.text}</div>
    </div>
  );
}

function RerankingFlow({ sourceItems, rerankedItems, enabled }) {
  const wrapperRef = React.useRef(null);
  const leftRefs = React.useRef(new Map());
  const rightRefs = React.useRef(new Map());
  const [lines, setLines] = React.useState([]);
  const [hoveredResultId, setHoveredResultId] = React.useState(null);

  const registerLeftRef = React.useCallback((id, node) => {
    if (!id) return;
    if (node) leftRefs.current.set(id, node);
    else leftRefs.current.delete(id);
  }, []);

  const registerRightRef = React.useCallback((id, node) => {
    if (!id) return;
    if (node) rightRefs.current.set(id, node);
    else rightRefs.current.delete(id);
  }, []);

  React.useLayoutEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper || !enabled || !rerankedItems.length) {
      setLines([]);
      return;
    }

    const updateLines = () => {
      const wrapperRect = wrapper.getBoundingClientRect();
      const nextLines = rerankedItems
        .map((item) => {
          const leftNode = leftRefs.current.get(item.result_id);
          const rightNode = rightRefs.current.get(item.result_id);
          if (!leftNode || !rightNode) return null;

          const leftRect = leftNode.getBoundingClientRect();
          const rightRect = rightNode.getBoundingClientRect();

          return {
            id: item.result_id,
            x1: leftRect.right - wrapperRect.left,
            y1: leftRect.top - wrapperRect.top + leftRect.height / 2,
            x2: rightRect.left - wrapperRect.left,
            y2: rightRect.top - wrapperRect.top + rightRect.height / 2,
          };
        })
        .filter(Boolean);

      setLines(nextLines);
    };

    updateLines();
    window.addEventListener("resize", updateLines);
    return () => window.removeEventListener("resize", updateLines);
  }, [enabled, rerankedItems, sourceItems]);

  return (
    <div className="rerank-flow-wrapper" ref={wrapperRef}>
      <svg className="rerank-connector-layer" aria-hidden="true">
        <defs>
          <marker
            id="rerank-line-arrowhead"
            markerWidth="8"
            markerHeight="8"
            refX="7"
            refY="4"
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" fill="currentColor" />
          </marker>
        </defs>
        {lines.map((line) => (
          <line
            key={line.id}
            className={
              hoveredResultId
                ? line.id === hoveredResultId
                  ? "rerank-line-active"
                  : "rerank-line-dimmed"
                : ""
            }
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
            markerEnd="url(#rerank-line-arrowhead)"
          />
        ))}
      </svg>
      <TwoColumnLayout
        left={
          <div className="column-content">
            <h4 className="section-title">Retrieved Chunks</h4>
            <div className="rerank-card-list">
              {!sourceItems.length ? (
                <p>No retrieved chunks yet. Run retrieval first.</p>
              ) : (
                sourceItems.map((item) => (
                  <RerankCard
                    key={item.result_id}
                    item={item}
                    registerRef={registerLeftRef}
                    onMouseEnter={() => setHoveredResultId(item.result_id)}
                    onMouseLeave={() => setHoveredResultId(null)}
                  />
                ))
              )}
            </div>
          </div>
        }
        right={
          <div className="column-content">
            <h4 className="section-title">Reranked Chunks</h4>
            <div className="rerank-card-list">
              {!enabled ? (
                <p>Enable reranking to compare the new order.</p>
              ) : !rerankedItems.length ? (
                <p>Run reranking to see the reordered chunks.</p>
              ) : (
                rerankedItems.map((item) => (
                  <RerankCard
                    key={item.result_id}
                    item={item}
                    variant="reranked"
                    registerRef={registerRightRef}
                    onMouseEnter={() => setHoveredResultId(item.result_id)}
                    onMouseLeave={() => setHoveredResultId(null)}
                  />
                ))
              )}
            </div>
          </div>
        }
      />
    </div>
  );
}

function BenchmarkExplainCard({
  item,
  variant = "source",
  registerRef,
  isExpected = false,
  onMouseEnter,
  onMouseLeave,
}) {
  const scoreLabel =
    variant === "reranked"
      ? item.relevance_score ?? item.score ?? null
      : item.total_score ?? item.score ?? null;
  const badgePrefix = variant === "reranked" ? "Rerank" : "Search";

  return (
    <div
      className={`chunk-card rerank-card benchmark-explain-card ${variant === "reranked" ? "rerank-card-target" : "rerank-card-source"}${isExpected ? " benchmark-expected-card" : ""}`}
      ref={(node) => registerRef(item.result_id, node)}
      data-result-id={item.result_id}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="benchmark-card-badges">
        <span className="chunk-tag">
          {badgePrefix} #{item.rank ?? "?"}
        </span>
        {scoreLabel !== null && scoreLabel !== undefined && (
          <span className="rerank-score-pill">
            {variant === "reranked" ? "Relevance" : "Score"}: {Number(scoreLabel).toFixed(4)}
          </span>
        )}
      </div>
      <div className="rerank-meta-row benchmark-meta-row">
        {item.original_rank ? (
          <span className="rerank-rank-label">
            From search #{item.original_rank}
          </span>
        ) : (
          <span className="rerank-rank-label">
            Chunk ID: {item.result_id ?? "N/A"}
          </span>
        )}
      </div>
      <div className="chunk-text">{item.text}</div>
    </div>
  );
}

function BenchmarkExplainFlow({ result, onBack }) {
  const wrapperRef = React.useRef(null);
  const leftRefs = React.useRef(new Map());
  const rightRefs = React.useRef(new Map());
  const [lines, setLines] = React.useState([]);
  const [hoveredResultId, setHoveredResultId] = React.useState(null);

  const retrievalItems = result?.retrieval_results || [];
  const rerankedItems = result?.rerank_results || [];
  const expectedChunkIds = React.useMemo(
    () => new Set(result?.expected_chunk_ids || []),
    [result],
  );

  const registerLeftRef = React.useCallback((id, node) => {
    if (!id) return;
    if (node) leftRefs.current.set(id, node);
    else leftRefs.current.delete(id);
  }, []);

  const registerRightRef = React.useCallback((id, node) => {
    if (!id) return;
    if (node) rightRefs.current.set(id, node);
    else rightRefs.current.delete(id);
  }, []);

  React.useLayoutEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper || !rerankedItems.length) {
      setLines([]);
      return;
    }

    const updateLines = () => {
      const wrapperRect = wrapper.getBoundingClientRect();
      const nextLines = rerankedItems
        .map((item) => {
          const leftNode = leftRefs.current.get(item.result_id);
          const rightNode = rightRefs.current.get(item.result_id);
          if (!leftNode || !rightNode) return null;

          const leftRect = leftNode.getBoundingClientRect();
          const rightRect = rightNode.getBoundingClientRect();

          return {
            id: item.result_id,
            moved:
              item.original_rank !== undefined &&
              item.original_rank !== null &&
              item.rank !== undefined &&
              item.original_rank !== item.rank,
            x1: leftRect.right - wrapperRect.left,
            y1: leftRect.top - wrapperRect.top + leftRect.height / 2,
            x2: rightRect.left - wrapperRect.left,
            y2: rightRect.top - wrapperRect.top + rightRect.height / 2,
          };
        })
        .filter(Boolean);

      setLines(nextLines);
    };

    updateLines();
    window.addEventListener("resize", updateLines);
    return () => window.removeEventListener("resize", updateLines);
  }, [rerankedItems, retrievalItems]);

  return (
    <div className="benchmark-explain-view">
      <div className="benchmark-detail-header">
        <button
          className="action-button benchmark-back-button"
          type="button"
          onClick={onBack}
        >
          ← Back
        </button>
        <div className="benchmark-detail-title-group">
          <div className="benchmark-detail-label">Benchmark Explain</div>
          <h3 className="benchmark-detail-question">{result?.question}</h3>
        </div>
      </div>

      <div className="rerank-flow-wrapper benchmark-explain-flow" ref={wrapperRef}>
        <svg className="rerank-connector-layer" aria-hidden="true">
          <defs>
            <marker
              id="benchmark-rerank-line-arrowhead"
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M 0 0 L 8 4 L 0 8 z" fill="currentColor" />
            </marker>
          </defs>
          {lines.map((line) => (
            <line
              key={line.id}
              className={[
                line.moved ? "rerank-line-moved" : "rerank-line-static",
                hoveredResultId
                  ? line.id === hoveredResultId
                    ? "rerank-line-active"
                    : "rerank-line-dimmed"
                  : "",
              ].filter(Boolean).join(" ")}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              markerEnd="url(#benchmark-rerank-line-arrowhead)"
            />
          ))}
        </svg>
        <TwoColumnLayout
          left={
            <div className="column-content">
              <h4 className="section-title">Retrieved Chunks</h4>
              <div className="rerank-card-list">
                {!retrievalItems.length ? (
                  <p>No retrieved chunks found for this benchmark query.</p>
                ) : (
                  retrievalItems.map((item) => (
                    <BenchmarkExplainCard
                      key={item.result_id}
                      item={item}
                      registerRef={registerLeftRef}
                      isExpected={expectedChunkIds.has(item.result_id)}
                      onMouseEnter={() => setHoveredResultId(item.result_id)}
                      onMouseLeave={() => setHoveredResultId(null)}
                    />
                  ))
                )}
              </div>
            </div>
          }
          right={
            <div className="column-content">
              <h4 className="section-title">Reranked Chunks</h4>
              <div className="rerank-card-list">
                {!rerankedItems.length ? (
                  <p>No reranked chunks available for this benchmark query.</p>
                ) : (
                  rerankedItems.map((item) => (
                    <BenchmarkExplainCard
                      key={item.result_id}
                      item={item}
                      variant="reranked"
                      registerRef={registerRightRef}
                      isExpected={expectedChunkIds.has(item.result_id)}
                      onMouseEnter={() => setHoveredResultId(item.result_id)}
                      onMouseLeave={() => setHoveredResultId(null)}
                    />
                  ))
                )}
              </div>
            </div>
          }
        />
      </div>
    </div>
  );
}

function QueryStep() {
  const [, setCommand] = useModelState("command");
  const [searchSubTab, setSearchSubTab] = useModelState("search_sub_tab");
  const [ragQuery, setRagQuery] = useModelState("rag_query");
  const [ragResults, setRagResults] = useModelState("rag_results");
  const [error] = useModelState("error");
  const [queryEmbedding, setQueryEmbedding] = useModelState("query_embedding");
  const [, setAllDocEmbeddings] = useModelState("all_doc_embeddings");
  const [, setRerankResults] = useModelState("rerank_results");
  const [retrievalStatus] = useModelState("retrieval_status");
  const [isLoading, setIsLoading] = React.useState(false);

  const [topK, setTopK] = useModelState("top_k");
  const [scoreThreshold, setScoreThreshold] = useModelState("score_threshold");
  const [fulltextPenalty, setFulltextPenalty] = useModelState("fulltext_penalty");
  const [vectorPenalty, setVectorPenalty] = useModelState("vector_penalty");
  const [rerankingEnabled] = useModelState("reranking_enabled");
  const [constraints] = useModelState("constraints");
  const [configurationStatus] = useModelState("configuration_status");

  const [embeddingModels] = useModelState("embedding_models");
  const [queryEmbeddingIndex, setQueryEmbeddingIndex] = useModelState("query_embedding_index");
  const safeEmbeddingModels = React.useMemo(() => {
    return Array.isArray(embeddingModels) ? embeddingModels : [];
  }, [embeddingModels]);
  const hasEmbeddingModels = safeEmbeddingModels.length > 0;
  const isConfigurationReady = configurationStatus?.phase === "completed";
  const configurationBlockReason = getConfigurationBlockReason(configurationStatus);
  const { minK, maxK } = getConstraintKBounds(constraints);
  const hasFixedK = minK != null && maxK != null && minK === maxK;
  const topKMin = minK ?? 1;
  const topKMax = maxK ?? null;
  const topKDisabled = hasFixedK;

  React.useEffect(() => {
    const phase = retrievalStatus?.phase || "idle";
    if (phase === "running") {
      setIsLoading(true);
      return;
    }
    if (phase === "completed" || phase === "error" || error) {
      setIsLoading(false);
    }
  }, [retrievalStatus, error]);

  React.useEffect(() => {
    const clampedTopK = clampInteger(topK, topKMin, topKMax);
    if (clampedTopK !== topK) {
      setTopK(clampedTopK);
    }
  }, [topK, topKMin, topKMax, setTopK]);

  const handleAsk = (isHybrid) => {
    const mainQuery = (ragQuery || "").trim();
    if (!mainQuery) return;

    // BASIC VALIDATION/CLEANUP
    const kValue = Number(topK);
    if (isNaN(kValue) || kValue < topKMin || (topKMax != null && kValue > topKMax)) {
      setTopK(clampInteger(topK, topKMin, topKMax));
    }

    // scoreThreshold can be null/empty string, which is fine, but if it's a number, validate it.
    if (scoreThreshold !== null && scoreThreshold !== undefined && scoreThreshold !== "") {
      const thresholdValue = Number(scoreThreshold);
      if (isNaN(thresholdValue) || thresholdValue < 0 || thresholdValue > 1) {
        console.error("Invalid Score Threshold value. Ignoring for command.");
      }
    }

    if (isHybrid) {
      const fp = Number(fulltextPenalty);
      const vp = Number(vectorPenalty);
      if (isNaN(fp) || fp < 0 || fp > 100) setFulltextPenalty(50);
      if (isNaN(vp) || vp < 0 || vp > 100) setVectorPenalty(50);
    }
    
    setRagResults([]);
    setQueryEmbedding([]);
    setAllDocEmbeddings([]);
    setRerankResults([]);

    setIsLoading(true);
    setCommand(isHybrid ? "hybrid_search" : "vector_search");
    setSearchSubTab(isHybrid ? 2 : 1);
  };

  const hybridEnabled = Number(searchSubTab || 1) === 2;

  const RetrievalSettings = (
    <div id="settings-section">
      <div className="control-group search-query-group">
        <FieldLabel
          label="Query"
          help="Enter the search question you want to run against the indexed chunks."
        />
        <input
          type="text"
          value={ragQuery || ""}
          onChange={(e) => setRagQuery(e.target.value)}
          placeholder="Ask your question here..."
        />
      </div>

      <div className="control-group search-query-embedding-group">
        <FieldLabel
          label="Query Embedding Model"
          help="Choose which embedding model converts your query into a vector at retrieval time."
        />
        <select
          value={queryEmbeddingIndex ?? 0}
          onChange={(e) => setQueryEmbeddingIndex(Number(e.target.value))}
          disabled={!hasEmbeddingModels}
        >
          {hasEmbeddingModels ? (
            safeEmbeddingModels.map((model, idx) => (
              <option key={idx} value={idx}>
                {model}
              </option>
            ))
          ) : (
            <option value={0}>Loading models...</option>
          )}
        </select>
      </div>

      <div className="control-group search-top-k-group">
        <FieldLabel
          label="Top K"
          help="Return the top K highest-scoring chunks from vector or hybrid search."
        />
        <input
          type="number"
          min={topKMin}
          max={topKMax ?? undefined}
          step="1"
          value={topK ?? minK ?? 5}
          onChange={(e) => {
            setTopK(clampInteger(e.target.value, topKMin, topKMax));
          }}
          disabled={topKDisabled}
          placeholder="e.g., 5"
        />
      </div>

      <div className="control-group search-score-threshold-group">
        <FieldLabel
          label="Score Threshold"
          help="Optionally filter out low-similarity results before keeping the final Top K chunks."
        />
        <input
          type="number"
          min="0"
          max="1"
          step="0.01"
          value={scoreThreshold ?? ""}
          onChange={(e) => {
            const val = e.target.value;
            setScoreThreshold(val === "" ? null : Number(val));
          }}
          placeholder="e.g., 0.8 (Optional)"
        />
      </div>

      <div className="control-group search-hybrid-group">
        <FieldLabel
          label="Hybrid Search"
          help="Blend vector similarity with full-text search scoring so keyword matches can influence ranking."
        />
        <label className="inline-checkbox-label">
          <input
            type="checkbox"
            checked={hybridEnabled}
            onChange={(e) => setSearchSubTab(e.target.checked ? 2 : 1)}
          />
          Enable Hybrid Search
        </label>
      </div>

      {hybridEnabled ? (
        <div className="control-group search-fulltext-penalty-group">
          <FieldLabel
            label={`Fulltext Penalty (${fulltextPenalty ?? 50})`}
            help="Adjust how strongly the full-text portion influences the hybrid search score."
          />
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={fulltextPenalty ?? 50}
            onChange={(e) => setFulltextPenalty(Number(e.target.value))}
          />
        </div>
      ) : (
        <div className="control-group search-fulltext-penalty-group" />
      )}

      {hybridEnabled ? (
        <div className="control-group search-vector-penalty-group">
          <FieldLabel
            label={`Vector Penalty (${vectorPenalty ?? 50})`}
            help="Adjust how strongly the vector similarity portion influences the hybrid search score."
          />
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={vectorPenalty ?? 50}
            onChange={(e) => setVectorPenalty(Number(e.target.value))}
          />
        </div>
      ) : (
        <div className="control-group search-vector-penalty-group" />
      )}

      <div className="control-group search-action-group">
        <FieldLabel
          label="Action"
          help="Run the current retrieval configuration and populate the visualization and retrieved documents panels."
        />
        <div>
          <button
            className="action-button"
            type="button"
            onClick={() => handleAsk(hybridEnabled)}
            disabled={isLoading || !ragQuery?.trim() || !hasEmbeddingModels}
          >
            {isLoading ? "Searching..." : hybridEnabled ? "Run Hybrid Search" : "Run Search"}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div id="rag-section">
      {RetrievalSettings}

      <TwoColumnLayout
        align="stretch" // Changed from flex-start to stretch for uniform height
        left={
          <div className="column-content">
            <h4 className="section-title">Visualization</h4>
            <RetrievalVisualizationOnly />
          </div>
        }
        right={
          <div className="column-content">
            <h4 className="section-title">Retrieved Documents</h4>
            <RetrievedDocumentsList />
          </div>
        }
      />
    </div>
  );
}

function VectorSearchStep() {
  return <QueryStep />;
}

function RerankingStep() {
  const [rerankingEnabled, setRerankingEnabled] = useModelState("reranking_enabled");
  const [rerankerModels] = useModelState("reranker_models");
  const [selectedRerankerIndex, setSelectedRerankerIndex] = useModelState("selected_reranker_index");
  const [rerankTopK, setRerankTopK] = useModelState("rerank_top_k");
  const [topK, setTopK] = useModelState("top_k");
  const [constraints] = useModelState("constraints");
  const [ragResults] = useModelState("rag_results");
  const [rerankResults] = useModelState("rerank_results");
  const [error] = useModelState("error");
  const [, setCommand] = useModelState("command");
  const [isLoading, setIsLoading] = React.useState(false);
  const { minK, maxK } = getConstraintKBounds(constraints);
  const hasFixedK = minK != null && maxK != null && minK === maxK;
  const rerankTopKMin = minK ?? 1;
  const rerankTopKMax = Math.max(
    rerankTopKMin,
    Math.min(
      topK ?? rerankTopKMin,
      maxK ?? topK ?? rerankTopKMin,
      Math.max((ragResults || []).length, 1),
    ),
  );

  React.useEffect(() => {
    if (!isLoading) return;
    if ((rerankResults && rerankResults.length > 0) || error) {
      setIsLoading(false);
    }
  }, [rerankResults, error, isLoading]);

  React.useEffect(() => {
    if (!rerankingEnabled) return;
    const clampedTopK = clampInteger(topK, minK ?? 1, maxK ?? null);
    if (clampedTopK !== topK) {
      setTopK(clampedTopK);
      return;
    }
    const clampedRerankTopK = clampInteger(rerankTopK, rerankTopKMin, rerankTopKMax);
    if (clampedRerankTopK !== rerankTopK) {
      setRerankTopK(clampedRerankTopK);
    }
  }, [
    rerankingEnabled,
    minK,
    maxK,
    topK,
    rerankTopK,
    rerankTopKMin,
    rerankTopKMax,
    setTopK,
    setRerankTopK,
  ]);

  const handleRunReranking = () => {
    if (!rerankingEnabled) return;
    setIsLoading(true);
    setCommand("rerank_results");
  };

  return (
    <div id="reranking-section">
      <div id="settings-section">
        <div className="control-group reranking-toggle-group">
          <FieldLabel
            label="Reranking"
            help="Turn on a reranker to reorder retrieved chunks after the initial search step."
          />
          <label className="inline-checkbox-label">
            <input
              type="checkbox"
              checked={!!rerankingEnabled}
              onChange={(e) => setRerankingEnabled(e.target.checked)}
            />
            Enable Reranking
          </label>
        </div>

        {rerankingEnabled && (
          <>
            <div className="control-group">
              <FieldLabel
                label="Reranker Model"
                help="Choose the reranking model that scores retrieved chunks for final ordering."
              />
              <select
                value={selectedRerankerIndex ?? 0}
                onChange={(e) => setSelectedRerankerIndex(Number(e.target.value))}
                disabled={!rerankerModels || rerankerModels.length === 0}
              >
                {rerankerModels && rerankerModels.length > 0 ? (
                  rerankerModels.map((model, idx) => (
                    <option key={idx} value={idx}>
                      {model}
                    </option>
                  ))
                ) : (
                  <option value={0}>No rerankers provided</option>
                )}
              </select>
            </div>

            <div className="control-group">
              <FieldLabel
                label="Rerank Top K"
                help="Set how many retrieved chunks are passed into the reranker for reordering."
              />
              <input
                type="number"
                min={rerankTopKMin}
                max={rerankTopKMax}
                step="1"
                value={rerankTopK ?? rerankTopKMin}
                onChange={(e) => {
                  setRerankTopK(clampInteger(e.target.value, rerankTopKMin, rerankTopKMax));
                }}
                disabled={hasFixedK}
              />
            </div>

            <div className="control-group">
              <FieldLabel
                label="Action"
                help="Run reranking on the currently retrieved chunks using the selected reranker."
              />
              <div>
                <button
                  className="action-button"
                  type="button"
                  onClick={handleRunReranking}
                  disabled={
                    isLoading ||
                    !rerankingEnabled ||
                    !rerankerModels ||
                    rerankerModels.length === 0 ||
                    !(ragResults || []).length
                  }
                >
                  {isLoading ? "Reranking..." : "Run Reranking"}
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      <RerankingFlow
        sourceItems={ragResults || []}
        rerankedItems={rerankResults || []}
        enabled={!!rerankingEnabled}
      />
    </div>
  );
}

function formatBenchmarkScore(value) {
  return typeof value === "number" ? value.toFixed(1) : "N/A";
}

function formatUsd(value) {
  return typeof value === "number" ? `$${value.toFixed(6)}` : "N/A";
}

function BenchmarkSummary({
  summary,
  scoreboardAvailable,
  scoreboardLoading,
  scoreboardEntryName,
  setScoreboardEntryName,
  onPublish,
}) {
  const retrievalK = summary?.retrieval_k ?? summary?.top_k ?? null;
  const rerankK = summary?.rerank_k ?? null;
  const meanRetrievalLatencyMs = summary?.retrieval_mean_latency_ms;
  const meanRerankLatencyMs = summary?.rerank_mean_latency_ms;
  const meanTotalLatencyMs = summary?.total_mean_latency_ms;
  const meanRetrievalTokenCount = summary?.retrieval_mean_token_count;
  const meanRerankTokenCount = summary?.rerank_mean_token_count;
  const meanContextTokenCount = summary?.llm_context_mean_token_count;
  const embeddingQueryTokenMean = summary?.embedding_mean_token_count;
  const embeddingCorpusTokenCount = summary?.embedding_corpus_token_count;
  const embeddingQueryTotalTokenCount = summary?.embedding_total_token_count;
  const rerankTotalTokenCount = summary?.rerank_total_token_count;

  if (!summary || !Object.keys(summary).length) {
    return <p>No benchmark run yet.</p>;
  }

  return (
    <div className="doc-card benchmark-summary-card">
      <div className="benchmark-summary-grid">
        <div className="benchmark-summary-item benchmark-summary-item-wide">
          <div className="benchmark-summary-label">Queries</div>
          <div className="benchmark-summary-value">{summary.query_count ?? 0}</div>
        </div>

        <div className="benchmark-summary-item benchmark-summary-item-wide">
          <div className="benchmark-score-row">
            <div className="benchmark-score-block">
              <div className="benchmark-summary-label">Overall Score</div>
              <div className="benchmark-summary-value">{formatBenchmarkScore(summary.overall_score)} / 100</div>
              {summary.score_weights ? (
                <div className="benchmark-summary-label">
                  {`Weights A/L/C: ${Number(summary.score_weights.accuracy ?? 0).toFixed(2)} / ${Number(summary.score_weights.latency ?? 0).toFixed(2)} / ${Number(summary.score_weights.cost ?? 0).toFixed(2)}`}
                </div>
              ) : null}
            </div>
            {scoreboardAvailable ? (
              <div className="benchmark-publish-controls">
                <FieldLabel
                  label="Publish"
                  help="Save this benchmark score to the scoreboard with the name you enter here."
                />
                <div className="scoreboard-publish-row">
                  <input
                    className="scoreboard-name-input"
                    type="text"
                    value={scoreboardEntryName || ""}
                    onChange={(e) => setScoreboardEntryName(e.target.value)}
                    placeholder="your name"
                  />
                  <button
                    className="action-button scoreboard-publish-button"
                    type="button"
                    onClick={onPublish}
                    disabled={
                      scoreboardLoading ||
                      typeof summary?.overall_score !== "number" ||
                      !(scoreboardEntryName || "").trim()
                    }
                  >
                    {scoreboardLoading ? "Publishing..." : "Publish to Scoreboard"}
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>

        <div className="benchmark-summary-item benchmark-summary-item-wide benchmark-category-card">
          <div className="benchmark-category-title">Accuracy</div>
          <div className="benchmark-metric-pairs">
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell benchmark-metric-cell-wide">
                <div className="benchmark-summary-label">Accuracy Score</div>
                <div className="benchmark-summary-value">{formatBenchmarkScore(summary.accuracy_score)} / 100</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">{`Recall@${retrievalK ?? "k"}`}</div>
                <div className="benchmark-summary-value">{Number(summary.retrieval_mean_recall ?? 0).toFixed(4)}</div>
              </div>
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">{`Rerank Recall@${rerankK ?? "k"}`}</div>
                <div className="benchmark-summary-value">{summary.reranking_enabled ? Number(summary.rerank_mean_recall ?? 0).toFixed(4) : "N/A"}</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">{`nDCG@${retrievalK ?? "k"}`}</div>
                <div className="benchmark-summary-value">{Number(summary.retrieval_mean_ndcg ?? 0).toFixed(4)}</div>
              </div>
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">{`Rerank nDCG@${rerankK ?? "k"}`}</div>
                <div className="benchmark-summary-value">{summary.reranking_enabled ? Number(summary.rerank_mean_ndcg ?? 0).toFixed(4) : "N/A"}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="benchmark-summary-item benchmark-summary-item-wide benchmark-category-card">
          <div className="benchmark-category-title">Latency</div>
          <div className="benchmark-metric-pairs">
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell benchmark-metric-cell-wide">
                <div className="benchmark-summary-label">Latency Score</div>
                <div className="benchmark-summary-value">{formatBenchmarkScore(summary.latency_score)} / 100</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Avg Retrieval Latency</div>
                <div className="benchmark-summary-value">{typeof meanRetrievalLatencyMs === "number" ? `${meanRetrievalLatencyMs.toFixed(2)} ms` : "N/A"}</div>
              </div>
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Avg Rerank Latency</div>
                <div className="benchmark-summary-value">{summary.reranking_enabled && typeof meanRerankLatencyMs === "number" ? `${meanRerankLatencyMs.toFixed(2)} ms` : "N/A"}</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell benchmark-metric-cell-wide">
                <div className="benchmark-summary-label">Avg Total Latency</div>
                <div className="benchmark-summary-value">{typeof meanTotalLatencyMs === "number" ? `${meanTotalLatencyMs.toFixed(2)} ms` : "N/A"}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="benchmark-summary-item benchmark-summary-item-wide benchmark-category-card">
          <div className="benchmark-category-title">Cost</div>
          <div className="benchmark-metric-pairs">
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Cost Score</div>
                <div className="benchmark-summary-value">{formatBenchmarkScore(summary.cost_score)} / 100</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Context Tokens</div>
                <div className="benchmark-summary-value">{typeof meanContextTokenCount === "number" ? Math.round(meanContextTokenCount).toLocaleString() : "N/A"}</div>
              </div>
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Reranked Content Tokens</div>
                <div className="benchmark-summary-value">{summary.reranking_enabled && typeof meanRerankTokenCount === "number" ? Math.round(meanRerankTokenCount).toLocaleString() : "N/A"}</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Embedding Tokens</div>
                <div className="benchmark-summary-value">{typeof embeddingQueryTotalTokenCount === "number" ? embeddingQueryTotalTokenCount.toLocaleString() : "N/A"}</div>
              </div>
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Rerank Tokens</div>
                <div className="benchmark-summary-value">{summary.reranking_enabled && typeof rerankTotalTokenCount === "number" ? rerankTotalTokenCount.toLocaleString() : "N/A"}</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Avg Embedding Tokens</div>
                <div className="benchmark-summary-value">{typeof embeddingQueryTokenMean === "number" ? Math.round(embeddingQueryTokenMean).toLocaleString() : "N/A"}</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Embedding Cost</div>
                <div className="benchmark-summary-value">{formatUsd(summary.embedding_cost_usd)}</div>
              </div>
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Context LLM Cost</div>
                <div className="benchmark-summary-value">{formatUsd(summary.llm_context_cost_usd)}</div>
              </div>
            </div>
            <div className="benchmark-metric-pair-row">
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Rerank Cost</div>
                <div className="benchmark-summary-value">{summary.reranking_enabled ? formatUsd(summary.rerank_cost_usd) : "N/A"}</div>
              </div>
              <div className="benchmark-metric-cell">
                <div className="benchmark-summary-label">Total Cost</div>
                <div className="benchmark-summary-value">
                  {formatUsd(summary.total_cost_usd)}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BenchmarkResultsList({ results, onExplain }) {
  if (!results || !results.length) {
    return <p>No benchmark results yet.</p>;
  }

  return (
    <div className="doc-card-container">
      {results.map((row) => (
        <div key={row.qid} className="doc-card benchmark-result-card">
          <div className="benchmark-question">{row.question}</div>
          <div><span className="doc-key">Relevant Chunks:</span> <span className="doc-value">{row.relevant_chunk_count}</span></div>
          <div className="benchmark-category-card">
            <div className="benchmark-category-title">Accuracy</div>
            <div className="benchmark-result-metric-grid">
              <div className="benchmark-metric-pair-row">
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Retrieval Recall:</span> <span className="doc-value">{Number(row.retrieval_recall ?? 0).toFixed(4)}</span>
                </div>
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Rerank Recall:</span> <span className="doc-value">{row.rerank_recall == null ? "N/A" : Number(row.rerank_recall).toFixed(4)}</span>
                </div>
              </div>
              <div className="benchmark-metric-pair-row">
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Retrieval nDCG:</span> <span className="doc-value">{Number(row.retrieval_ndcg ?? 0).toFixed(4)}</span>
                </div>
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Rerank nDCG:</span> <span className="doc-value">{row.rerank_ndcg == null ? "N/A" : Number(row.rerank_ndcg).toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="benchmark-category-card">
            <div className="benchmark-category-title">Latency</div>
            <div className="benchmark-result-metric-grid">
              <div className="benchmark-metric-pair-row">
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Retrieval Latency:</span> <span className="doc-value">{typeof row.retrieval_latency_ms === "number" ? `${row.retrieval_latency_ms.toFixed(2)} ms` : "N/A"}</span>
                </div>
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Rerank Latency:</span> <span className="doc-value">{typeof row.rerank_latency_ms === "number" ? `${row.rerank_latency_ms.toFixed(2)} ms` : "N/A"}</span>
                </div>
              </div>
              <div className="benchmark-metric-pair-row">
                <div className="benchmark-metric-cell benchmark-metric-cell-wide">
                  <span className="doc-key">Total Latency:</span> <span className="doc-value">{typeof row.total_latency_ms === "number" ? `${row.total_latency_ms.toFixed(2)} ms` : "N/A"}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="benchmark-category-card">
            <div className="benchmark-category-title">Cost</div>
            <div className="benchmark-result-metric-grid">
              <div className="benchmark-metric-pair-row">
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Content Tokens:</span> <span className="doc-value">{typeof row.retrieval_token_count === "number" ? row.retrieval_token_count.toLocaleString() : "N/A"}</span>
                </div>
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Reranked Content Tokens:</span> <span className="doc-value">{typeof row.rerank_token_count === "number" ? row.rerank_token_count.toLocaleString() : "N/A"}</span>
                </div>
              </div>
              <div className="benchmark-metric-pair-row">
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Embedding Tokens:</span> <span className="doc-value">{typeof row.embedding_model_token_count === "number" ? row.embedding_model_token_count.toLocaleString() : "N/A"}</span>
                </div>
                <div className="benchmark-metric-cell">
                  <span className="doc-key">Rerank Tokens:</span> <span className="doc-value">{typeof row.rerank_model_token_count === "number" ? row.rerank_model_token_count.toLocaleString() : "N/A"}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="benchmark-actions-row">
            <button
              className="action-button"
              type="button"
              onClick={() => onExplain(row)}
            >
              Explain
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function BenchmarkStep() {
  const [benchmarkDatasetLoaded] = useModelState("benchmark_dataset_loaded");
  const [benchmarkDatasetName] = useModelState("benchmark_dataset_name");
  const [benchmarkSummary] = useModelState("benchmark_summary");
  const [benchmarkResults] = useModelState("benchmark_results");
  const [benchmarkRunning] = useModelState("benchmark_running");
  const [benchmarkStatus] = useModelState("benchmark_status");
  const [configurationStatus] = useModelState("configuration_status");
  const [scoreboardAvailable] = useModelState("scoreboard_available");
  const [scoreboardLoading] = useModelState("scoreboard_loading");
  const [scoreboardEntryName, setScoreboardEntryName] = useModelState("scoreboard_entry_name");
  const [, setCommand] = useModelState("command");
  const [selectedBenchmarkQid, setSelectedBenchmarkQid] = React.useState(null);

  const selectedBenchmarkResult = React.useMemo(
    () => (benchmarkResults || []).find((row) => row.qid === selectedBenchmarkQid) || null,
    [benchmarkResults, selectedBenchmarkQid],
  );

  React.useEffect(() => {
    if (!selectedBenchmarkQid) return;
    const stillExists = (benchmarkResults || []).some((row) => row.qid === selectedBenchmarkQid);
    if (!stillExists) {
      setSelectedBenchmarkQid(null);
    }
  }, [benchmarkResults, selectedBenchmarkQid]);

  const handleExplain = (row) => {
    setSelectedBenchmarkQid(row.qid);
  };
  const isConfigurationReady = configurationStatus?.phase === "completed";
  const configurationBlockReason = getConfigurationBlockReason(configurationStatus);

  return (
    <div id="benchmark-section">
      <div id="settings-section">
        <div className="control-group">
          <FieldLabel
            label="Dataset"
            help="Shows which benchmark dataset is loaded for evaluating retrieval and reranking quality."
          />
          <input
            type="text"
            readOnly
            disabled
            value={benchmarkDatasetLoaded ? (benchmarkDatasetName || "benchmark_dataset") : "No benchmark dataset loaded"}
          />
        </div>
        <div className="control-group">
          <FieldLabel
            label="Run"
            help="Execute the benchmark over all dataset questions using the current playground settings."
          />
          <div>
            <button
              className="action-button"
              type="button"
              onClick={() => setCommand("run_benchmark")}
              disabled={!benchmarkDatasetLoaded || !!benchmarkRunning}
            >
              {benchmarkRunning ? "Running..." : "Run Benchmark"}
            </button>
          </div>
        </div>
      </div>

      {selectedBenchmarkResult ? (
        <BenchmarkExplainFlow
          result={selectedBenchmarkResult}
          onBack={() => setSelectedBenchmarkQid(null)}
        />
      ) : (
        <TwoColumnLayout
          left={
            <div className="column-content">
                <h4 className="section-title">Summary</h4>
                <div id="doc-list-container">
                <ProgressStatusPanel
                  status={benchmarkStatus}
                  defaultTitle="Benchmark status"
                  completeTitle="Benchmark complete"
                  errorTitle="Benchmark failed"
                  defaultMessage="Preparing benchmark..."
                />
                <BenchmarkSummary
                  summary={benchmarkSummary}
                  scoreboardAvailable={!!scoreboardAvailable}
                  scoreboardLoading={!!scoreboardLoading}
                  scoreboardEntryName={scoreboardEntryName}
                  setScoreboardEntryName={setScoreboardEntryName}
                  onPublish={() => setCommand("publish_scoreboard")}
                />
                </div>
              </div>
            }
          right={
            <div className="column-content">
              <h4 className="section-title">Per Query Results</h4>
              <div id="doc-list-container">
                <BenchmarkResultsList results={benchmarkResults || []} onExplain={handleExplain} />
              </div>
            </div>
          }
        />
      )}
    </div>
  );
}

function formatScoreboardRank(rank) {
  if (rank === 1) return "🥇";
  if (rank === 2) return "🥈";
  if (rank === 3) return "🥉";
  return `${rank}.`;
}

function ScoreboardStep() {
  const [scoreboardEntries] = useModelState("scoreboard_entries");
  const [scoreboardLoading] = useModelState("scoreboard_loading");
  const [scoreboardAvailable] = useModelState("scoreboard_available");
  const [, setCommand] = useModelState("command");

  React.useEffect(() => {
    if (scoreboardAvailable) {
      setCommand("load_scoreboard");
    }
  }, [scoreboardAvailable, setCommand]);

  if (!scoreboardAvailable) {
    return (
      <div id="benchmark-section">
        <div className="column-content">
          <h4 className="section-title">Scoreboard</h4>
          <p>No scoreboard connection configured.</p>
        </div>
      </div>
    );
  }

  return (
    <div id="benchmark-section">
      <div className="column-content">
        <div className="scoreboard-header">
          <h4 className="section-title">Scoreboard</h4>
        </div>
        <div className="doc-card">
          {scoreboardLoading ? null : !(scoreboardEntries || []).length ? (
            <p>No published scores yet for the current weights and constraints.</p>
          ) : (
            <div className="scoreboard-table-wrap">
              <table className="scoreboard-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Final Score</th>
                    <th>Accuracy</th>
                    <th>Latency</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {(scoreboardEntries || []).map((entry) => (
                    <tr key={`${entry.rank}-${entry.name}-${entry.published_at || ""}`} className="scoreboard-row">
                      <td className="scoreboard-rank-cell">
                        <strong className="scoreboard-rank-badge">{formatScoreboardRank(entry.rank)}</strong>
                        <span>{entry.name || "Anonymous"}</span>
                      </td>
                      <td>{formatBenchmarkScore(entry.overall_score)} / 100</td>
                      <td>{formatBenchmarkScore(entry.accuracy_score)} / 100</td>
                      <td>{formatBenchmarkScore(entry.latency_score)} / 100</td>
                      <td>{formatBenchmarkScore(entry.cost_score)} / 100</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function detectInitialTheme() {
  if (typeof document === "undefined") return "light";

  const html = document.documentElement;
  const body = document.body;

  // **1. NEW: Check for Google Colab/Panel UI theme attribute on <html>**
  const colabTheme = html.getAttribute("theme");
  if (colabTheme === "dark" || colabTheme === "light") {
    return colabTheme;
  }

  // 2. Check for Google Colab dark theme class (Less reliable, but kept as a secondary check)
  //    (This was originally step 2 in your code, moved down)
  if (body.classList.contains("theme-dark")) return "dark";

  // 3. Check for JupyterLab/Jupyter Notebook theme attribute
  const jlLight = body.getAttribute("data-jp-theme-light");
  if (jlLight === "false") return "dark";
  if (jlLight === "true") return "light";

  // 4. Fallback to system preference (if available)
  if (typeof window !== "undefined" && window.matchMedia) {
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      return "dark";
    }
  }
  return "light";
}

function useNotebookTheme() {
  const [theme, setTheme] = React.useState(detectInitialTheme);

  React.useEffect(() => {
    // We observe both <html> and <body>
    const html = document.documentElement;
    const body = document.body;

    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.attributeName === "theme" && m.target === html) {
          // **NEW: Check <html> 'theme' attribute (Primary Colab change)**
          const colabTheme = html.getAttribute("theme");
          if (colabTheme === "dark" || colabTheme === "light") {
            setTheme(colabTheme);
            return;
          }
        }

        // Existing JupyterLab/Notebook attribute check
        if (m.attributeName === "data-jp-theme-light" && m.target === body) {
          const jlLight = body.getAttribute("data-jp-theme-light");
          setTheme(jlLight === "false" ? "dark" : "light");
          return;
        }

        // Existing Colab class change check (less common for theme switches)
        if (m.attributeName === "class" && m.target === body) {
          if (body.classList.contains("theme-dark")) {
            setTheme("dark");
          } else {
            // Re-evaluate the theme, which will check the <html> tag first
            setTheme(detectInitialTheme());
          }
          return;
        }
      }
    });

    if (html) {
      // Observe <html> for 'theme' attribute changes (Colab)
      observer.observe(html, {
        attributes: true,
        attributeFilter: ["theme"],
      });
    }

    if (body) {
      // Observe <body> for JupyterLab attribute and Colab class changes
      observer.observe(body, {
        attributes: true,
        attributeFilter: ["data-jp-theme-light", "class"],
      });
    }


    // System preference listener (only necessary if no notebook theme is set)
    // We keep this to handle pure system fallbacks, but note that `detectInitialTheme` 
    // is called on changes that might clear the notebook-set theme.
    let mq;
    const handleMq = (e) => {
      // Only change if the notebook environment hasn't explicitly set a theme
      if (!html.getAttribute("theme") && !body.getAttribute("data-jp-theme-light")) {
        setTheme(e.matches ? "dark" : "light");
      }
    }

    if (window.matchMedia) {
      mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (mq.addEventListener) {
        mq.addEventListener("change", handleMq);
      } else if (mq.addListener) {
        mq.addListener(handleMq);
      }
    }

    return () => {
      observer.disconnect();
      if (mq) {
        if (mq.removeEventListener) {
          mq.removeEventListener("change", handleMq);
        } else if (mq.removeListener) {
          mq.removeListener(handleMq);
        }
      }
    };
  }, []);

  return theme;
}

function WidgetHeader() {
  const [, setCommand] = useModelState("command");
  const [overrideSettingsPanel] = useModelState("override_settings_panel");
  const [overrideModelApiKey, setOverrideModelApiKey] = useModelState("override_model_api_key");
  const [overrideMongoUri, setOverrideMongoUri] = useModelState("override_mongodb_uri");
  const [overridesStatus] = useModelState("overrides_status");
  const [mongoDbName] = useModelState("mongo_db_name");
  const [mongoCollectionName] = useModelState("mongo_collection_name");
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);

  const destinationLabel =
    mongoDbName && mongoCollectionName
      ? `${mongoDbName}.${mongoCollectionName}`
      : "current database.collection";

  return (
    <div className="widget-shell-header">
      <div className="widget-header">
        <div className="widget-title-row">
          <h2 className="widget-title">MongoDB AI Search Playground</h2>
          <HelpTooltip
            message="An interactive playground to tune chunking, embeddings, retrieval, reranking, and benchmarking for a MongoDB-backed AI search workflow."
            ariaLabel="What this playground is"
          />
        </div>
        {overrideSettingsPanel ? (
          <button
            className={`icon-button settings-toggle${isSettingsOpen ? " active" : ""}`}
            type="button"
            aria-expanded={isSettingsOpen}
            aria-label="Open override settings"
            onClick={() => setIsSettingsOpen((open) => !open)}
          >
            <SettingsIcon />
          </button>
        ) : null}
      </div>

      {overrideSettingsPanel && isSettingsOpen ? (
        <div className="override-settings-panel">
          <div className="override-settings-grid">
            <div className="control-group">
              <FieldLabel
                label="Atlas Model API Key"
                help="Override the API key used by the configured embedding and reranking models for this widget session."
              />
              <input
                className="playground-input"
                type="password"
                value={overrideModelApiKey || ""}
                onChange={(e) => setOverrideModelApiKey(e.target.value)}
                placeholder="Enter an override key"
              />
            </div>
            <div className="control-group override-settings-wide">
              <FieldLabel
                label="MongoDB Connection String"
                help={`Override the active MongoDB URI while keeping this playground pointed at ${destinationLabel}.`}
              />
              <input
                className="playground-input"
                type="password"
                value={overrideMongoUri || ""}
                onChange={(e) => setOverrideMongoUri(e.target.value)}
                placeholder="mongodb+srv://..."
              />
            </div>
            <div className="control-group override-settings-actions">
              <FieldLabel
                label="Apply Overrides"
                help="Reconnect the widget with the override values entered above."
              />
              <div>
                <button
                  className="action-button"
                  type="button"
                  onClick={() => setCommand("apply_overrides")}
                >
                  Apply
                </button>
              </div>
            </div>
          </div>
          {overridesStatus ? (
            <div className="override-status-message">{overridesStatus}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function App() {
  const [currentStep, setCurrentStep] = useModelState("current_step");
  const [benchmarkTabVisible] = useModelState("benchmark_tab_visible");
  const [scoreboardTabVisible] = useModelState("scoreboard_tab_visible");
  const theme = useNotebookTheme();
  const visibleSteps = React.useMemo(() => {
    const steps = [1, 2, 3, 4];
    if (benchmarkTabVisible) steps.push(5);
    if (scoreboardTabVisible) steps.push(6);
    return steps;
  }, [benchmarkTabVisible, scoreboardTabVisible]);

  React.useEffect(() => {
    const normalizedStep = Number(currentStep || 1);
    if (!visibleSteps.includes(normalizedStep)) {
      setCurrentStep(visibleSteps[visibleSteps.length - 1] || 1);
    }
  }, [currentStep, visibleSteps, setCurrentStep]);

  return (
    <div
      id="mongodb-ai-playground"
      className={theme === "dark" ? "dark-theme" : "light-theme"}
    >
      <div className="content-wrapper">
        <WidgetHeader />
        <StepsNav
          currentStep={Number(currentStep || 1)}
          onChange={setCurrentStep}
          benchmarkTabVisible={!!benchmarkTabVisible}
          scoreboardTabVisible={!!scoreboardTabVisible}
        />
        {Number(currentStep || 1) === 1 && <ChunkingStep />}
        {Number(currentStep || 1) === 2 && <EmbeddingStep />}
        {Number(currentStep || 1) === 3 && <VectorSearchStep />}
        {Number(currentStep || 1) === 4 && <RerankingStep />}
        {benchmarkTabVisible && Number(currentStep || 1) === 5 && <BenchmarkStep />}
        {scoreboardTabVisible && Number(currentStep || 1) === 6 && <ScoreboardStep />}
      </div>
      <ErrorBox />
    </div>
  );
}

export const render = createRender(App);
export default { render };

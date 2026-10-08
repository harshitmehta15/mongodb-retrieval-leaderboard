import json
import os
from datetime import datetime, timezone
from http import HTTPStatus
from urllib.parse import parse_qs, urlparse

from pymongo import MongoClient, DESCENDING


CHALLENGE_CONFIG = {
    "event": "MongoDB User Group Dublin",
    "event_date": "2026-10-08",
    "dataset_id": "dublin-mug-2026-dcc-digital-strategy-v1",
    "source_document_title": "Dublin City Council Digital Transformation Strategy 2025-2030",
    "source_document_url": "https://www.dublincity.ie/sites/default/files/2025-04/dcc-digital-transformation-strategy-2025-2030.pdf",
    "source_document_sha256": "6e6158068517f2a37af7dc42b6f159a7eb4fb7a40851c6cd84e80083fb46ee66",
    "question_count": 18,
    "score_weights": {"accuracy": 0.4, "latency": 0.2, "cost": 0.4},
    "constraints": {"min_k": 5, "max_k": 15, "min_chunk_size": 125, "max_chunk_size": 1500},
    "embedding_models": ["voyage-4-lite", "voyage-4", "voyage-4-large", "voyage-context-4"],
    "reranker_models": ["rerank-2.5-lite", "rerank-2.5"],
    "submission_deadline": "08 Oct 2026, 8:50 PM Dublin time",
    "leaderboard_names": "Real names are shown on the public leaderboard.",
}


def json_response(handler, payload, status=HTTPStatus.OK):
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler, body, status=HTTPStatus.OK, content_type="text/html; charset=utf-8"):
    data = body.encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json_body(handler):
    length = int(handler.headers.get("content-length") or 0)
    if length <= 0:
        return {}
    if length > 256_000:
        raise ValueError("Request body is too large")
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw or "{}")


def get_query(handler):
    return parse_qs(urlparse(handler.path).query)


def leaderboard_collection():
    uri = os.environ.get("LEADERBOARD_MONGODB_URI", "").strip()
    database = os.environ.get("LEADERBOARD_DATABASE", "").strip()
    collection = os.environ.get("LEADERBOARD_COLLECTION", "").strip()
    if not uri or not database or not collection:
        return None
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[database][collection]


def require_submission_code(payload):
    configured = os.environ.get("SUBMISSION_CODE", "").strip()
    if not configured:
        return
    provided = str(payload.get("submission_code") or "").strip()
    if provided != configured:
        raise PermissionError("Invalid submission code")


def as_float(value, field, minimum=0.0, maximum=100.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number")
    if number < minimum or number > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return round(number, 4)


def build_submission(payload, client_ip=""):
    name = str(payload.get("name") or "").strip()[:80]
    if not name:
        raise ValueError("Name or display name is required")

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    source = summary if summary else payload

    overall = as_float(source.get("overall_score"), "overall_score")
    accuracy = as_float(source.get("accuracy_score"), "accuracy_score")
    latency = as_float(source.get("latency_score"), "latency_score")
    cost = as_float(source.get("cost_score"), "cost_score")

    return {
        "name": name,
        "public_name": name,
        "consent_public_name": True,
        "overall_score": overall,
        "accuracy_score": accuracy,
        "latency_score": latency,
        "cost_score": cost,
        "notes": str(payload.get("notes") or "").strip()[:1000],
        "dataset_id": CHALLENGE_CONFIG["dataset_id"],
        "score_weights": CHALLENGE_CONFIG["score_weights"],
        "constraints": CHALLENGE_CONFIG["constraints"],
        "created_at": datetime.now(timezone.utc),
        "client_ip_hint": client_ip[:80],
    }


def save_submission_from_handler(handler):
    collection = leaderboard_collection()
    if collection is None:
        json_response(handler, {"error": "Leaderboard MongoDB environment variables are not configured."}, HTTPStatus.SERVICE_UNAVAILABLE)
        return
    payload = read_json_body(handler)
    require_submission_code(payload)
    client_ip = handler.headers.get("x-forwarded-for", "").split(",")[0].strip()
    doc = build_submission(payload, client_ip=client_ip)
    collection.insert_one(doc)
    json_response(handler, {"ok": True, "submission": public_submission(doc)})


def public_submission(doc, rank=None):
    row = {
        "rank": rank,
        "name": doc.get("public_name") or "Anonymous",
        "overall_score": doc.get("overall_score"),
        "accuracy_score": doc.get("accuracy_score"),
        "latency_score": doc.get("latency_score"),
        "cost_score": doc.get("cost_score"),
        "created_at": doc.get("created_at").isoformat() if hasattr(doc.get("created_at"), "isoformat") else str(doc.get("created_at") or ""),
    }
    return row


def load_leaderboard(limit=50):
    collection = leaderboard_collection()
    if collection is None:
        return []
    docs = collection.find(
        {"dataset_id": CHALLENGE_CONFIG["dataset_id"]},
        {"_id": 0, "public_name": 1, "overall_score": 1, "accuracy_score": 1, "latency_score": 1, "cost_score": 1, "created_at": 1},
    ).sort("overall_score", DESCENDING).limit(int(limit))
    return [public_submission(doc, rank=index + 1) for index, doc in enumerate(docs)]

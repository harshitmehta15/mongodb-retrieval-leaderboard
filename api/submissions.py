from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

from ._shared import (
    build_submission,
    json_response,
    leaderboard_collection,
    load_leaderboard,
    read_json_body,
    require_submission_code,
)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            rows = load_leaderboard(limit=50)
            json_response(self, {"submissions": rows})
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        try:
            collection = leaderboard_collection()
            if collection is None:
                json_response(self, {"error": "Leaderboard MongoDB environment variables are not configured."}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            payload = read_json_body(self)
            require_submission_code(payload)
            client_ip = self.headers.get("x-forwarded-for", "").split(",")[0].strip()
            doc = build_submission(payload, client_ip=client_ip)
            collection.insert_one(doc)
            json_response(self, {"ok": True, "submission": {k: v for k, v in doc.items() if k not in {"_id", "client_ip_hint", "name"}}})
        except PermissionError as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except ValueError as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

from ._shared import (
    json_response,
    load_leaderboard,
    save_submission_from_handler,
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
            save_submission_from_handler(self)
        except PermissionError as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except ValueError as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

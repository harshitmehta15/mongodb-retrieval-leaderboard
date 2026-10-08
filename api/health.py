from http.server import BaseHTTPRequestHandler

from ._shared import json_response, leaderboard_collection


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        configured = leaderboard_collection() is not None
        json_response(self, {"ok": True, "leaderboard_configured": configured})

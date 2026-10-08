from http.server import BaseHTTPRequestHandler

from ._shared import CHALLENGE_CONFIG, json_response


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        json_response(self, CHALLENGE_CONFIG)

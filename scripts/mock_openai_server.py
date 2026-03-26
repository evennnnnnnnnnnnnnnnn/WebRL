"""
Minimal OpenAI-compatible completion server for local rollout testing.

Returns a fixed action response so we can test the full rollout pipeline
without needing vLLM or a real model. The response is always a simple
"click" action that the WebArena environment can parse.

Usage:
    python scripts/mock_openai_server.py --port 8000
"""

import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Fixed action responses that WebArena can parse
MOCK_RESPONSES = [
    "Let me analyze the page.\nclick [1]",
    "I see the page content.\nscroll [down]",
    "Based on the observation.\nclick [2]",
    "Let me proceed.\nexit",
]
response_idx = 0


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health" or self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/health":
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.wfile.write(json.dumps({
                    "data": [{"id": "mock-model", "object": "model"}]
                }).encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        global response_idx

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            request = {}

        # Cycle through mock responses
        text = MOCK_RESPONSES[response_idx % len(MOCK_RESPONSES)]
        response_idx += 1

        if "/completions" in self.path:
            # Completion API format
            response = {
                "id": f"cmpl-mock-{response_idx}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.get("model", "mock-model"),
                "choices": [{
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        elif "/chat/completions" in self.path:
            # Chat API format
            response = {
                "id": f"chatcmpl-mock-{response_idx}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        # Quieter logging
        print(f"[MockServer] {args[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), MockOpenAIHandler)
    print(f"Mock OpenAI server running on http://localhost:{args.port}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()

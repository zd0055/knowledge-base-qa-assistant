import json
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_loader import load_config
from src.embedding import EmbeddingManager
from src.retrieval import Retriever
from src.rag import RAGChain

logger = logging.getLogger(__name__)


class RAGAPIHandler(BaseHTTPRequestHandler):
    """HTTP API handler for RAG queries."""

    config = None
    rag_chain = None

    @classmethod
    def initialize(cls):
        cls.config = load_config()
        log_cfg = cls.config.get("logging", {})
        logging.basicConfig(
            level=getattr(logging, log_cfg.get("level", "INFO")),
            format=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        )
        embedding_manager = EmbeddingManager(cls.config)
        retriever = Retriever(cls.config, embedding_manager)
        cls.rag_chain = RAGChain(cls.config, retriever)

    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/health":
            self._send_json(200, {
                "status": "ok",
                "service": "knowledge-base-qa",
                "version": "1.0.0",
            })
        else:
            self._send_json(404, {
                "error": {"code": "not_found", "message": f"Endpoint not found: {parsed.path}"},
            })

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/v1/query":
            self._handle_query()
        else:
            self._send_json(404, {
                "error": {"code": "not_found", "message": f"Endpoint not found: {parsed.path}"},
            })

    def _handle_query(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json(400, {
                    "error": {"code": "bad_request", "message": "Request body is required"},
                })
                return

            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            question = body.get("question", "").strip()

            if not question:
                self._send_json(422, {
                    "error": {
                        "code": "validation_error",
                        "message": "Field 'question' is required and must be non-empty",
                        "details": [{"field": "question", "message": "Must be a non-empty string", "code": "required"}],
                    },
                })
                return

            logger.info("API query: %s", question[:100])
            result = self.rag_chain.query(question)

            response = {
                "data": {
                    "answer": result["answer"],
                    "sources": [
                        {
                            "source": doc.metadata.get("source", "unknown"),
                            "content_preview": doc.page_content[:300],
                        }
                        for doc in result["sources"]
                    ],
                    "source_count": result["source_count"],
                }
            }
            self._send_json(200, response)

        except json.JSONDecodeError:
            self._send_json(400, {
                "error": {"code": "bad_request", "message": "Invalid JSON in request body"},
            })
        except Exception as e:
            logger.exception("API query failed")
            self._send_json(500, {
                "error": {"code": "internal_error", "message": "Internal server error"},
            })


def create_server(host="0.0.0.0", port=8765):
    """Create and return the HTTP server instance."""
    RAGAPIHandler.initialize()
    server = HTTPServer((host, port), RAGAPIHandler)
    return server


def run_server(host="0.0.0.0", port=8765):
    """Run the RAG HTTP API server."""
    server = create_server(host, port)
    print(f"RAG API server running at http://{host}:{port}")
    print(f"  POST /api/v1/query  - Ask a question")
    print(f"  GET  /api/v1/health - Health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    run_server()

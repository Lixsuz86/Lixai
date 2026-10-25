"""
LixAI Yerel REST API Sunucusu (Standart Kütüphane Tabanlı / Sıfır Ek Bağımlılık)
Uç Noktalar:
  GET  /health           - Sunucu ve donanım sağlık durumu
  POST /generate         - Metin üretimi
"""
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from src.engine.interface import LixAI
from src.utils.logger import setup_logger

logger = setup_logger("lixai.api")
ai_instance = None


class LixAIRequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._set_headers(200)
            response = {
                "status": "healthy",
                "device": "cpu",
                "threads": ai_instance.num_threads if ai_instance else 12,
                "vocab_size": ai_instance.model.vocab_size if ai_instance else None,
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path == "/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                prompt = data.get("prompt", "")
                tokens = int(data.get("tokens", 30))
                temperature = float(data.get("temperature", 0.8))
                quantize = bool(data.get("quantize", False))

                generated = ai_instance.generate(
                    prompt=prompt,
                    max_new_tokens=tokens,
                    temperature=temperature,
                    use_quantized=quantize,
                )
                self._set_headers(200)
                self.wfile.write(json.dumps({"prompt": prompt, "generated": generated}).encode("utf-8"))
            except Exception as e:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))


def run_server(host: str = "127.0.0.1", port: int = 8000):
    global ai_instance
    ai_instance = LixAI()
    server = HTTPServer((host, port), LixAIRequestHandler)
    logger.info(f"LixAI API sunucusu http://{host}:{port} üzerinde başlatıldı.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Sunucu durduruldu.")
        server.server_close()


if __name__ == "__main__":
    run_server()

"""
Render.com free tier icin sleep olmamasi icin
bot ile birlikte calisan minimal HTTP sunucusu.
UptimeRobot bu endpoint'i ping'ler.
"""
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Marte is alive!")
    def log_message(self, format, *args):
        pass

def start_ping_server(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), PingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

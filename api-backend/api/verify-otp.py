from http.server import BaseHTTPRequestHandler
import json, requests

SUPABASE_URL = "https://fegouedpioqqzsuazbeb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZlZ291ZWRwaW9xcXpzdWF6YmViIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODMxNjc4MTIsImV4cCI6MjA5ODc0MzgxMn0.8GaW6cjoNhvI6k5ms4I_yVzSXcKSL40197f4Slrvin8"

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            email = body.get("email", "").strip()
            code = body.get("code", "").strip()
            
            if not email or not code:
                self._respond(400, {"error": "Email and code required"})
                return
            
            resp = requests.post(
                SUPABASE_URL + "/rest/v1/rpc/verify_otp_code",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": "Bearer " + SUPABASE_KEY,
                    "Content-Type": "application/json"
                },
                json={"p_email": email, "p_code": code}
            )
            
            if resp.status_code == 200:
                self._respond(200, {"valid": resp.json()})
            else:
                self._respond(500, {"error": "Verify failed"})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

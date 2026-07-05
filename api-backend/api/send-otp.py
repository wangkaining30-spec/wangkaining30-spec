from http.server import BaseHTTPRequestHandler
import json, random, smtplib, os, requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SUPABASE_URL = "https://fegouedpioqqzsuazbeb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZlZ291ZWRwaW9xcXpzdWF6YmViIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4MzE2NzgxMiwiZXhwIjoyMDk4NzQzODEyfQ.QLftr52yekOQ-hunwzHWuJqqYwPHkEqltWOMIfJzC60"

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
            
            if not email or "@" not in email:
                self._respond(400, {"error": "Invalid email"})
                return
            
            code = str(random.randint(100000, 999999))
            
            resp = requests.post(
                SUPABASE_URL + "/rest/v1/rpc/request_otp",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": "Bearer " + SUPABASE_KEY,
                    "Content-Type": "application/json"
                },
                json={"p_email": email}
            )
            
            if resp.status_code != 200:
                self._respond(500, {"error": "DB error"})
                return
            
            msg = MIMEMultipart()
            msg["From"] = "ke.chen@qq.com"
            msg["To"] = email
            msg["Subject"] = "客尘AI - 验证码 " + code
            html = "<h1 style='text-align:center;color:#d4a853;font-size:40px'>" + code + "</h1><p style='text-align:center'>5分钟内有效</p>"
            msg.attach(MIMEText(html, "html"))
            
            with smtplib.SMTP_SSL("smtp.qq.com", 465) as server:
                server.login("ke.chen@qq.com", "gqsygdkqqspzceff")
                server.send_message(msg)
            
            self._respond(200, {"success": True})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

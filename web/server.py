#!/usr/bin/env python3
"""
ds4 模型 HTTP API 服务器
用法: python3 server.py [--port 8080] [--model ds4_final.gguf]
"""

import http.server
import json
import subprocess
import os
import sys
import argparse
import urllib.parse
from pathlib import Path

# ========== 配置 ==========
MODEL_DIR = Path(__file__).parent.parent / "ds4pro"
MODEL_FILE = "ds4_final.gguf"
BINARY = "./train_ds4"

def run_inference(prompt: str, max_tokens: int = 64) -> dict:
    """调用 train_ds4 once 进行单次推理"""
    model_path = str(MODEL_DIR / MODEL_FILE)
    try:
        result = subprocess.run(
            [BINARY, "once", model_path, prompt],
            cwd=str(MODEL_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # 提取最后一行非空输出（跳过banner）
        lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
        output = lines[-1] if lines else ""
        # 过滤 banner 行
        if output.startswith("╔") or output.startswith("║") or output.startswith("╚"):
            output = ""
        return {
            "success": True,
            "response": output,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "response": "", "error": "推理超时"}
    except Exception as e:
        return {"success": False, "response": "", "error": str(e)}

# ========== HTTP 请求处理器 ==========
class DS4Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {args[0]}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # API: 健康检查
        if path == "/api/health":
            self.send_json({"status": "ok", "model": MODEL_FILE})
            return

        # API: 推理（GET方式）
        if path == "/api/generate":
            qs = urllib.parse.parse_qs(parsed.query)
            prompt = qs.get("prompt", [""])[0]
            if not prompt:
                self.send_json({"success": False, "error": "缺少 prompt 参数"}, 400)
                return
            result = run_inference(prompt)
            self.send_json(result)
            return

        # 静态文件
        if path == "/" or path == "":
            path = "/index.html"
        self.serve_static(path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_json({"success": False, "error": "JSON 解析失败"}, 400)
                return
            prompt = data.get("prompt", "")
            if not prompt:
                self.send_json({"success": False, "error": "缺少 prompt 字段"}, 400)
                return
            result = run_inference(prompt)
            self.send_json(result)
            return

        self.send_error(404)

    def serve_static(self, path: str):
        """提供静态文件"""
        # 安全检查
        safe_path = os.path.normpath(path).lstrip("/")
        file_path = Path(__file__).parent / safe_path

        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }.get(file_path.suffix, "application/octet-stream")

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="ds4 HTTP API Server")
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")
    parser.add_argument("--model", type=str, default="ds4_final.gguf", help="模型文件名")
    args = parser.parse_args()

    global MODEL_FILE
    MODEL_FILE = args.model

    print(f"""
╔═══════════════════════════════════╗
║     ds4 Web API Server           ║
║     模型: {MODEL_FILE:<23}║
║     端口: {args.port:<23}║
║     地址: http://localhost:{args.port:<5}║
╚═══════════════════════════════════╝
""")
    server = http.server.HTTPServer(("0.0.0.0", args.port), DS4Handler)
    print(f"✅ 服务已启动，访问 http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
        server.shutdown()


if __name__ == "__main__":
    main()

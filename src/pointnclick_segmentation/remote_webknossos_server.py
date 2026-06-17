from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pointnclick_segmentation.model_store import default_app_dir
from pointnclick_segmentation.utils import ensure_dir
from pointnclick_segmentation.webknossos_bridge import (
    _layer_names,
    mask_to_row_runs,
    read_padded_grayscale_crop,
)


USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")
PASSWORD_MIN_LENGTH = 8
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30
PASSWORD_HASH_ITERATIONS = 310_000


@dataclass
class RemoteWebKnossosServerConfig:
    checkpoint_path: str
    host: str = "0.0.0.0"
    port: int = 8765
    webknossos_url: str = "https://webknossos.org"
    color_layer: str = "color"
    mag: str = "1"
    crop_size: int = 512
    threshold: float = 0.5
    image_size: int | None = None
    device_name: str = "cuda"
    timeout_s: int = 120
    output_dir: str = str(default_app_dir() / "remote_webknossos_server")
    database_path: str = str(default_app_dir() / "remote_webknossos_server" / "server.db")
    secret_key_path: str = str(default_app_dir() / "remote_webknossos_server" / "server_secret.key")


class PointnClickHttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class TokenCipher:
    def __init__(self, key_path: str | Path) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError(
                "The remote WebKnossos server stores user auth tokens and needs the "
                "Python package 'cryptography'. Install it with: pip install cryptography"
            ) from exc

        self.key_path = Path(key_path)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        if self.key_path.exists():
            key = self.key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            self.key_path.write_bytes(key + b"\n")
        self._fernet = Fernet(key)

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")


class AccountStore:
    def __init__(self, database_path: str | Path, cipher: TokenCipher) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.cipher = cipher
        self.lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    webknossos_token_encrypted TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(username) REFERENCES users(username)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS sessions_username_idx ON sessions(username)")

    def create_user(self, username: str, password: str, webknossos_token: str | None = None) -> dict[str, Any]:
        username = validate_username(username)
        validate_password(password)
        now = int(time.time())
        encrypted_token = self.cipher.encrypt(webknossos_token.strip()) if webknossos_token and webknossos_token.strip() else None
        try:
            with self.lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users(username, password_hash, webknossos_token_encrypted, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (username, hash_password(password), encrypted_token, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise PointnClickHttpError(409, "That username already exists.") from exc
        return self.get_user(username)

    def verify_user(self, username: str, password: str) -> dict[str, Any]:
        username = username.strip()
        with self.lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None or not verify_password(password, str(row["password_hash"])):
            raise PointnClickHttpError(401, "Invalid username or password.")
        return row_to_user(row, self.cipher)

    def get_user(self, username: str) -> dict[str, Any]:
        with self.lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            raise PointnClickHttpError(404, "User not found.")
        return row_to_user(row, self.cipher)

    def set_webknossos_token(self, username: str, webknossos_token: str) -> dict[str, Any]:
        token = webknossos_token.strip()
        if not token:
            raise PointnClickHttpError(400, "WebKnossos token cannot be blank.")
        now = int(time.time())
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE users SET webknossos_token_encrypted = ?, updated_at = ? WHERE username = ?",
                (self.cipher.encrypt(token), now, username),
            )
        return self.get_user(username)

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self.lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(token_hash, username, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (hash_session_token(token), username, now + SESSION_TTL_SECONDS, now),
            )
        return token

    def user_for_session(self, token: str) -> dict[str, Any]:
        token = token.strip()
        if not token:
            raise PointnClickHttpError(401, "Missing session token.")
        token_hash = hash_session_token(token)
        now = int(time.time())
        with self.lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT users.*
                FROM sessions
                JOIN users ON users.username = sessions.username
                WHERE sessions.token_hash = ? AND sessions.expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
        if row is None:
            raise PointnClickHttpError(401, "Session expired or invalid. Sign in again.")
        return row_to_user(row, self.cipher)


class RemoteWebKnossosPredictor:
    def __init__(self, config: RemoteWebKnossosServerConfig) -> None:
        try:
            import webknossos as wk
        except ImportError as exc:
            raise RuntimeError(
                "The remote WebKnossos server needs the Python package named 'webknossos'. "
                "Install it with: pip install webknossos"
            ) from exc

        from pointnclick_segmentation.infer import LoadedPredictor

        self.config = config
        self.wk = wk
        self.output_dir = ensure_dir(config.output_dir)
        self.events_path = self.output_dir / "remote_events.jsonl"
        self.lock = threading.Lock()
        self.predictor = LoadedPredictor(
            checkpoint_path=config.checkpoint_path,
            image_size=config.image_size,
            crop_size=config.crop_size,
            device_name=config.device_name,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "remote-webknossos",
            "webknossos_url": self.config.webknossos_url,
            "color_layer": self.config.color_layer,
            "mag": self.config.mag,
            "device": self.config.device_name,
            "crop_size": self.config.crop_size,
            "auth_required": True,
        }

    def predict(self, user: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        request_t0 = time.perf_counter()
        webknossos_token = user.get("webknossos_token")
        if not webknossos_token:
            raise PointnClickHttpError(400, "Your account does not have a WebKnossos auth token saved yet.")

        dataset_url = str(payload.get("dataset_url") or payload.get("dataset") or "").strip()
        if not dataset_url:
            raise PointnClickHttpError(400, "Prediction request is missing dataset_url.")

        position = as_int_triplet(payload.get("position"), "position")
        segment_id = int(payload.get("segment_id", 0))
        if segment_id <= 0:
            raise PointnClickHttpError(400, "segment_id must be a positive integer.")

        color_layer = str(payload.get("color_layer") or self.config.color_layer).strip() or self.config.color_layer
        mag = str(payload.get("mag") or self.config.mag).strip() or self.config.mag
        click_x, click_y, click_z = position
        half = self.config.crop_size // 2
        minx = click_x - half
        miny = click_y - half
        width = self.config.crop_size
        height = self.config.crop_size
        timings_ms: dict[str, float] = {}

        # The webknossos Python context is process-global in some releases. Keep the
        # whole request serialized so simultaneous users never mix auth contexts.
        with self.lock:
            open_t0 = time.perf_counter()
            with self.wk.webknossos_context(
                url=self.config.webknossos_url,
                token=webknossos_token,
                timeout=self.config.timeout_s,
            ):
                dataset = self.wk.RemoteDataset.open(dataset_name_or_url=dataset_url)
                layer = get_layer(dataset, color_layer)
                mag_view = get_mag_view(layer, mag, self.wk)
                layer_names = _layer_names(dataset)
                timings_ms["open_dataset"] = (time.perf_counter() - open_t0) * 1000.0

                read_t0 = time.perf_counter()
                image, read_bounds = read_padded_grayscale_crop(
                    mag_view=mag_view,
                    minx=minx,
                    miny=miny,
                    z=click_z,
                    width=width,
                    height=height,
                )
                timings_ms["read_em"] = (time.perf_counter() - read_t0) * 1000.0

            predict_t0 = time.perf_counter()
            pred_mask = self.predictor.predict(
                image=image,
                x=click_x - minx,
                y=click_y - miny,
                threshold=self.config.threshold,
            )
            timings_ms["predict_mask"] = (time.perf_counter() - predict_t0) * 1000.0

        runs_t0 = time.perf_counter()
        runs = mask_to_row_runs(pred_mask, minx=minx, miny=miny)
        num_pixels = int(sum(x1 - x0 for _y, x0, x1 in runs))
        timings_ms["encode_runs"] = (time.perf_counter() - runs_t0) * 1000.0
        timings_ms["request_total"] = (time.perf_counter() - request_t0) * 1000.0

        response = {
            "status": "ok",
            "segment_id": segment_id,
            "position": [click_x, click_y, click_z],
            "z": click_z,
            "dataset_url": dataset_url,
            "color_layer": color_layer,
            "available_layers": layer_names,
            "bbox": [minx, minx + width - 1, miny, miny + height - 1, click_z, click_z],
            "read_bbox": read_bounds,
            "runs": runs,
            "num_pixels": num_pixels,
            "timings_ms": timings_ms,
        }
        self._record_event(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "username": user["username"],
                "request": {
                    "dataset_url": dataset_url,
                    "position": [click_x, click_y, click_z],
                    "segment_id": segment_id,
                },
                "response": {
                    "num_pixels": num_pixels,
                    "num_runs": len(runs),
                    "bbox": response["bbox"],
                    "read_bbox": response["read_bbox"],
                },
                "timings_ms": timings_ms,
            }
        )
        return response

    def _record_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


class RemoteWebKnossosServer:
    def __init__(self, config: RemoteWebKnossosServerConfig) -> None:
        self.config = config
        ensure_dir(config.output_dir)
        self.accounts = AccountStore(config.database_path, TokenCipher(config.secret_key_path))
        self.predictor = RemoteWebKnossosPredictor(config)

    def run(self) -> None:
        handler = make_handler(self)
        server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
        print("PointnClick remote WebKnossos server is running.")
        print(f"Listening on: http://{self.config.host}:{self.config.port}")
        print("For iPads on Tailscale, use this PC's Tailscale IP as the bridge URL.")
        try:
            server.serve_forever()
        finally:
            server.server_close()

    def route(self, method: str, path: str, body: dict[str, Any], headers: Any) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path in {"/health", "/api/health"}:
            return 200, self.predictor.health()

        if method == "POST" and path == "/api/auth/register":
            user = self.accounts.create_user(
                username=str(body.get("username") or ""),
                password=str(body.get("password") or ""),
                webknossos_token=str(body.get("webknossos_token") or ""),
            )
            session_token = self.accounts.create_session(user["username"])
            return 200, auth_response(user, session_token)

        if method == "POST" and path == "/api/auth/login":
            user = self.accounts.verify_user(
                username=str(body.get("username") or ""),
                password=str(body.get("password") or ""),
            )
            session_token = self.accounts.create_session(user["username"])
            return 200, auth_response(user, session_token)

        if method == "GET" and path == "/api/me":
            user = self._require_user(headers)
            return 200, public_user(user)

        if method == "POST" and path == "/api/me/webknossos-token":
            user = self._require_user(headers)
            updated = self.accounts.set_webknossos_token(
                username=user["username"],
                webknossos_token=str(body.get("webknossos_token") or ""),
            )
            return 200, public_user(updated)

        if method == "POST" and path == "/api/predict":
            user = self._require_user(headers)
            return 200, self.predictor.predict(user, body)

        if method == "POST" and path == "/predict":
            raise PointnClickHttpError(401, "Sign in to the PointnClick extension before using the remote server.")

        raise PointnClickHttpError(404, "Not found.")

    def _require_user(self, headers: Any) -> dict[str, Any]:
        auth = str(headers.get("Authorization", ""))
        if not auth.lower().startswith("bearer "):
            raise PointnClickHttpError(401, "Missing Authorization bearer token.")
        return self.accounts.user_for_session(auth.split(" ", 1)[1])


def run_remote_webknossos_server(config: RemoteWebKnossosServerConfig) -> None:
    RemoteWebKnossosServer(config).run()


def make_handler(server: RemoteWebKnossosServer) -> type[BaseHTTPRequestHandler]:
    class PointnClickRemoteHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self._send_bytes(b"", status=204)

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] in {"/", "/dashboard"}:
                self._send_bytes(DASHBOARD_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
                return
            self._handle_json_route("GET")

        def do_POST(self) -> None:
            self._handle_json_route("POST")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle_json_route(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            try:
                body = self._read_json() if method == "POST" else {}
                status, response = server.route(method, path, body, self.headers)
                self._send_json(response, status=status)
            except PointnClickHttpError as exc:
                self._send_json({"status": "error", "message": exc.message}, status=exc.status)
            except Exception as exc:
                self._send_json({"status": "error", "message": str(exc)}, status=500)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
            self._send_bytes(json.dumps(data).encode("utf-8"), status=status, content_type="application/json")

        def _send_bytes(self, data: bytes, status: int = 200, content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            if data:
                self.wfile.write(data)

    return PointnClickRemoteHandler


def validate_username(username: str) -> str:
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise PointnClickHttpError(400, "Username must be 3-64 characters and use letters, numbers, '.', '_', or '-'.")
    return username


def validate_password(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise PointnClickHttpError(400, f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_HASH_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.urlsafe_b64decode(salt_raw.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_raw.encode("ascii"))
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def row_to_user(row: sqlite3.Row, cipher: TokenCipher) -> dict[str, Any]:
    encrypted = row["webknossos_token_encrypted"]
    token = cipher.decrypt(str(encrypted)) if encrypted else ""
    return {
        "username": str(row["username"]),
        "webknossos_token": token,
        "has_webknossos_token": bool(token),
        "created_at": int(row["created_at"]),
        "updated_at": int(row["updated_at"]),
    }


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "username": user["username"],
        "has_webknossos_token": bool(user.get("webknossos_token")),
    }


def auth_response(user: dict[str, Any], session_token: str) -> dict[str, Any]:
    data = public_user(user)
    data["session_token"] = session_token
    data["expires_in_seconds"] = SESSION_TTL_SECONDS
    return data


def as_int_triplet(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise PointnClickHttpError(400, f"{name} must be [x, y, z].")
    return int(round(float(value[0]))), int(round(float(value[1]))), int(round(float(value[2])))


def get_layer(dataset: Any, color_layer: str) -> Any:
    if hasattr(dataset, "get_layer"):
        try:
            return dataset.get_layer(color_layer)
        except Exception as exc:
            names = _layer_names(dataset)
            if len(names) == 1 and color_layer == "color":
                return dataset.get_layer(names[0])
            raise RuntimeError(
                f"Could not open WebKnossos color layer '{color_layer}'. "
                f"Available layers: {', '.join(names) or 'unknown'}"
            ) from exc
    layers = getattr(dataset, "layers", {})
    try:
        return layers[color_layer]
    except Exception as exc:
        raise RuntimeError(f"Could not open WebKnossos layer '{color_layer}'") from exc


def get_mag_view(layer: Any, mag: str, wk: Any) -> Any:
    attempts: list[Any] = [mag]
    try:
        attempts.append(int(mag))
    except ValueError:
        pass
    try:
        attempts.append(wk.Mag(mag))
    except Exception:
        pass
    last_error: Exception | None = None
    for candidate in attempts:
        try:
            return layer.get_mag(candidate)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not open magnification '{mag}'") from last_error


DASHBOARD_HTML = r"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PointnClick Server</title>
    <style>
      :root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      body { margin: 0; background: #f6f7f9; color: #15191f; }
      main { max-width: 720px; margin: 0 auto; padding: 24px; }
      h1 { font-size: 24px; margin: 0 0 4px; }
      p { margin: 0 0 18px; color: #5b6472; }
      section { background: #fff; border: 1px solid #d8dde6; border-radius: 8px; padding: 16px; margin: 14px 0; }
      label { display: block; font-size: 13px; font-weight: 650; margin: 12px 0 5px; }
      input, textarea { box-sizing: border-box; width: 100%; font: inherit; border: 1px solid #b9c1cf; border-radius: 6px; padding: 10px; background: #fff; color: inherit; }
      textarea { min-height: 88px; resize: vertical; }
      button { border: 0; border-radius: 6px; padding: 10px 14px; font: inherit; font-weight: 700; background: #0f62fe; color: #fff; margin: 12px 8px 0 0; }
      button.secondary { background: #e8ecf2; color: #172033; }
      pre { white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 6px; min-height: 40px; }
      @media (prefers-color-scheme: dark) {
        body { background: #101318; color: #f3f4f6; }
        section { background: #171b22; border-color: #303845; }
        input, textarea { background: #11151c; border-color: #48515f; }
        button.secondary { background: #303845; color: #f3f4f6; }
      }
    </style>
  </head>
  <body>
    <main>
      <h1>PointnClick Server</h1>
      <p>Use this page over Tailscale to create an account and save your WebKnossos auth token.</p>

      <section>
        <h2>Sign In</h2>
        <label for="username">Username</label>
        <input id="username" autocomplete="username">
        <label for="password">Password</label>
        <input id="password" type="password" autocomplete="current-password">
        <button id="login">Sign in</button>
        <button id="register" class="secondary">Create account</button>
      </section>

      <section>
        <h2>WebKnossos Token</h2>
        <label for="wk-token">Auth token</label>
        <textarea id="wk-token" spellcheck="false" autocomplete="off"></textarea>
        <button id="save-token">Save token</button>
        <button id="me" class="secondary">Check account</button>
      </section>

      <section>
        <h2>Status</h2>
        <button id="health" class="secondary">Server health</button>
        <pre id="status">Not signed in.</pre>
      </section>
    </main>
    <script>
      const statusEl = document.getElementById("status");
      const tokenKey = "pointnclickSessionToken";
      function setStatus(value) {
        statusEl.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
      }
      async function api(path, options = {}) {
        const headers = {"Content-Type": "application/json", ...(options.headers || {})};
        const token = localStorage.getItem(tokenKey);
        if (token) headers.Authorization = `Bearer ${token}`;
        const response = await fetch(path, {...options, headers});
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || `HTTP ${response.status}`);
        return data;
      }
      document.getElementById("login").onclick = async () => {
        try {
          const data = await api("/api/auth/login", {
            method: "POST",
            body: JSON.stringify({
              username: document.getElementById("username").value,
              password: document.getElementById("password").value,
            }),
          });
          localStorage.setItem(tokenKey, data.session_token);
          setStatus(data);
        } catch (error) { setStatus(error.message); }
      };
      document.getElementById("register").onclick = async () => {
        try {
          const data = await api("/api/auth/register", {
            method: "POST",
            body: JSON.stringify({
              username: document.getElementById("username").value,
              password: document.getElementById("password").value,
              webknossos_token: document.getElementById("wk-token").value,
            }),
          });
          localStorage.setItem(tokenKey, data.session_token);
          setStatus(data);
        } catch (error) { setStatus(error.message); }
      };
      document.getElementById("save-token").onclick = async () => {
        try {
          setStatus(await api("/api/me/webknossos-token", {
            method: "POST",
            body: JSON.stringify({webknossos_token: document.getElementById("wk-token").value}),
          }));
        } catch (error) { setStatus(error.message); }
      };
      document.getElementById("me").onclick = async () => {
        try { setStatus(await api("/api/me")); } catch (error) { setStatus(error.message); }
      };
      document.getElementById("health").onclick = async () => {
        try { setStatus(await api("/health", {headers: {}})); } catch (error) { setStatus(error.message); }
      };
    </script>
  </body>
</html>
"""

"""HTTP fetcher yang sopan: patuh robots.txt, rate-limit per host, retry, batas ukuran."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

USER_AGENT = "UGM-Academic-RAG-Bot/0.1 (student HPC course project; polite crawler)"


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status: int = 0
    content_type: str = ""
    content: bytes = b""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300 and bool(self.content)

    @property
    def is_pdf(self) -> bool:
        return self.content[:5] == b"%PDF-" or "pdf" in self.content_type.lower()

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.lower() and not self.is_pdf


class PoliteFetcher:
    def __init__(
        self,
        delay_s: float = 1.5,
        timeout_s: float = 45.0,
        max_bytes: int = 60 * 2**20,
        retries: int = 2,
        respect_robots: bool = True,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.retries = retries
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "id,en;q=0.8"})
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- robots
    def _robots_for(self, url: str) -> RobotFileParser | None:
        u = urlparse(url)
        base = f"{u.scheme}://{u.netloc}"
        if base in self._robots:
            return self._robots[base]
        rp: RobotFileParser | None = RobotFileParser()
        try:
            self._wait(u.netloc)
            r = self.session.get(base + "/robots.txt", timeout=15)
            if r.status_code in (401, 403):
                rp.parse(["User-agent: *", "Disallow: /"])
            elif r.status_code >= 400:
                rp = None  # tidak ada robots.txt -> semua diizinkan
            else:
                rp.parse(r.text.splitlines())
        except requests.RequestException:
            rp = None
        self._robots[base] = rp
        return rp

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rp = self._robots_for(url)
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    def _wait(self, host: str) -> None:
        with self._lock:
            last = self._last_hit.get(host, 0.0)
            wait = self.delay_s - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_hit[host] = time.monotonic()

    # ---------------------------------------------------------------- fetch
    def fetch(self, url: str) -> FetchResult:
        res = FetchResult(url=url)
        if urlparse(url).scheme not in ("http", "https"):
            res.error = "hanya URL http/https yang didukung"
            return res
        if not self.allowed(url):
            res.error = "diblokir robots.txt"
            return res
        host = urlparse(url).netloc
        for attempt in range(self.retries + 1):
            self._wait(host)
            try:
                with self.session.get(url, timeout=self.timeout_s, stream=True, allow_redirects=True) as r:
                    res.status = r.status_code
                    res.final_url = r.url
                    res.content_type = r.headers.get("Content-Type", "")
                    if r.status_code >= 500 and attempt < self.retries:
                        time.sleep(2 ** attempt)
                        continue
                    if r.status_code >= 400:
                        res.error = f"HTTP {r.status_code}"
                        return res
                    buf = bytearray()
                    for part in r.iter_content(64 * 1024):
                        buf.extend(part)
                        if len(buf) > self.max_bytes:
                            res.error = f"file melebihi batas {self.max_bytes // 2**20} MB"
                            return res
                    res.content = bytes(buf)
                    res.error = None
                    return res
            except requests.RequestException as exc:
                res.error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        return res

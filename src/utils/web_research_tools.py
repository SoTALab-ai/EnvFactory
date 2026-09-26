"""Constrained web research tools for schema-generation agents."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from agents import function_tool
from ddgs import DDGS


DEFAULT_MAX_PAGE_BYTES = 2_000_000
DEFAULT_MAX_PAGE_CHARS = 20_000
MAX_PAGE_CHARS = 50_000
USER_AGENT = "EnvFactory-SchemaGen/1.0 (+https://github.com/SoTALab-ai/EnvFactory)"
AGENT_BROWSER_ENV = "ENVFACTORY_AGENT_BROWSER_BIN"


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not supported")

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve hostname: {parsed.hostname}") from exc

    if not addresses:
        raise ValueError(f"Unable to resolve hostname: {parsed.hostname}")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private, loopback, link-local, and reserved addresses are blocked")


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ReadableHTMLParser(HTMLParser):
    _ignored_tags = {"script", "style", "noscript", "svg", "canvas", "template"}
    _block_tags = {
        "article",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.markdown_alternates: list[str] = []
        self.meta_descriptions: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value for key, value in attrs if key and value}
        if tag == "link":
            rel = attributes.get("rel", "").lower().split()
            if "alternate" in rel and attributes.get("type", "").lower() == "text/markdown":
                href = attributes.get("href")
                if href:
                    self.markdown_alternates.append(href)
        if tag == "meta" and attributes.get("name", "").lower() == "description":
            description = attributes.get("content")
            if description:
                self.meta_descriptions.append(description)
        if tag in self._ignored_tags:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self._block_tags:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self._block_tags:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)


def _normalize_text(parts: list[str]) -> str:
    text = "".join(parts).replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _extract_page_text(body: str, content_type: str) -> tuple[str, str]:
    if "html" not in content_type.lower() and "<html" not in body[:500].lower():
        return "", body.strip()

    parser = _ReadableHTMLParser()
    parser.feed(body)
    parser.close()
    content = _normalize_text(parser.text_parts)
    if len(content) < 200 and parser.meta_descriptions:
        content = _normalize_text([*parser.text_parts, "\n", *parser.meta_descriptions])
    return _normalize_text(parser.title_parts), content


def _find_markdown_alternate(body: str, base_url: str) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(body)
    parser.close()
    if not parser.markdown_alternates:
        return ""
    return urljoin(base_url, parser.markdown_alternates[0])


def _fetch_public_page(url: str) -> tuple[str, str, str]:
    _validate_public_url(url)
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/markdown,text/html,text/plain,application/json,application/xml",
        },
    )
    opener = build_opener(_SafeRedirectHandler())
    with opener.open(request, timeout=15) as response:
        final_url = response.geturl()
        _validate_public_url(final_url)
        content_type = response.headers.get_content_type()
        if not (
            content_type.startswith("text/")
            or content_type in {"application/json", "application/xml", "application/xhtml+xml"}
        ):
            raise ValueError(f"Unsupported content type: {content_type}")
        raw = response.read(DEFAULT_MAX_PAGE_BYTES + 1)
        if len(raw) > DEFAULT_MAX_PAGE_BYTES:
            raise ValueError(f"Page exceeds {DEFAULT_MAX_PAGE_BYTES} byte limit")
        charset = response.headers.get_content_charset() or "utf-8"
    return final_url, content_type, raw.decode(charset, errors="replace")


def search_web_impl(query: str, max_results: int = 5) -> str:
    """Search the public web and return compact result metadata as JSON."""
    query = query.strip()
    if not query:
        return _json_result({"error": "query must not be empty", "results": []})

    max_results = max(1, min(max_results, 10))
    try:
        with DDGS(timeout=10) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return _json_result({"error": f"search failed: {exc}", "query": query, "results": []})

    results = []
    for item in raw_results:
        url = item.get("href") or item.get("url") or ""
        if not url:
            continue
        results.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("body") or item.get("description") or "",
            }
        )
    return _json_result({"query": query, "results": results})


def _read_webpage_native(url: str, max_chars: int) -> dict[str, Any]:
    try:
        final_url, content_type, body = _fetch_public_page(url)
    except (HTTPError, URLError, OSError, ValueError, LookupError) as exc:
        return {"error": str(exc), "url": url, "backend": "native"}

    title, content = _extract_page_text(body, content_type)
    alternate_error = ""
    if "html" in content_type.lower():
        markdown_url = _find_markdown_alternate(body, final_url)
        if markdown_url:
            try:
                markdown_final_url, markdown_type, markdown_body = _fetch_public_page(markdown_url)
                _, markdown_content = _extract_page_text(markdown_body, markdown_type)
                if markdown_content:
                    final_url = markdown_final_url
                    content = markdown_content
            except (HTTPError, URLError, OSError, ValueError, LookupError) as exc:
                alternate_error = str(exc)

    if not title and content.startswith("# "):
        title = content.splitlines()[0][2:].strip()
    truncated = len(content) > max_chars
    result = {
        "requested_url": url,
        "url": final_url,
        "title": title,
        "content": content[:max_chars],
        "truncated": truncated,
        "backend": "native",
    }
    if alternate_error:
        result["alternate_error"] = alternate_error
    return result


def _read_with_agent_browser(url: str, max_chars: int) -> dict[str, Any]:
    _validate_public_url(url)
    executable = os.environ.get(AGENT_BROWSER_ENV) or shutil.which("agent-browser")
    if not executable:
        raise RuntimeError("agent-browser executable is not installed")

    hostname = urlparse(url).hostname
    session = f"envfactory-schema-{os.getpid()}-{uuid4().hex[:8]}"
    command = [
        executable,
        "--session",
        session,
        "read",
        url,
        "--json",
        "--allowed-domains",
        hostname,
        "--content-boundaries",
        "--max-output",
        str(max_chars),
        "--timeout",
        "15000",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"agent-browser read failed: {exc}") from exc
    finally:
        try:
            subprocess.run(
                [executable, "--session", session, "close"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"agent-browser read failed: {detail[-500:]}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("agent-browser returned invalid JSON") from exc
    if not payload.get("success") or payload.get("error"):
        raise RuntimeError(f"agent-browser read failed: {payload.get('error')}")

    data = payload.get("data") or {}
    content = data.get("content") or ""
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("agent-browser returned no readable content")
    final_url = data.get("finalUrl") or data.get("url") or url
    _validate_public_url(final_url)
    title = ""
    if content.startswith("# "):
        title = content.splitlines()[0][2:].strip()
    return {
        "requested_url": url,
        "url": final_url,
        "title": title,
        "content": content[:max_chars],
        "truncated": bool(data.get("truncated")) or len(content) > max_chars,
        "content_type": data.get("contentType", ""),
        "source": data.get("source", ""),
        "backend": "agent-browser",
    }


def read_webpage_impl(url: str, max_chars: int = DEFAULT_MAX_PAGE_CHARS) -> str:
    """Read a public page with agent-browser, falling back to bounded HTTP."""
    max_chars = max(1_000, min(max_chars, MAX_PAGE_CHARS))
    try:
        result = _read_with_agent_browser(url, max_chars)
    except (RuntimeError, ValueError) as browser_exc:
        result = _read_webpage_native(url, max_chars)
        result["browser_error"] = str(browser_exc)
    return _json_result(result)


@function_tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the public web for API documentation and return titles, URLs, and snippets."""
    return search_web_impl(query=query, max_results=max_results)


@function_tool
def read_webpage(url: str, max_chars: int = DEFAULT_MAX_PAGE_CHARS) -> str:
    """Read a public HTTP(S) page and return its title and extracted textual content."""
    return read_webpage_impl(url=url, max_chars=max_chars)

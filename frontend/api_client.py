"""Thin HTTP client for the FastAPI backend. All network calls from the UI go through here."""
from typing import Any

import httpx

import texts as T
from config import BACKEND_URL, REQUEST_TIMEOUT


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None, errors: list[dict] | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.errors = errors or []


def _request(method: str, path: str, admin_key: str | None = None, timeout: float | None = None, **kwargs) -> Any:
    headers = kwargs.pop("headers", {})
    if admin_key:
        headers["X-Admin-Key"] = admin_key
    try:
        response = httpx.request(
            method, f"{BACKEND_URL}{path}", headers=headers, timeout=timeout or REQUEST_TIMEOUT, **kwargs
        )
    except httpx.TimeoutException as exc:
        raise ApiError(T.ERR_TIMEOUT) from exc
    except httpx.HTTPError as exc:
        raise ApiError(T.ERR_CONNECTION) from exc

    try:
        data = response.json()
    except ValueError:
        data = None
    if response.is_error:
        detail = data.get("detail") if isinstance(data, dict) else None
        errors = data.get("errors") if isinstance(data, dict) else None
        raise ApiError(detail if isinstance(detail, str) else T.ERR_UNKNOWN, response.status_code, errors)
    return data


def health() -> dict:
    return _request("GET", "/health", timeout=5)


def chat(message: str, temperature: float) -> dict:
    return _request("POST", "/chat", json={"message": message, "temperature": temperature})


def list_documents(admin_key: str) -> list[dict]:
    return _request("GET", "/admin/documents", admin_key=admin_key)


def upload_documents(admin_key: str, files: list[tuple[str, bytes, str]]) -> dict:
    """files: list of (filename, content, mime_type)."""
    return _request("POST", "/admin/upload", admin_key=admin_key, files=[("files", f) for f in files])


def delete_document(admin_key: str, doc_id: str) -> dict:
    return _request("DELETE", f"/admin/documents/{doc_id}", admin_key=admin_key)

"""HTTP client wrapping the FastAPI endpoints.

Keeps every request in one place so pages stay focused on the UI.
Base URL comes from the ``API_BASE_URL`` env variable (default
``http://localhost:8000``) so the dashboard can point at a remote
instance without touching the code.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_TIMEOUT = 120.0


class APIError(RuntimeError):
    """Raised when the API returns a non-2xx status."""

    def __init__(self, status_code: int, message: str, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.payload = payload or {}


class APIClient:
    """Thin synchronous client over ``httpx``."""

    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = (
            base_url or os.environ.get("API_BASE_URL") or "http://localhost:8000"
        ).rstrip("/")
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs)

    def _post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(method, url, **kwargs)
        if response.status_code >= 400:
            try:
                payload = response.json()
                message = (
                    payload.get("error", {}).get("message")
                    or payload.get("detail")
                    or response.text
                )
            except ValueError:
                payload = None
                message = response.text
            raise APIError(response.status_code, message, payload)
        if not response.content:
            return {}
        return response.json()

    def info(self) -> dict[str, Any]:
        return self._get("/")

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def upload_dataset(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._post(
            "/datasets/upload",
            files={"file": (filename, content, "text/csv")},
        )

    def summary(self, dataset_id: str) -> dict[str, Any]:
        return self._get(f"/datasets/{dataset_id}/summary")

    def clean(self, dataset_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._post(f"/datasets/{dataset_id}/clean", json=body or {})

    def charts(self, dataset_id: str) -> dict[str, Any]:
        return self._get(f"/datasets/{dataset_id}/charts")

    def train(self, dataset_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/datasets/{dataset_id}/train", json=body)

    def insights(self, dataset_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._post(f"/datasets/{dataset_id}/insights", json=body or {})

    def explain(self, model_id: str) -> dict[str, Any]:
        return self._get(f"/models/{model_id}/explain")

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._post("/agent/chat", json=body)

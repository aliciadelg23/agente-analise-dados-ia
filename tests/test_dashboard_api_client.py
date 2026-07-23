"""Unit tests for the dashboard APIClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dashboard.api_client import APIClient, APIError


def _mock_response(status_code: int, json_body: object | None = None, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.content = b"body" if json_body or text else b""
    response.text = text
    if json_body is not None:
        response.json.return_value = json_body
    else:
        response.json.side_effect = ValueError("no json")
    return response


class _StubHttpxClient:
    def __init__(self, response: MagicMock) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict]] = []

    def __enter__(self) -> _StubHttpxClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: object) -> MagicMock:
        self.calls.append((method, url, dict(kwargs)))
        return self.response


def test_base_url_defaults_to_localhost() -> None:
    client = APIClient(base_url=None)
    assert client.base_url.startswith("http")


def test_base_url_reads_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://api.example.com/")

    client = APIClient()

    assert client.base_url == "http://api.example.com"


def test_get_returns_parsed_json_on_success() -> None:
    stub = _StubHttpxClient(_mock_response(200, {"status": "ok"}))
    client = APIClient(base_url="http://api.example.com")

    with patch("dashboard.api_client.httpx.Client", return_value=stub):
        result = client.health()

    assert result == {"status": "ok"}
    method, url, _ = stub.calls[0]
    assert method == "GET"
    assert url.endswith("/health")


def test_post_forwards_json_body() -> None:
    stub = _StubHttpxClient(_mock_response(201, {"model_id": "abc"}))
    client = APIClient(base_url="http://api.example.com")

    with patch("dashboard.api_client.httpx.Client", return_value=stub):
        result = client.train("dataset-1", {"target_column": "churn"})

    assert result == {"model_id": "abc"}
    method, url, kwargs = stub.calls[0]
    assert method == "POST"
    assert url.endswith("/datasets/dataset-1/train")
    assert kwargs.get("json") == {"target_column": "churn"}


def test_upload_forwards_multipart_file() -> None:
    stub = _StubHttpxClient(_mock_response(201, {"dataset_id": "id-1"}))
    client = APIClient(base_url="http://api.example.com")

    with patch("dashboard.api_client.httpx.Client", return_value=stub):
        result = client.upload_dataset("data.csv", b"a,b\n1,2\n")

    assert result == {"dataset_id": "id-1"}
    _, _, kwargs = stub.calls[0]
    assert "files" in kwargs
    filename, content, ctype = kwargs["files"]["file"]
    assert filename == "data.csv"
    assert content == b"a,b\n1,2\n"
    assert ctype == "text/csv"


def test_error_response_raises_api_error_with_message() -> None:
    stub = _StubHttpxClient(
        _mock_response(400, {"error": {"code": "invalid", "message": "Bad request"}})
    )
    client = APIClient(base_url="http://api.example.com")

    with (
        patch("dashboard.api_client.httpx.Client", return_value=stub),
        pytest.raises(APIError) as excinfo,
    ):
        client.health()

    assert excinfo.value.status_code == 400
    assert excinfo.value.message == "Bad request"


def test_error_response_without_json_falls_back_to_text() -> None:
    stub = _StubHttpxClient(_mock_response(500, None, text="internal boom"))
    client = APIClient(base_url="http://api.example.com")

    with (
        patch("dashboard.api_client.httpx.Client", return_value=stub),
        pytest.raises(APIError) as excinfo,
    ):
        client.health()

    assert excinfo.value.status_code == 500
    assert "boom" in excinfo.value.message

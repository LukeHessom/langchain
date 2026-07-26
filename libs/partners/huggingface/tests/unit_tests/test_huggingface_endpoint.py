"""Tests for HuggingFaceEndpoint with local/custom endpoint_url (no HF API calls)."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_huggingface.embeddings.huggingface_endpoint import (
    HuggingFaceEndpointEmbeddings,
)
from langchain_huggingface.llms.huggingface_endpoint import (
    HuggingFaceEndpoint,
    _is_huggingface_hosted_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (None, False),
        ("", False),
        ("http://localhost:8010/", False),
        ("http://127.0.0.1:8080", False),
        ("http://my-tgi.internal/", False),
        ("https://api.inference-api.azure-api.net/", False),
        ("https://abc.huggingface.co/inference", True),
        ("https://xyz.hf.space/", True),
    ],
)
def test_is_huggingface_hosted_url(
    url: str | None,
    expected: bool,  # noqa: FBT001
) -> None:
    """URL helper: local/custom vs HF-hosted."""
    assert _is_huggingface_hosted_url(url) is expected


@patch(
    "huggingface_hub.AsyncInferenceClient",
)
@patch("huggingface_hub.InferenceClient")
def test_local_endpoint_does_not_pass_api_key(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """With a local endpoint_url we don't pass api_key so the client doesn't hit HF."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpoint(  # type: ignore[call-arg]
        endpoint_url="http://localhost:8010/",
        max_new_tokens=64,
    )

    mock_inference_client.assert_called_once()
    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs.get("api_key") is None
    assert call_kwargs.get("model") == "http://localhost:8010/"

    mock_async_client.assert_called_once()
    async_call_kwargs = mock_async_client.call_args[1]
    assert async_call_kwargs.get("api_key") is None


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_hosted_endpoint_keeps_api_key(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """HF-hosted endpoint_url still gets the token."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpoint(  # type: ignore[call-arg]
        endpoint_url="https://abc.huggingface.co/inference",
        max_new_tokens=64,
        huggingfacehub_api_token="hf_xxx",  # noqa: S106
    )

    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs.get("api_key") == "hf_xxx"


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_endpoint_embeddings_uses_custom_endpoint_url(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Embedding clients use a custom endpoint without forwarding an HF token."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpointEmbeddings(
        endpoint_url="http://localhost:8081",
        huggingfacehub_api_token="hf_xxx",  # noqa: S106
    )

    mock_inference_client.assert_called_once()
    call_kwargs = mock_inference_client.call_args[1]
    assert call_kwargs.get("model") == "http://localhost:8081"
    assert call_kwargs.get("token") is None
    assert "base_url" not in call_kwargs

    mock_async_client.assert_called_once()
    async_call_kwargs = mock_async_client.call_args[1]
    assert async_call_kwargs.get("model") == "http://localhost:8081"
    assert async_call_kwargs.get("token") is None
    assert "base_url" not in async_call_kwargs


@pytest.mark.parametrize("field_name", ["model", "repo_id"])
def test_huggingface_endpoint_embeddings_rejects_endpoint_url_with_model(
    field_name: str,
) -> None:
    """Only one model source may be configured."""
    model_kwargs: dict[str, Any] = {
        field_name: "sentence-transformers/all-mpnet-base-v2",
        "endpoint_url": "http://localhost:8081",
    }
    with pytest.raises(ValueError, match="not more than one"):
        HuggingFaceEndpointEmbeddings(**model_kwargs)


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_huggingface_endpoint_embeddings_keeps_token_for_hf_endpoint(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """Official Hugging Face endpoints continue to receive the configured token."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    HuggingFaceEndpointEmbeddings(
        endpoint_url="https://example.huggingface.co/inference",
        huggingfacehub_api_token="hf_xxx",  # noqa: S106
    )

    assert mock_inference_client.call_args[1].get("token") == "hf_xxx"
    assert mock_async_client.call_args[1].get("token") == "hf_xxx"

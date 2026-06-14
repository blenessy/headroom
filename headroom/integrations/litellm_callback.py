"""LiteLLM callback — add Headroom compression to LiteLLM with one line.

    # Local mode (compression runs in-process):
    import litellm
    from headroom.integrations.litellm_callback import HeadroomCallback

    litellm.callbacks = [HeadroomCallback()]

    # Cloud mode (managed CCR, TOIN, analytics via Headroom Cloud):
    litellm.callbacks = [HeadroomCallback(api_key="hdr_xxx")]

Compression runs in LiteLLM's proxy pre-call hook (the proxy is what invokes
``async_pre_call_hook``). Cloud mode requires httpx: pip install httpx
"""

from __future__ import annotations

import json
import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

logger = logging.getLogger(__name__)

# LiteLLM call types this callback compresses (chat + text completions).
# Other call types (embeddings, moderation, etc.) pass through untouched.
_COMPRESSIBLE_CALL_TYPES = frozenset(
    {"completion", "acompletion", "text_completion", "atext_completion"}
)

_DEFAULT_CLOUD_URL = "https://api.headroomlabs.ai"


class HeadroomCallback(CustomLogger):
    """LiteLLM callback that compresses messages before each API call.

    Subclasses LiteLLM's CustomLogger and overrides ``async_pre_call_hook``,
    which the LiteLLM proxy invokes for every request.

    Two modes:
    - Local (default): Compresses in-process using headroom.compress().
    - Cloud (api_key set): Calls Headroom Cloud API for managed compression
      with org-scoped CCR, TOIN learning, and analytics dashboards.

    Usage (local):
        litellm.callbacks = [HeadroomCallback()]

    Usage (cloud):
        litellm.callbacks = [HeadroomCallback(api_key="hdr_xxx")]

    Usage (cloud with LiteLLM proxy config):
        # litellm_config.yaml
        litellm_settings:
          callbacks: [headroom.integrations.litellm_callback.HeadroomCallback]
        environment_variables:
          HEADROOM_API_KEY: "hdr_xxx"
    """

    def __init__(
        self,
        min_tokens: int = 500,
        model_limit: int = 200000,
        hooks: Any = None,
        api_key: str | None = None,
        api_url: str | None = None,
    ) -> None:
        super().__init__()
        self._min_tokens = min_tokens
        self._model_limit = model_limit
        self._hooks = hooks
        self._total_saved = 0

        # Cloud mode: if api_key is set, compress via Headroom Cloud API
        # Falls back to HEADROOM_API_KEY env var
        import os

        self._api_key = api_key or os.environ.get("HEADROOM_API_KEY", "").strip() or None
        self._api_url = (
            api_url or os.environ.get("HEADROOM_API_URL", "").strip() or _DEFAULT_CLOUD_URL
        ).rstrip("/")
        self._client: Any = None  # Lazy-initialized httpx.AsyncClient
        self._comp_results = {}

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all calls."""
        return self._total_saved

    @property
    def cloud_mode(self) -> bool:
        """Whether cloud compression is enabled."""
        return self._api_key is not None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> dict[str, Any] | None:
        """Called by the LiteLLM proxy before each API call. Compresses messages."""

        if call_type not in _COMPRESSIBLE_CALL_TYPES:
            return data

        messages = data.get("messages", [])
        model = data.get("model", "")

        if not messages:
            return data

        try:
            if self._api_key:
                result = await self._cloud_compress(messages, model)
            else:
                result = self._local_compress(messages, model)
            if result and result.get("tokens_saved", 0) > 0 and "messages" in result:
                if "litellm_call_id" in data:
                    self._comp_results[data["litellm_call_id"]] = result
                data["messages"] = result["messages"]
                self._total_saved += result["tokens_saved"]

        except Exception as e:
            logger.warning("Headroom compression failed, using original messages: %s", e)

        return data

    def _local_compress(self, messages: list[dict], model: str) -> dict[str, Any] | None:
        """Compress locally using headroom.compress()."""
        from headroom.compress import compress

        result = compress(
            messages=messages,
            model=model or "claude-sonnet-4-5-20250929",
            model_limit=self._model_limit,
            hooks=self._hooks,
        )
        return {
            "messages": result.messages,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
        }

    async def _cloud_compress(self, messages: list[dict], model: str) -> dict[str, Any] | None:
        """Compress via Headroom Cloud API (managed CCR, TOIN, analytics)."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx is required for Headroom Cloud mode: pip install httpx"
                ) from e
            self._client = httpx.AsyncClient(timeout=30.0)

        client = self._client
        assert client is not None
        resp = await client.post(
            f"{self._api_url}/v1/saas/compress",
            headers={
                "X-Headroom-Key": self._api_key,
                "Content-Type": "application/json",
            },
            content=json.dumps(
                {
                    "messages": messages,
                    "model": model or "claude-sonnet-4-5-20250929",
                    "model_limit": self._model_limit,
                }
            ),
        )

        if resp.status_code != 200:
            logger.warning("Headroom Cloud API error: %d %s", resp.status_code, resp.text[:200])
            return None

        result: dict[str, Any] = resp.json()
        return result

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ):
        call_id = request_data.get("litellm_call_id", "")
        self._comp_results.pop(call_id, None)

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        if not hasattr(response, "_hidden_params"):
            response._hidden_params = {}
        if "additional_headers" not in response._hidden_params:
            response._hidden_params["additional_headers"] = {}
        additional_headers = response._hidden_params["additional_headers"]
        call_id = data.get("litellm_call_id", "")
        result = self._comp_results.pop(call_id, {})
        tokens_before = result.get("tokens_before", 1)
        tokens_after = result.get("tokens_after", 1)
        additional_headers["x-headroom-tokens-saved"] = str(tokens_before - tokens_after)
        additional_headers["x-headroom-compression-ratio"] = str(tokens_before / tokens_after)
        additional_headers["x-headroom-total-saved"] = str(self._total_saved)
        response._hidden_params["additional_headers"] = additional_headers
        return response

    async def async_moderation_hook(  # call made in parallel to llm api call
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ):
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        # print("in async_post_call_streaming_hook")
        pass


proxy_handler_instance = HeadroomCallback()

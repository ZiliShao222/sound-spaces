from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


def image_array_to_data_url(image: np.ndarray) -> str:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3:
        raise RuntimeError(f"Expected image with 3 dimensions; got shape={tuple(array.shape)}")

    with io.BytesIO() as buffer:
        Image.fromarray(array).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _request_json(
    url: str,
    *,
    api_key: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if payload is None:
        request = urllib.request.Request(url, headers=headers, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=max(int(timeout_sec), 1)) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        details = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {err.code} from Qwen service: {details}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Failed to reach Qwen service: {err}") from err

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Invalid JSON response from Qwen service: {raw[:500]}") from err

    if not isinstance(parsed, dict):
        raise RuntimeError("Qwen service response must be a JSON object")
    return parsed


def chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, Any]],
    timeout_sec: int,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    payload = {
        "model": str(model),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": list(messages),
    }
    return _request_json(
        api_base.rstrip("/") + "/chat/completions",
        api_key=api_key,
        payload=payload,
        timeout_sec=timeout_sec,
    )


def extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    message = first.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()

    return ""


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text).strip()
    if not raw:
        raise RuntimeError("Qwen returned empty text")

    candidates: List[str] = [raw]

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match is not None:
        candidates.append(fenced_match.group(1).strip())

    start_index = raw.find("{")
    end_index = raw.rfind("}")
    if start_index >= 0 and end_index > start_index:
        candidates.append(raw[start_index : end_index + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError(f"Failed to parse JSON object from Qwen output: {raw[:500]}")

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def _image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _request_json(url: str, *, api_key: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is None:
        req = urllib.request.Request(url, headers=headers, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for a local OpenAI-compatible Qwen VL API.")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--prompt", type=str, default="Return JSON with keys primary_goal and context_nodes. Keep it short.")
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    models_payload = _request_json(args.api_base.rstrip("/") + "/models", api_key=args.api_key)
    models = models_payload.get("data", [])
    if not models:
        raise RuntimeError("No models returned by /models")

    model_name = args.model or str(models[0]["id"])
    print(f"Using model: {model_name}")

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": args.prompt}]
    if args.image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(args.image)},
            }
        )

    payload = {
        "model": model_name,
        "temperature": 0.1,
        "max_tokens": 256,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a precise structured-output vision-language assistant."}],
            },
            {"role": "user", "content": user_content},
        ],
    }

    completion = _request_json(args.api_base.rstrip("/") + "/chat/completions", api_key=args.api_key, payload=payload)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object root in {path}")
    return payload


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _flatten_instances(raw_instances: Any) -> Dict[str, Dict[str, Any]]:
    flattened: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_instances, dict):
        return flattened

    for key, value in raw_instances.items():
        if isinstance(value, dict) and "semantic_id" in value:
            flattened[str(key)] = value
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict):
                    flattened[str(nested_key)] = nested_value
    return flattened


def _split_instance_key(instance_key: str) -> Tuple[Optional[str], Optional[str]]:
    token = str(instance_key).strip()
    if "_" not in token:
        return None, None
    category, object_id = token.rsplit("_", 1)
    if not category or not object_id:
        return None, None
    return category, object_id


def _score_view(view: Dict[str, Any]) -> float:
    frame_cov = float(view.get("frame_cov", 0.0) or 0.0)
    iou = float(view.get("iou", 0.0) or 0.0)
    detections = float(view.get("yolo_num_detections", 0.0) or 0.0)
    det_score = min(detections, 5.0) / 5.0 * 100.0
    return 0.5 * frame_cov + 0.3 * iou + 0.2 * det_score


def _resolve_instance_image_candidates(
    input_path: Path,
    output_root: Path,
    instance_key: str,
    max_images: int,
) -> List[Tuple[Path, str, float]]:
    category, object_id = _split_instance_key(instance_key)
    if category is None or object_id is None:
        return []

    views_json = input_path.parent / category / object_id / "views.json"
    if not views_json.is_file():
        return []

    try:
        payload = _load_json(views_json)
    except Exception:
        return []

    views = payload.get("views")
    if not isinstance(views, list) or not views:
        return []

    candidates: List[Tuple[Path, str, float]] = []
    for idx, view in enumerate(views):
        if not isinstance(view, dict):
            continue
        rel = view.get("goal_image")
        if not isinstance(rel, str) or not rel.strip():
            continue
        abs_path = (output_root / rel).resolve()
        if not abs_path.is_file():
            continue
        score = _score_view(view)
        candidates.append((abs_path, rel, score))

    candidates.sort(key=lambda item: item[2], reverse=True)
    if max_images > 0:
        candidates = candidates[:max_images]
    return candidates


def _image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _build_system_prompt() -> str:
    return (
        "You are a concise vision-language assistant for embodied navigation datasets. "
        "Output exactly one English sentence only."
    )


def _build_user_prompt(category: str) -> str:
    return (
        "Describe the TARGET object instance shown in these views. "
        "Rules: (1) English only; (2) <=15 words; (3) include target COLOR; "
        "(4) include at least one spatial relation with nearby objects; "
        "(5) mention target category exactly once: "
        f"'{category}'; (6) do not mention material/size/shape; "
        "(7) relation words should come from: next to, near, on, under, between. "
        "(8) do not use viewpoint-sensitive relations such as in front of / behind / left / right. "
        "Output only the sentence, no explanation."
    )


def _extract_response_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
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
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return " ".join(chunks).strip()
    return ""


def _normalize_sentence(text: str) -> str:
    sent = str(text).strip().strip("`\"'")
    sent = " ".join(sent.split())
    if not sent:
        return sent
    if sent.endswith("."):
        return sent
    return sent + "."


def _word_count(text: str) -> int:
    return len([token for token in text.replace(".", " ").split() if token])


def _contains_category(text: str, category: str) -> bool:
    lower = " " + text.lower() + " "
    return f" {category.lower()} " in lower


def _call_qwen_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: Sequence[Path],
    timeout: int,
) -> Dict[str, Any]:
    endpoint = api_base.rstrip("/") + "/chat/completions"

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for image_path in image_paths:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(image_path)},
            }
        )

    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 64,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        details = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {err.code} from Qwen API: {details}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Failed to reach Qwen API: {err}") from err

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Invalid JSON response from Qwen API: {raw[:500]}") from err

    if not isinstance(parsed, dict):
        raise RuntimeError("Qwen API response is not an object")
    return parsed


def _generate_description(
    *,
    api_base: str,
    api_key: str,
    model: str,
    category: str,
    image_paths: Sequence[Path],
    timeout: int,
    retries: int,
) -> Tuple[Optional[str], str]:
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(category)
    last_reason = "empty"

    for _ in range(max(1, retries)):
        resp = _call_qwen_chat_completion(
            api_base=api_base,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=image_paths,
            timeout=timeout,
        )
        text = _normalize_sentence(_extract_response_text(resp))
        if text and _contains_category(text, category):
            return text, "ok"
        if text:
            last_reason = "missing_category"
        else:
            last_reason = "empty"

    return None, last_reason


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_with_description.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate concise English descriptions for image-available instances using qwen3-vl-plus.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input trajectory_dataset.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: *_with_description.json).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for resolving goal_image relative paths (default: input/../..).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-QgWdM03NkfNrFfMA576126F43fAa4b0eBb635d80C6D2Cc91",
        help="Qwen API key. If omitted, uses DASHSCOPE_API_KEY/QWEN_API_KEY/OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.vveai.com/v1",
        help="OpenAI-compatible base URL for Qwen service.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-plus",
        help="Vision-language model name.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=3,
        help="Max number of views per instance sent to the model.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count per instance when output fails constraints.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between API calls.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap for debug runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing descriptions if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Input not found: {input_path}")

    output_path = (args.output or _default_output_path(input_path)).expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else input_path.parent.parent.resolve()
    )

    api_key = (
        args.api_key
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Provide --api-key or set DASHSCOPE_API_KEY/QWEN_API_KEY/OPENAI_API_KEY."
        )

    payload = _load_json(input_path)
    instances = _flatten_instances(payload.get("instances"))
    if not instances:
        raise RuntimeError("No instances found in input dataset.")

    updated = 0
    skipped_no_image = 0
    skipped_existing = 0
    failed = 0

    instance_keys = sorted(instances.keys())
    if isinstance(args.max_instances, int) and args.max_instances > 0:
        instance_keys = instance_keys[: args.max_instances]

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Instances to process: {len(instance_keys)}")
    print(f"Model: {args.model}")

    for idx, instance_key in enumerate(instance_keys):
        instance = instances[instance_key]
        if not isinstance(instance, dict):
            failed += 1
            continue

        image_payload = instance.get("image")
        render_views = image_payload.get("render_views") if isinstance(image_payload, dict) else None
        if not isinstance(render_views, list) or len(render_views) == 0:
            skipped_no_image += 1
            continue

        if not args.overwrite and isinstance(instance.get("description"), str) and instance.get("description").strip():
            skipped_existing += 1
            continue

        category = str(instance.get("category", "object")).strip() or "object"

        candidates = _resolve_instance_image_candidates(
            input_path=input_path,
            output_root=output_root,
            instance_key=instance_key,
            max_images=max(1, int(args.max_images)),
        )
        if len(candidates) == 0:
            instance["description_meta"] = {
                "status": "missing_images",
                "model": args.model,
            }
            failed += 1
            print(f"[{idx+1}/{len(instance_keys)}] {instance_key}: missing image files")
            continue

        image_paths = [item[0] for item in candidates]
        rel_paths = [item[1] for item in candidates]

        try:
            description, status = _generate_description(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                category=category,
                image_paths=image_paths,
                timeout=max(1, int(args.timeout)),
                retries=max(1, int(args.retries)),
            )
        except Exception as exc:
            failed += 1
            instance["description_meta"] = {
                "status": "api_error",
                "model": args.model,
                "error": str(exc),
                "source_images": rel_paths,
            }
            print(f"[{idx+1}/{len(instance_keys)}] {instance_key}: api_error {exc}")
            time.sleep(max(0.0, float(args.sleep)))
            continue

        if description is None:
            failed += 1
            instance["description_meta"] = {
                "status": status,
                "model": args.model,
                "source_images": rel_paths,
            }
            print(f"[{idx+1}/{len(instance_keys)}] {instance_key}: invalid ({status})")
            time.sleep(max(0.0, float(args.sleep)))
            continue

        instance["description"] = description
        modalities = instance.get("modalities")
        if isinstance(modalities, list):
            if "description" not in modalities:
                modalities.append("description")
        else:
            instance["modalities"] = ["description"]

        instance["description_meta"] = {
            "status": "ok",
            "model": args.model,
            "word_count": _word_count(description),
            "source_images": rel_paths,
            "num_source_views": len(rel_paths),
        }

        updated += 1
        print(f"[{idx+1}/{len(instance_keys)}] {instance_key}: {description}")
        time.sleep(max(0.0, float(args.sleep)))

    _dump_json(output_path, payload)

    print("\nDone.")
    print(f"Updated descriptions: {updated}")
    print(f"Skipped (no image): {skipped_no_image}")
    print(f"Skipped (existing): {skipped_existing}")
    print(f"Failed: {failed}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

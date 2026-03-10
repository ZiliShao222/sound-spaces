#!/usr/bin/env python3

"""Standalone YOLO demo for object detection on SoundSpaces-rendered images.

Example:
    python scripts/yolo26_demo.py \
        --input output/QUCTc6BB5sX/tv_monitor/85 \
        --model /path/to/yolo26.pt \
        --output-dir output/yolo26_demo \
        --target-classes tv,tv_monitor,chair
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    for token in ("/", "\\", "_", "-"):
        normalized = normalized.replace(token, " ")
    return " ".join(normalized.split())


def deduplicate(values: Iterable[str]) -> List[str]:
    results: List[str] = []
    seen = set()
    for value in values:
        normalized = normalize_label(value)
        if not normalized or normalized in seen:
            continue
        results.append(normalized)
        seen.add(normalized)
    return results


def load_aliases(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    alias_path = path.expanduser().resolve()
    with alias_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Alias JSON must be an object: {alias_path}")

    aliases: Dict[str, List[str]] = {}
    for raw_key, raw_value in payload.items():
        if isinstance(raw_value, str):
            values = [raw_value]
        elif isinstance(raw_value, list):
            values = [str(item) for item in raw_value]
        else:
            raise RuntimeError(
                f"Alias value for '{raw_key}' must be a string or list: {alias_path}"
            )
        aliases[normalize_label(raw_key)] = deduplicate([raw_key, *values])
    return aliases


def target_aliases_for(
    target_label: str,
    aliases: Mapping[str, Sequence[str]],
) -> List[str]:
    normalized = normalize_label(target_label)
    return deduplicate([normalized, *(aliases.get(normalized, []))])


def matches_target(
    class_name: str,
    target_labels: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
) -> bool:
    detected = normalize_label(class_name)
    for target_label in target_labels:
        for alias in target_aliases_for(target_label, aliases):
            if detected == alias or detected in alias or alias in detected:
                return True
    return False


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(value), 3) for value in self.bbox_xyxy],
        }


def collect_images(input_path: Path) -> List[Path]:
    resolved = input_path.expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(f"Input path does not exist: {resolved}")
    if resolved.is_file():
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            raise RuntimeError(f"Unsupported image file: {resolved}")
        return [resolved]

    images = [
        path
        for path in sorted(resolved.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"No images found under: {resolved}")
    return images


def load_yolo(model_path: Union[str, Path]):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "`ultralytics` is not installed in the current environment. "
            "Please install it in your `ss` virtual environment first, then rerun this demo."
        ) from exc

    return YOLO(str(model_path))


def image_to_rgb_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def ensure_rgb_array(image: np.ndarray) -> np.ndarray:
    """Normalize arbitrary image tensors into uint8 HxWx3 RGB arrays."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise RuntimeError(f"Expected HxWxC image array, got shape {array.shape}")

    if array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] != 3:
        raise RuntimeError(f"Unsupported channel count for YOLO inference: {array.shape[2]}")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def run_inference(
    model: Any,
    image: np.ndarray,
    *,
    device: Optional[str],
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> List[Detection]:
    image = ensure_rgb_array(image)
    predict_kwargs: Dict[str, Any] = {
        "source": image,
        "conf": float(conf_threshold),
        "iou": float(iou_threshold),
        "max_det": int(max_det),
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", {})

    xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
    conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes, "conf") else np.empty((0,))
    cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes, "cls") else np.empty((0,))

    detections: List[Detection] = []
    for index, bbox in enumerate(xyxy):
        class_id = int(cls[index]) if index < len(cls) else -1
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id))
        elif isinstance(names, list) and 0 <= class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = str(class_id)

        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(conf[index]) if index < len(conf) else 0.0,
                bbox_xyxy=tuple(float(value) for value in bbox[:4]),
            )
        )

    detections.sort(key=lambda detection: float(detection.confidence), reverse=True)
    return detections


def annotate_image(
    image: np.ndarray,
    detections: Sequence[Detection],
    *,
    target_labels: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
) -> Image.Image:
    rendered = Image.fromarray(image)
    draw = ImageDraw.Draw(rendered)
    for detection in detections:
        matched = matches_target(detection.class_name, target_labels, aliases)
        color = (60, 220, 100) if matched else (255, 196, 0)
        x1, y1, x2, y2 = detection.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_y = max(0.0, y1 - 18.0)
        try:
            left, top, right, bottom = draw.textbbox((x1, text_y), label)
            draw.rectangle([left - 2, top - 2, right + 2, bottom + 2], fill=color)
        except Exception:
            pass
        draw.text((x1, text_y), label, fill=(0, 0, 0))
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone YOLO26-style object detection demo for SoundSpaces images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Image file or directory containing images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to local YOLO/YOLO26 weights, e.g. /path/to/yolo26.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/yolo26_demo"),
        help="Directory used to save annotated images and the summary JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, e.g. cpu, 0, cuda:0.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=50,
        help="Maximum detections per image.",
    )
    parser.add_argument(
        "--target-classes",
        type=str,
        default="",
        help="Optional comma-separated target classes to highlight, e.g. chair,sofa,tv_monitor.",
    )
    parser.add_argument(
        "--aliases-json",
        type=Path,
        default=None,
        help="Optional JSON mapping from target labels to model class-name aliases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = collect_images(args.input)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_yolo(args.model)
    aliases = load_aliases(args.aliases_json)
    target_labels = deduplicate(args.target_classes.split(","))

    input_root = args.input.expanduser().resolve()
    if input_root.is_file():
        input_root = input_root.parent

    results: List[Dict[str, Any]] = []
    for image_path in images:
        rgb = image_to_rgb_array(image_path)
        detections = run_inference(
            model,
            rgb,
            device=args.device,
            conf_threshold=float(args.conf_threshold),
            iou_threshold=float(args.iou_threshold),
            max_det=int(args.max_det),
        )

        try:
            relative_image = image_path.relative_to(input_root)
        except ValueError:
            relative_image = image_path.name

        annotated_path = output_dir / relative_image
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        annotate_image(
            rgb,
            detections,
            target_labels=target_labels,
            aliases=aliases,
        ).save(annotated_path)

        matched_detections = [
            detection.to_dict()
            for detection in detections
            if matches_target(detection.class_name, target_labels, aliases)
        ]
        results.append(
            {
                "image": str(image_path),
                "annotated_image": str(annotated_path),
                "image_size": {
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                },
                "num_detections": int(len(detections)),
                "matched_target_detections": matched_detections,
                "detections": [detection.to_dict() for detection in detections],
            }
        )
        print(
            f"[YOLO26 demo] {image_path.name}: {len(detections)} detections, "
            f"matched targets = {len(matched_detections)}"
        )

    summary = {
        "model": str(args.model),
        "input": str(args.input),
        "output_dir": str(output_dir),
        "device": args.device,
        "conf_threshold": float(args.conf_threshold),
        "iou_threshold": float(args.iou_threshold),
        "max_det": int(args.max_det),
        "target_classes": target_labels,
        "aliases_json": str(args.aliases_json) if args.aliases_json is not None else None,
        "num_images": int(len(results)),
        "results": results,
    }
    summary_path = output_dir / "detections.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()

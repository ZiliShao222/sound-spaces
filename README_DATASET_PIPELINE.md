# SoundSpaces 数据集构建流水线（从 0 到 `.json.gz`）

这份文档给一条完整主线：

1. 从场景导出可用 instance（含可渲染图像筛选）
2. 基于 `valid_instances.json` 采样多目标 episode
3. 打包为 `SemanticAudioNav` 使用的 `.json.gz`
4. （可选）可视化与评测

---

## 0. 环境与前置条件

- 仓库根目录：`/home/RufengChen/sound-spaces`
- 推荐环境：`conda activate ss`
- 场景文件存在：`data/scene_datasets/mp3d/<scene>/<scene>.glb`
- scene dataset config 存在：`data/scene_datasets/mp3d/mp3d.scene_dataset_config.json`
- YOLO 权重存在：`models/yolo26x.pt`

建议先在仓库根目录执行：

```bash
pwd
conda activate ss
```

---

## 1. 一条命令变量（推荐先设）

```bash
SCENE=QUCTc6BB5sX
OUT=output/${SCENE}
mkdir -p "${OUT}"
```

---

## 2. 导出 valid instances（含图像筛选）

> 默认逻辑已经是：`yolo26x` + `conf >= 0.7`，并且传感器默认 `512x512, hfov=90`。

```bash
python scripts/export_valid_instances.py "${SCENE}" \
  --output "${OUT}/valid_instances.json" \
  --image-report "${OUT}/invalid_image_instances.json" \
  --output-root output \
  --save-images \
  --yolo-model models/yolo26x.pt \
  --yolo-conf-threshold 0.7 \
  --width 512 --height 512 --hfov 90
```

主要输出：

- `${OUT}/valid_instances.json`
- `${OUT}/invalid_image_instances.json`

快速检查：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path("output/QUCTc6BB5sX/valid_instances.json")
d=json.loads(p.read_text())
print("scene:", d.get("scene_name"))
print("categories:", len(d.get("instances", {})))
print("num_valid_instances:", d.get("num_valid_instances"))
PY
```

---

## 3. 从 valid instances 构建轨迹数据集

```bash
python scripts/build_trajectories_from_valid_instances.py \
  --input "${OUT}/valid_instances.json" \
  --output "${OUT}/trajectory_dataset.json" \
  --num-trajectories 500 \
  --min-goals 2 \
  --max-goals 5 \
  --seed 42 \
  --sim-start \
  --min-goal-distance 4.0
```

说明：

- `--sim-start`：起点从仿真器 navmesh floor 采样。
- `--min-goal-distance 4.0`：约束起点到目标、目标到目标的接地距离。
- 结果文件：`${OUT}/trajectory_dataset.json`

当前输出顶层结构（简化）：

- `dataset / version / scene_name / scene_id`
- `sampling`
- `episodes`
- `instances`

其中每个 `episode.goals` 当前格式是二维列表，例如：

```json
[
  ["chair_579", "image_0"],
  ["sink_446", "object"]
]
```

含义：`[instance_key, modality_token]`。

---

## 4. 打包成 `.json.gz`

```bash
python scripts/pack_semantic_audionav_json_gz.py \
  --input "${OUT}/trajectory_dataset.json" \
  --output "${OUT}/trajectory_dataset.json.gz"
```

输出：

- `${OUT}/trajectory_dataset.json.gz`

快速查看 gzip 是否可读：

```bash
python - <<'PY'
import gzip, json
p="output/QUCTc6BB5sX/trajectory_dataset.json.gz"
with gzip.open(p, "rt", encoding="utf-8") as f:
    d=json.load(f)
print("scene:", d.get("scene"))
print("episodes:", len(d.get("episodes", [])))
PY
```

---

## 5. （可选）评测脚本调用示例

`semantic_audionav_eval.py` 通常需要在你已经正确配置 Habitat/SoundSpaces 环境后运行。

```bash
python scripts/semantic_audionav_eval.py \
  --split val \
  --dataset-path "${OUT}/trajectory_dataset.json.gz" \
  --scene "${SCENE}" \
  --num-episodes 10
```

如果需要指定 scene 根目录或 scene dataset config，可追加：

- `--scenes-dir ...`
- `--scene-dataset-config ...`

---

## 6. （可选）把全部 instance 画到 GLB（按类别着色）

```bash
python scripts/render_valid_instances_glb.py \
  --input "${OUT}/valid_instances.json" \
  --scene-name "${SCENE}" \
  --output "${OUT}/valid_instances_bbox.glb"
```

---

## 7. 常见问题

### 7.1 ALSA 报警（`cannot find card '0'`）

通常是无声卡环境的告警，很多情况下可忽略，不影响 JSON 生成。

### 7.2 `Failed to satisfy min-distance constraints ...`

表示当前采样约束过严。可尝试：

- 降低 `--min-goal-distance`（例如 `3.0`）
- 增大 `--distance-max-attempts`
- 减少 `--num-trajectories` 或减小 `--max-goals`

### 7.3 模块缺失（如 `ModuleNotFoundError: habitat`）

确认已进入正确环境（例如 `conda activate ss`），并确保该环境已安装 Habitat/SoundSpaces 依赖。

---

## 8. 最短“全流程”命令清单（可直接复制）

```bash
conda activate ss
SCENE=QUCTc6BB5sX
OUT=output/${SCENE}
mkdir -p "${OUT}"

python scripts/export_valid_instances.py "${SCENE}" \
  --output "${OUT}/valid_instances.json" \
  --image-report "${OUT}/invalid_image_instances.json" \
  --output-root output --save-images \
  --yolo-model models/yolo26x.pt --yolo-conf-threshold 0.7 \
  --width 512 --height 512 --hfov 90

python scripts/build_trajectories_from_valid_instances.py \
  --input "${OUT}/valid_instances.json" \
  --output "${OUT}/trajectory_dataset.json" \
  --num-trajectories 500 --min-goals 2 --max-goals 5 --seed 42 \
  --sim-start --min-goal-distance 4.0

python scripts/pack_semantic_audionav_json_gz.py \
  --input "${OUT}/trajectory_dataset.json" \
  --output "${OUT}/trajectory_dataset.json.gz"
```


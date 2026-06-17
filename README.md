# wakareeru-inference

`wakareeru-inference` 是 `wakareeru` 项目的 serverless 推理后端。当前目标是作为 RunPod / Azure 等 serverless 平台的 worker 代码运行：平台负责接收 HTTP 请求并调用本仓库提供的 handler，本仓库负责加载本地模型并完成完整图片识别。

## 快速启动

安装本仓库：

```bash
pip install -e .
```

本地 smoke test：

```bash
python scripts/smoke_test_handler.py \
  --image /path/to/test.jpg \
  --config configs/service_config.yaml \
  --top-k 5
```

## 依赖

当前依赖写在 `pyproject.toml`。其中 `wakareeru` 主仓库通过 Git dependency 安装，因为分类模型定义、加载和 crop 分类逻辑来自主仓库的 `model_core`。

如果 serverless / GPU 平台镜像已经提供 CUDA 版 `torch` 和 `torchvision`，不要用普通 `pip install -e .` 重新解析并覆盖 PyTorch。镜像构建时使用：

```bash
pip install -r requirements-image.txt
pip install --no-deps -e .
```

`requirements-image.txt` 只包含平台镜像需要额外安装的非 PyTorch 运行依赖；`torch` / `torchvision` 由基础镜像负责。

当前配置使用 `main` 分支：

```toml
"wakareeru @ git+https://github.com/SniperPigeon/wakareeru.git@main"
```

正式发布时建议改成固定 tag，例如：

```toml
"wakareeru @ git+https://github.com/SniperPigeon/wakareeru.git@v0.1.0"
```

目前尚无 Dockerfile，Docker / serverless 镜像将在之后补齐。

## 模型准备

本仓库不会自动下载或拉取模型。需要自行把模型放到 `configs/service_config.yaml` 中配置的本地路径。

模型放置在`CloudFlare r2`的`models`桶内，不公开。若获得Endpoint及Keys请自行下载。

默认配置：

```yaml
detector:
  model_path: "models/grounding-dino"

classifier:
  model_dir: "models/wakareeru-0.1.0-alpha.1"
```

期望目录：

```text
models/
  grounding-dino/
    config.json
    model.safetensors
    preprocessor_config.json
    tokenizer.json
    tokenizer_config.json
    vocab.txt
    ...
  wakareeru-0.1.0-alpha.1/
    backbone/
    model_config.json
    labels.json
    classifier.safetensors
    processor/
    manifest.json
```

Wakareeru 分类模型由主仓库运行 `python -m trainer.export_inference_model` 导出。分类 artifact 必须是完整本地目录：`model_core.load_classifier` 会从 `classifier.model_dir` 读取本地 `backbone/`、`processor/` 和 `classifier.safetensors`，缺失时直接报错，不回退到 Hugging Face cache 或联网下载。

分类输入尺寸以 artifact 内 `model_config.json` 的 `image_size` 为准；主仓库导出时会同步 `processor/preprocessor_config.json` 的默认 `size` / `crop_size`。本仓库不要另行硬编码分类 resize/crop 尺寸。

## Serverless 入口

handler 入口：

```text
wakareeru_inference.handler.handler
```

冷启动行为：

1. 读取 `configs/service_config.yaml`，或读取环境变量 `WAKAREERU_SERVICE_CONFIG` 指定的配置文件。
2. 从 `detector.model_path` 加载本地 Grounding-DINO。
3. 从 `classifier.model_dir` 加载本地 Wakareeru 分类模型 artifact。
4. 在 worker 生命周期内复用已加载模型处理后续请求。

## API 端点

当前仓库暂无自建 HTTP API endpoint，例如 FastAPI / Flask。

API endpoint 由 serverless 平台负责。例如 RunPod / Azure 接收到外部 HTTP 请求后，将请求转换为 event 并调用：

```python
handler(event)
```

如果之后要部署到普通 VM、Container App 或 Kubernetes，再补 HTTP API 层。

## 输入格式

目前仅支持 base64 图片输入。字段名固定为 `input.image_base64`。

请求示例：

```json
{
  "input": {
    "image_base64": "/9j/4AAQSkZJRgABAQ...",
    "top_k": 5
  }
}
```

也支持 data URL：

```json
{
  "input": {
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
    "top_k": 5
  }
}
```

字段说明：

- `image_base64`：必填，base64 编码图片。
- `top_k`：可选，返回每个主体的前 K 个分类结果；缺省使用 `configs/service_config.yaml` 中的 `classifier.top_k`。

## 输出格式

状态字段枚举：

- 顶层 `status`：`ok`、`no_detection`、`error`。
- `detection.status`：`detected`、`fallback_whole_image`。
- `classification.status`：`classified`、`low_confidence`、`no_prediction`。

成功响应：

```json
{
  "status": "ok",
  "subject_count": 1,
  "subjects": [
    {
      "index": 0,
      "detection": {
        "status": "detected",
        "bbox": [120, 80, 900, 520],
        "score": 0.74,
        "label": "a train"
      },
      "classification": {
        "top_prediction": {
          "label_id": 0,
          "label": "101系",
          "probability": 0.8
        },
        "top_k": [
          {
            "label_id": 0,
            "label": "101系",
            "probability": 0.8
          }
        ],
        "confusion_group": null,
        "group_candidates": []
      }
    }
  ]
}
```

如果 GDINO 没有检测到主体，并且配置允许整图 fallback：

```json
{
  "status": "ok",
  "subject_count": 1,
  "subjects": [
    {
      "index": 0,
      "detection": {
        "status": "fallback_whole_image",
        "bbox": null,
        "score": null,
        "label": null
      },
      "classification": {
        "top_prediction": {
          "label_id": 0,
          "label": "101系",
          "probability": 0.8
        },
        "top_k": [],
        "confusion_group": null,
        "group_candidates": []
      }
    }
  ]
}
```

如果配置为检测失败即返回失败：

```json
{
  "status": "no_detection",
  "subjects": []
}
```

错误响应：

```json
{
  "status": "error",
  "error": {
    "type": "ValueError",
    "message": "Request input must contain input.image_base64"
  }
}
```

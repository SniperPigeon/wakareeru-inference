# wakareeru-inference

Serverless inference service for Wakareeru.

Handler entrypoint:

```text
wakareeru_inference.handler.handler
```

Cold start behavior:

1. Read `configs/service_config.yaml`, or `WAKAREERU_SERVICE_CONFIG` when set.
2. Load local Grounding-DINO from `detector.model_path`.
3. Load local Wakareeru classifier artifact from `classifier.model_dir`.
4. Keep both models resident for endpoint calls.

Request shape:

```json
{
  "input": {
    "image": "<base64 image or data URL>",
    "top_k": 5
  }
}
```

Response shape:

```json
{
  "status": "ok",
  "subject_count": 1,
  "subjects": [
    {
      "index": 0,
      "detection": {
        "status": "detected",
        "bbox": [0, 0, 100, 100],
        "score": 0.9,
        "label": "a train"
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
Inference backend for project 'wakareeru'

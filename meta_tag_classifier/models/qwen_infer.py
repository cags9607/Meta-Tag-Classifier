from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


_CONFIG_FILENAMES = [
    "model_config.json",
    "training_metadata.json",
    "label_mapping.json",
    "meta.json",
]

_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
_LABEL_MAPPING_FILENAME = "label_mapping.json"


@dataclass
class QwenClassifierConfig:
    artifacts_dir: Path
    base_model_name: str
    adapter_dir: Path
    max_length: int = 512
    batch_size: int = 8
    load_in_4bit: bool = True
    id2label: Dict[int, str] | None = None
    label2id: Dict[str, int] | None = None
    label_mapping_path: Path | None = None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding = "utf-8"))


def _split_hf_repo_and_subfolder(raw: str) -> Tuple[str, str | None]:
    """
    Parse HuggingFace artifact references.

    Supported examples:
      - Trinotrotolueno/meta-tag-classifier
      - Trinotrotolueno/meta-tag-classifier/template_classifier
      - hf://Trinotrotolueno/meta-tag-classifier/template_classifier

    HuggingFace repo ids are owner/repo. Any additional path components are
    treated as a subfolder inside that model repo.
    """
    if raw.startswith("hf://"):
        raw = raw.replace("hf://", "", 1)

    parts = [part for part in raw.split("/") if part]

    if len(parts) < 2:
        raise ValueError(
            "HuggingFace artifact references must be owner/repo or owner/repo/subfolder. "
            f"Received: {raw}"
        )

    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) if len(parts) > 2 else None

    return repo_id, subfolder


def _resolve_artifacts_dir(artifacts_dir: str | Path) -> Path:
    """
    Resolve a local artifact directory or a HuggingFace Hub model repo.

    Accepted examples:
      - meta_tag_classifier/artifacts
      - /content/drive/MyDrive/Models/my_model
      - Trinotrotolueno/meta-tag-classifier
      - Trinotrotolueno/meta-tag-classifier/template_classifier

    For a HuggingFace Hub repo, the repo is downloaded through the local
    HuggingFace cache using huggingface_hub.snapshot_download. Authentication
    can be provided with the usual HF_TOKEN/HUGGINGFACE_HUB_TOKEN env vars.
    """
    raw = str(artifacts_dir).strip()
    path = Path(raw)

    if path.exists():
        return path

    if raw.startswith("hf://") or ("/" in raw and not raw.startswith("/")):
        repo_id, subfolder = _split_hf_repo_and_subfolder(raw)
    else:
        raise FileNotFoundError(f"Artifacts directory does not exist: {artifacts_dir}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ImportError(
            "Could not resolve artifacts as a local path, and huggingface_hub is not installed. "
            "Install huggingface_hub or pass a local ARTIFACTS_DIR."
        ) from exc

    if subfolder:
        allow_patterns = [
            f"{subfolder}/*.json",
            f"{subfolder}/*.safetensors",
            f"{subfolder}/*.bin",
            f"{subfolder}/*.model",
            f"{subfolder}/*.txt",
            f"{subfolder}/*.jinja",
            f"{subfolder}/adapter/*",
            f"{subfolder}/tokenizer*",
            f"{subfolder}/vocab*",
            f"{subfolder}/merges*",
            f"{subfolder}/special_tokens_map.json",
        ]
    else:
        allow_patterns = [
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.model",
            "*.txt",
            "*.jinja",
            "adapter/*",
            "tokenizer*",
            "vocab*",
            "merges*",
            "special_tokens_map.json",
        ]

    snapshot_root = Path(
        snapshot_download(
            repo_id = repo_id,
            repo_type = "model",
            allow_patterns = allow_patterns,
        )
    )

    if subfolder:
        artifact_subdir = snapshot_root / subfolder

        if not artifact_subdir.exists():
            raise FileNotFoundError(
                f"Downloaded HuggingFace repo {repo_id}, but subfolder was not found: {subfolder}"
            )

        return artifact_subdir

    return snapshot_root


def _find_config_path(artifacts_dir: Path) -> Path:
    for filename in _CONFIG_FILENAMES:
        path = artifacts_dir / filename
        if path.exists():
            return path

    adapter_dir = _find_adapter_dir(artifacts_dir, raw = {})
    label_mapping_path = _find_label_mapping_path(artifacts_dir, adapter_dir)

    if label_mapping_path is not None:
        return label_mapping_path

    raise FileNotFoundError(
        "Could not find a Qwen artifact config file. Expected one of: "
        f"{', '.join(_CONFIG_FILENAMES)} under {artifacts_dir}. "
        "For HuggingFace-style repos, include label_mapping.json next to adapter_config.json."
    )


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_id2label(raw: Any) -> Dict[int, str]:
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("Qwen artifact config must contain a non-empty id2label dictionary.")

    return {
        int(k): str(v)
        for k, v in raw.items()
    }


def _normalize_label2id(raw: Any, id2label: Dict[int, str]) -> Dict[str, int]:
    if isinstance(raw, dict) and raw:
        return {
            str(k): int(v)
            for k, v in raw.items()
        }

    return {
        label: idx
        for idx, label in id2label.items()
    }


def _find_adapter_dir(artifacts_dir: Path, raw: Dict[str, Any]) -> Path:
    candidates: List[Path] = []

    adapter_value = raw.get("adapter_dir")
    if adapter_value:
        adapter_path = Path(str(adapter_value))
        candidates.append(adapter_path if adapter_path.is_absolute() else artifacts_dir / adapter_path)

    # Packaged repo layout.
    candidates.append(artifacts_dir / "adapter")

    # HuggingFace Hub layout: adapter_config.json and adapter_model.safetensors at repo root.
    candidates.append(artifacts_dir)

    # Colab metadata may contain /content/.../adapter, which is invalid after download/copy.
    if adapter_value and Path(str(adapter_value)).name == "adapter":
        candidates.append(artifacts_dir / "adapter")

    for candidate in candidates:
        if (candidate / _ADAPTER_CONFIG_FILENAME).exists():
            return candidate

    # Return first candidate for a more useful error message downstream.
    return candidates[0] if candidates else artifacts_dir / "adapter"


def _find_label_mapping_path(artifacts_dir: Path, adapter_dir: Path | None = None) -> Path | None:
    candidates = [
        artifacts_dir / _LABEL_MAPPING_FILENAME,
    ]

    if adapter_dir is not None:
        candidates.append(adapter_dir / _LABEL_MAPPING_FILENAME)

    for path in candidates:
        if path.exists():
            return path

    return None


def _load_label_mapping(
    raw: Dict[str, Any],
    artifacts_dir: Path,
    adapter_dir: Path,
) -> Tuple[Dict[str, int], Dict[int, str], Path | None]:
    """
    Prefer label_mapping.json when available, because this is the artifact layout
    used by HuggingFace model repos in this project. Fall back to label2id/id2label
    in training_metadata.json/model_config.json.
    """
    label_mapping_path = _find_label_mapping_path(artifacts_dir, adapter_dir)

    if label_mapping_path is not None:
        mapping = _read_json(label_mapping_path)
        id2label = _normalize_id2label(mapping.get("id2label"))
        label2id = _normalize_label2id(mapping.get("label2id"), id2label)
        return label2id, id2label, label_mapping_path

    id2label = _normalize_id2label(raw.get("id2label"))
    label2id = _normalize_label2id(raw.get("label2id"), id2label)
    return label2id, id2label, None


def _load_adapter_config(adapter_dir: Path) -> Dict[str, Any]:
    path = adapter_dir / _ADAPTER_CONFIG_FILENAME
    if not path.exists():
        return {}
    return _read_json(path)


def load_qwen_config(
    artifacts_dir: str | Path,
    *,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    load_in_4bit: Optional[bool] = None,
) -> QwenClassifierConfig:
    artifacts_dir = _resolve_artifacts_dir(artifacts_dir)
    config_path = _find_config_path(artifacts_dir)
    raw = _read_json(config_path)

    adapter_dir = _find_adapter_dir(artifacts_dir, raw)

    if not adapter_dir.exists():
        raise FileNotFoundError(
            "Qwen adapter directory was not found. Expected adapter files either under: "
            f"{artifacts_dir / 'adapter'} or at the HuggingFace snapshot root {artifacts_dir}."
        )

    if not (adapter_dir / _ADAPTER_CONFIG_FILENAME).exists():
        raise FileNotFoundError(
            "Qwen adapter_config.json was not found. Expected it under: "
            f"{adapter_dir}"
        )

    adapter_config = _load_adapter_config(adapter_dir)

    base_model_name = (
        raw.get("base_model_name")
        or raw.get("model_name")
        or raw.get("pretrained_model_name_or_path")
        or adapter_config.get("base_model_name_or_path")
    )

    if not base_model_name:
        raise ValueError(
            f"{config_path} or {adapter_dir / _ADAPTER_CONFIG_FILENAME} must define the base model name. "
            "Expected one of base_model_name, model_name, pretrained_model_name_or_path, "
            "or adapter_config.base_model_name_or_path."
        )

    label2id, id2label, label_mapping_path = _load_label_mapping(raw, artifacts_dir, adapter_dir)

    declared_num_labels = raw.get("num_labels")
    if declared_num_labels is not None and int(declared_num_labels) != len(id2label):
        raise ValueError(
            f"num_labels={declared_num_labels} does not match len(id2label)={len(id2label)}."
        )

    env_batch_size = os.getenv("INFERENCE_BATCH_SIZE")
    env_max_length = os.getenv("MAX_LENGTH")
    env_load_in_4bit = os.getenv("LOAD_IN_4BIT")

    final_batch_size = int(
        batch_size
        if batch_size is not None
        else env_batch_size
        if env_batch_size is not None
        else raw.get("inference_batch_size", 8)
    )

    final_max_length = int(
        max_length
        if max_length is not None
        else env_max_length
        if env_max_length is not None
        else raw.get("max_length", 512)
    )

    final_load_in_4bit = _coerce_bool(
        load_in_4bit
        if load_in_4bit is not None
        else env_load_in_4bit
        if env_load_in_4bit is not None
        else raw.get("load_in_4bit"),
        default = True,
    )

    return QwenClassifierConfig(
        artifacts_dir = artifacts_dir,
        base_model_name = str(base_model_name),
        adapter_dir = adapter_dir,
        max_length = final_max_length,
        batch_size = final_batch_size,
        load_in_4bit = final_load_in_4bit,
        id2label = id2label,
        label2id = label2id,
        label_mapping_path = label_mapping_path,
    )


class QwenSequenceClassifier:
    """
    Batched inference wrapper for a Qwen sequence-classification LoRA adapter.

    Supported artifact layouts:

        Packaged repo layout:
            artifacts/
              model_config.json OR training_metadata.json
              label_mapping.json
              adapter/
                adapter_config.json
                adapter_model.safetensors
                tokenizer files...

        HuggingFace Hub snapshot layout:
            snapshot_root/
              adapter_config.json
              adapter_model.safetensors
              tokenizer files...
              label_mapping.json

    ARTIFACTS_DIR may also be a HuggingFace repo id, e.g. "owner/model_repo".
    """

    def __init__(
        self,
        artifacts_dir: str | Path,
        *,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        load_in_4bit: Optional[bool] = None,
    ):
        self.config = load_qwen_config(
            artifacts_dir,
            batch_size = batch_size,
            max_length = max_length,
            load_in_4bit = load_in_4bit,
        )

        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as exc:
            raise ImportError(
                "Qwen inference requires torch, transformers, peft, accelerate, bitsandbytes, and huggingface_hub. "
                "Install the repo requirements before loading the Qwen classifier."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.adapter_dir,
            use_fast = True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        model_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.base_model_name,
            "num_labels": len(self.config.id2label or {}),
            "id2label": self.config.id2label,
            "label2id": self.config.label2id,
            "torch_dtype": torch_dtype,
        }

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

            if self.config.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_quant_type = "nf4",
                    bnb_4bit_use_double_quant = True,
                    bnb_4bit_compute_dtype = torch_dtype,
                )
        else:
            model_kwargs["device_map"] = None

        base_model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        base_model.config.problem_type = "single_label_classification"

        self.model = PeftModel.from_pretrained(
            base_model,
            self.config.adapter_dir,
        )
        self.model.eval()

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def predict_texts(
        self,
        texts: Iterable[str],
        *,
        batch_size: Optional[int] = None,
        return_proba_matrix: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        texts = [str(x).strip() if x is not None else "" for x in texts]
        n = len(texts)

        labels = np.array([None] * n, dtype = object)
        confidences = np.full(n, np.nan, dtype = float)
        proba_rows: List[np.ndarray] = []

        if n == 0:
            if return_proba_matrix:
                return labels, confidences, np.empty((0, len(self.config.id2label or {})))
            return labels, confidences

        final_batch_size = int(batch_size or self.config.batch_size)
        id2label = self.config.id2label or {}

        with self.torch.no_grad():
            for start in range(0, n, final_batch_size):
                end = min(start + final_batch_size, n)
                batch_texts = texts[start:end]

                encoded = self.tokenizer(
                    batch_texts,
                    truncation = True,
                    max_length = self.config.max_length,
                    padding = True,
                    return_tensors = "pt",
                )

                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }

                outputs = self.model(**encoded)
                probs = self.torch.softmax(outputs.logits.float(), dim = -1).detach().cpu().numpy()
                pred_ids = probs.argmax(axis = 1)

                labels[start:end] = [id2label[int(i)] for i in pred_ids]
                confidences[start:end] = probs.max(axis = 1)

                if return_proba_matrix:
                    proba_rows.append(probs)

        if return_proba_matrix:
            return labels, confidences, np.vstack(proba_rows) if proba_rows else np.empty((0, len(id2label)))

        return labels, confidences

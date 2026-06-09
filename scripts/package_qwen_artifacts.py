#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_LABEL_MAPPING = {
    "mode": "multiclass",
    "label_col": "llm_label",
    "num_labels": 8,
    "label2id": {
        "App Download Portal": 0,
        "Content Farm": 1,
        "News Scraper": 2,
        "Novels or Short Stories": 3,
        "Online Games & Trivia": 4,
        "Other": 5,
        "Parked/Holding": 6,
        "Tool": 7,
    },
    "id2label": {
        "0": "App Download Portal",
        "1": "Content Farm",
        "2": "News Scraper",
        "3": "Novels or Short Stories",
        "4": "Online Games & Trivia",
        "5": "Other",
        "6": "Parked/Holding",
        "7": "Tool",
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding = "utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(
        json.dumps(data, indent = 2, ensure_ascii = False),
        encoding = "utf-8",
    )


def find_adapter_source(trained_dir: Path) -> Path:
    """
    Support both layouts:
      1. Colab output: trained_dir/adapter/adapter_config.json
      2. HuggingFace repo snapshot: trained_dir/adapter_config.json
    """
    candidates = [
        trained_dir / "adapter",
        trained_dir,
    ]

    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find adapter_config.json. Expected either trained-dir/adapter/ "
        f"or a HuggingFace-style root folder at {trained_dir}."
    )


def load_label_mapping(trained_dir: Path, adapter_source: Path) -> Dict[str, Any]:
    mapping_candidates = [
        trained_dir / "label_mapping.json",
        adapter_source / "label_mapping.json",
    ]

    for path in mapping_candidates:
        if path.exists():
            return read_json(path)

    metadata_path = trained_dir / "training_metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("label2id") and metadata.get("id2label"):
            id2label = {
                str(k): str(v)
                for k, v in metadata["id2label"].items()
            }
            label2id = {
                str(k): int(v)
                for k, v in metadata["label2id"].items()
            }
            return {
                "mode": metadata.get("mode", "multiclass"),
                "label_col": metadata.get("label_col", "llm_label"),
                "num_labels": int(metadata.get("num_labels", len(id2label))),
                "label2id": label2id,
                "id2label": id2label,
            }

    return DEFAULT_LABEL_MAPPING.copy()


def load_metadata(trained_dir: Path, adapter_source: Path, label_mapping: Dict[str, Any]) -> Dict[str, Any]:
    metadata_path = trained_dir / "training_metadata.json"
    adapter_config_path = adapter_source / "adapter_config.json"

    metadata: Dict[str, Any] = {}

    if metadata_path.exists():
        metadata.update(read_json(metadata_path))

    if adapter_config_path.exists():
        adapter_config = read_json(adapter_config_path)
    else:
        adapter_config = {}

    base_model_name = (
        metadata.get("base_model_name")
        or metadata.get("model_name")
        or metadata.get("pretrained_model_name_or_path")
        or adapter_config.get("base_model_name_or_path")
    )

    if not base_model_name:
        raise ValueError(
            "Could not infer base model. Provide training_metadata.json with model_name/base_model_name "
            "or adapter_config.json with base_model_name_or_path."
        )

    metadata.update({
        "backend": "qwen_sequence_classifier",
        "base_model_name": base_model_name,
        "adapter_dir": "adapter",
        "input_text_source": "selected_text_only",
        "label_mapping_file": "label_mapping.json",
        "num_labels": int(label_mapping["num_labels"]),
    })

    # Keep mappings in model_config too for backwards compatibility, but the runtime
    # will prefer label_mapping.json when present.
    metadata["label2id"] = label_mapping["label2id"]
    metadata["id2label"] = label_mapping["id2label"]

    return metadata


def copy_adapter(adapter_source: Path, target_adapter: Path, overwrite: bool) -> None:
    if target_adapter.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target adapter already exists: {target_adapter}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(target_adapter)

    ignore = shutil.ignore_patterns(
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
        "*.csv",
        "*.pkl",
    )

    shutil.copytree(adapter_source, target_adapter, ignore = ignore)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-dir", required = True)
    parser.add_argument("--artifacts-out", default = "meta_tag_classifier/artifacts")
    parser.add_argument("--overwrite", action = "store_true")
    args = parser.parse_args()

    trained_dir = Path(args.trained_dir)
    artifacts_out = Path(args.artifacts_out)

    if not trained_dir.exists():
        raise FileNotFoundError(f"trained_dir does not exist: {trained_dir}")

    adapter_source = find_adapter_source(trained_dir)
    label_mapping = load_label_mapping(trained_dir, adapter_source)
    metadata = load_metadata(trained_dir, adapter_source, label_mapping)

    artifacts_out.mkdir(parents = True, exist_ok = True)

    target_adapter = artifacts_out / "adapter"
    copy_adapter(adapter_source, target_adapter, overwrite = args.overwrite)

    write_json(artifacts_out / "label_mapping.json", label_mapping)
    write_json(artifacts_out / "model_config.json", metadata)

    print("Packaged Qwen artifacts")
    print(f"Adapter       : {target_adapter}")
    print(f"Label mapping : {artifacts_out / 'label_mapping.json'}")
    print(f"Config        : {artifacts_out / 'model_config.json'}")


if __name__ == "__main__":
    main()

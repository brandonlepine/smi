#!/usr/bin/env python3
"""
Utilities for loading sharded extraction outputs.

Expected directory layout:
  output_dir/
    manifest.pt
    extractions_shard_0000.pt
    extractions_shard_0001.pt
    ...

Manifest format:
  {
    "model_config": {...},
    "answer_token_ids": {...},
    "pair_metadata": [...],
    "shards": [
      {"shard_idx": 0, "filename": "...", ...},
      ...
    ]
  }

Shard format:
  {
    "original_results": {...},
    "cf_results": {...},
  }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, Optional

import torch


class ShardedExtractionStore:
    def __init__(self, extraction_dir: str | Path):
        self.extraction_dir = Path(extraction_dir)
        self.manifest_path = self.extraction_dir / "manifest.pt"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        self.manifest = torch.load(self.manifest_path, map_location="cpu", weights_only=False)
        self.model_config = self.manifest["model_config"]
        self.answer_token_ids = self.manifest.get("answer_token_ids", {})
        self.pair_metadata = self.manifest["pair_metadata"]
        self.shards = self.manifest["shards"]

        self._orig_index: Dict[str, int] = {}
        self._cf_index: Dict[str, int] = {}
        self._cache: Dict[int, Dict[str, Any]] = {}

        self._build_indexes()

    def _build_indexes(self):
        for shard_rec in self.shards:
            shard_idx = shard_rec["shard_idx"]
            shard_data = self._load_shard(shard_idx)

            for k in shard_data.get("original_results", {}).keys():
                self._orig_index[k] = shard_idx
            for k in shard_data.get("cf_results", {}).keys():
                self._cf_index[k] = shard_idx

        # clear cache after indexing
        self._cache = {}

    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        if shard_idx in self._cache:
            return self._cache[shard_idx]

        shard_rec = None
        for rec in self.shards:
            if rec["shard_idx"] == shard_idx:
                shard_rec = rec
                break
        if shard_rec is None:
            raise KeyError(f"Unknown shard_idx={shard_idx}")

        shard_path = self.extraction_dir / shard_rec["filename"]
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")

        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._cache[shard_idx] = data
        return data

    def clear_cache(self):
        self._cache = {}

    def get_original(self, key: str) -> Optional[Dict[str, Any]]:
        shard_idx = self._orig_index.get(key)
        if shard_idx is None:
            return None
        shard = self._load_shard(shard_idx)
        return shard.get("original_results", {}).get(key)

    def get_cf(self, key: str) -> Optional[Dict[str, Any]]:
        shard_idx = self._cf_index.get(key)
        if shard_idx is None:
            return None
        shard = self._load_shard(shard_idx)
        return shard.get("cf_results", {}).get(key)

    def iter_pair_metadata(self) -> Iterable[Dict[str, Any]]:
        yield from self.pair_metadata
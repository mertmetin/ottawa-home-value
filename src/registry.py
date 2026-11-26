"""Model versions on disk.

    models/registry/
      CURRENT                 the version the API serves (one line)
      promotions.jsonl        every retrain decision, promoted or not
      v20261126-0330/
        mid.lgb lo.lgb hi.lgb meta.json   the LightGBM model (train_gbm.GBM)
        card.json             what it was trained on, gate metrics, drift reference
        calib.json            nightly recalibration (overrides meta.json's calib)
        calib_history.jsonl   every recalibration, for the record

Promotion and rollback only rewrite CURRENT (atomically), so the API never
reads a half-written model: a version directory is complete before CURRENT
can point at it.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .train_gbm import GBM

KEEP_VERSIONS = 6  # old versions kept for rollback (the current one is always kept)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class Version:
    name: str
    path: Path

    @property
    def card(self) -> dict:
        return json.loads((self.path / "card.json").read_text())

    def load(self) -> GBM:
        g = GBM.load(self.path)
        if (self.path / "calib.json").exists():
            g.calib = json.loads((self.path / "calib.json").read_text())["calib"]
        return g

    def set_calib(self, calib: dict, info: dict) -> None:
        row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "calib": calib, **info}
        _atomic_write(self.path / "calib.json", json.dumps(row))
        append_jsonl(self.path / "calib_history.jsonl", row)


class Registry:
    def __init__(self, root: str | Path = "models/registry"):
        self.root = Path(root)

    def versions(self) -> list[Version]:
        if not self.root.exists():
            return []
        return [Version(p.name, p) for p in sorted(self.root.iterdir())
                if p.is_dir() and p.name.startswith("v") and (p / "card.json").exists()]

    def get(self, name: str) -> Version:
        v = Version(name, self.root / name)
        if not (v.path / "card.json").exists():
            raise KeyError(f"no model version {name}")
        return v

    def current(self) -> Version | None:
        f = self.root / "CURRENT"
        if not f.exists():
            return None
        return self.get(f.read_text().strip())

    def stamp(self) -> float:
        """Changes whenever the served model or its calibration changes."""
        cur = self.current()
        if cur is None:
            return 0.0
        files = [self.root / "CURRENT", cur.path / "calib.json"]
        return max(f.stat().st_mtime for f in files if f.exists())

    def save(self, g: GBM, card: dict, name: str | None = None) -> Version:
        name = name or time.strftime("v%Y%m%d-%H%M%S")
        tmp = self.root / f".{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        g.save(tmp)
        (tmp / "card.json").write_text(json.dumps({**card, "version": name}, indent=2, default=str))
        os.replace(tmp, self.root / name)
        return Version(name, self.root / name)

    def promote(self, name: str, why: str = "") -> None:
        self.get(name)  # must exist
        prev = self.current()
        _atomic_write(self.root / "CURRENT", name + "\n")
        append_jsonl(self.root / "promotions.jsonl", {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "serve", "version": name,
            "previous": prev.name if prev else None, "why": why})
        self.prune()

    def rollback(self) -> str:
        """Serve the version that was served before the current one."""
        cur = self.current()
        served = [r["version"] for r in read_jsonl(self.root / "promotions.jsonl") if r.get("event") == "serve"]
        have = {v.name for v in self.versions()}
        for name in reversed(served):
            if cur and name != cur.name and name in have:
                self.promote(name, why=f"rollback from {cur.name}")
                return name
        raise RuntimeError("no earlier version to roll back to")

    def prune(self, keep: int = KEEP_VERSIONS) -> None:
        cur = self.current()
        old = [v for v in self.versions() if not cur or v.name != cur.name]
        for v in old[:-keep] if len(old) > keep else []:
            shutil.rmtree(v.path)

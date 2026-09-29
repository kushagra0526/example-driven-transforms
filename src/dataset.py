"""Loading and sampling benchmark tasks.

Each task is a folder with a meta.json (feature tags) and a spec.json (a list of
input/output examples). Inputs are lists of column values; any value, input or
output, may be null.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "prose-benchmarks" / "Transformation.Text"


@dataclass(frozen=True)
class Example:
    inputs: tuple[str | None, ...]
    output: str | None


@dataclass
class Task:
    id: str
    features: list[str]
    synthetic: bool
    given: list[Example]
    held_out: Example

    @property
    def n_columns(self) -> int:
        return len(self.held_out.inputs)

    def unique_given(self) -> list[Example]:
        """Given examples with exact duplicates removed, first-occurrence order kept."""
        seen: set[Example] = set()
        out = []
        for ex in self.given:
            if ex not in seen:
                seen.add(ex)
                out.append(ex)
        return out

    def held_out_input_seen(self) -> bool:
        """True when the held-out input also appears among the given examples."""
        return any(ex.inputs == self.held_out.inputs for ex in self.given)


def _read_json(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def list_task_ids(root: Path = DEFAULT_ROOT) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"Benchmark folder not found: {root}\n"
            "Clone it first: git clone https://github.com/microsoft/prose-benchmarks data/prose-benchmarks"
        )
    return sorted(p.name for p in root.iterdir() if (p / "spec.json").is_file())


def sample_task_ids(n: int, seed: int, root: Path = DEFAULT_ROOT) -> list[str]:
    """A reproducible sample: the same seed and folder listing always give the same ids."""
    ids = list_task_ids(root)
    return random.Random(seed).sample(ids, min(n, len(ids)))


def load_task(task_id: str, root: Path = DEFAULT_ROOT) -> Task:
    folder = root / task_id
    meta = _read_json(folder / "meta.json")
    spec = _read_json(folder / "spec.json")
    examples = [Example(tuple(e["Input"]), e["Output"]) for e in spec["Examples"]]
    if len(examples) < 2:
        raise ValueError(f"{task_id}: need at least 2 examples, found {len(examples)}")
    return Task(
        id=task_id,
        features=list(meta.get("Features", [])),
        synthetic=bool(meta.get("Synthetic", False)),
        given=examples[:-1],
        held_out=examples[-1],
    )

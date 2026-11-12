"""Model folders: a model's networks, the network written as an entry, and what the agent is given.

A model folder (models/<name>/) holds

    config.yaml   sizes, the input space, the data set (every input unless it says otherwise), the weights'
                  names and shapes, the networks (id, file, sha256)
    model.py      reference(g, lv1): the network's own computation in the entry language
    zoo/*.npz     the networks' weights (no pickle)

Nothing in it says what the network was trained to do; that is in withheld/<name>/ (task.yaml with the
hygiene strings, references.py with reference entries), which only the verifier's side reads.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from . import lang
from .util import REPO_ROOT, read_yaml, sha256_file


class ModelError(RuntimeError):
    pass


def _load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        if not (self.path / "config.yaml").is_file():
            raise ModelError(f"{self.path} is not a model folder (no config.yaml)")
        self.config = read_yaml(self.path / "config.yaml")
        self.name = self.config["name"]
        self.sizes = dict(self.config["sizes"])
        self.n_ctx = int(self.sizes["n_ctx"])
        # The data set: every input of the input space unless config.yaml says otherwise (lang.py: data sets).
        self.data_set = dict(self.config.get("data_set") or {"kind": "all"})
        try:
            self.real = lang.real_inputs(self.data_set)
        except ValueError as exc:
            raise ModelError(f"the data set of {self.name} is not well described: {exc}") from None
        space = list(self.config["input_space"])
        if self.real != all(n == "real" for n in space) or (not self.real and not all(isinstance(n, int) for n in space)):
            raise ModelError("input_space says `real` for every position of a data set of numbers (kind normal), "
                             "and a whole number of tokens for every position otherwise")
        self.input_space = space if self.real else [int(n) for n in space]
        if len(self.input_space) != self.n_ctx:
            raise ModelError("input_space needs one entry per token position")
        if self.real and not self.data_set.get("says"):
            raise ModelError("a data set of numbers needs the sentence the agent is told about it (data_set.says)")
        try:
            self.n_total = lang.count_inputs(self.data_set, self.input_space)  # None: infinitely many
        except (ValueError, KeyError) as exc:
            raise ModelError(f"the data set of {self.name} is not well described: {exc}") from None
        self._module = None

    def can_enumerate(self) -> bool:
        """Whether every input of the data set can be listed, each counting once."""
        return lang.data_set_kind(self.data_set) == "all"

    def data_set_text(self) -> str:
        """The data set in words, for the agent's prompt: the model folder's own, or the default."""
        says = self.data_set.get("says")
        if says:
            return " ".join(str(says).split())
        return f"The data set is every input, each counted once: {self.n_total:,} inputs."

    # the networks
    def network(self, network_id: str) -> dict:
        for n in self.config["networks"]:
            if n["id"] == network_id:
                return n
        raise ModelError(f"{network_id} is not a network of {self.name}; it has {[n['id'] for n in self.config['networks']]}")

    def weights_path(self, network_id: str) -> Path:
        return self.path / self.network(network_id)["weights"]

    def load_weights(self, network_id: str) -> dict:
        entry = self.network(network_id)
        path = self.path / entry["weights"]
        if sha256_file(path) != entry["sha256"]:
            raise ModelError(f"the weights of {network_id} do not match the hash in config.yaml")
        with np.load(path, allow_pickle=False) as z:
            weights = {k: z[k] for k in z.files}
        shapes = {k: list(v.shape) for k, v in weights.items()}
        if shapes != {k: list(v) for k, v in self.config["weights"].items()}:
            raise ModelError(f"the weights of {network_id} have shapes {shapes}, not those of config.yaml")
        return weights

    # the network as an entry
    def source(self) -> str:
        return (self.path / "model.py").read_text()

    def module(self):
        if self._module is None:
            self._module = _load_py(self.path / "model.py", f"level1_model_{self.name.replace('-', '_')}")
        return self._module

    def reference_entry(self, weights: dict) -> dict:
        g = lang.Graph(weights, self.n_ctx, real=self.real)
        return g.entry(predict=self.module().reference(g, lang))

    # what the agent is given about the network besides its weights and model.py
    def info(self) -> dict:
        return {"sizes": self.sizes, "input_space": self.input_space, "n_outputs": int(self.config["n_outputs"]),
                "weights": {k: list(v) for k, v in self.config["weights"].items()},
                "data_set": {k: v for k, v in self.data_set.items() if k != "says"}}

    # the verifier's side only
    def withheld(self) -> Path:
        return self.path.parent.parent / "withheld" / self.name

    def task(self) -> dict:
        p = self.withheld() / "task.yaml"
        return read_yaml(p) if p.is_file() else {}

    def hygiene(self) -> list:
        return [str(s) for s in (self.task().get("hygiene") or [])]

    def reference_entries(self, weights: dict) -> dict:
        """The withheld reference entries, built for these weights: {name: entry}. Empty without the folder."""
        p = self.withheld() / "references.py"
        if not p.is_file():
            return {}
        mod = _load_py(p, f"level1_withheld_{self.name.replace('-', '_')}")
        out = {}
        for name, fn in getattr(mod, "ENTRIES", {}).items():
            g = lang.Graph(weights, self.n_ctx, real=self.real)
            out[name] = g.entry(**fn(g, lang))
        for name, fn in getattr(mod, "ENTRIES_WITH_REFERENCE", {}).items():
            g = lang.Graph(weights, self.n_ctx, real=self.real)
            out[name] = g.entry(**fn(g, lang, self.module().reference))
        return out


def load_model(path) -> Model:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return Model(p)

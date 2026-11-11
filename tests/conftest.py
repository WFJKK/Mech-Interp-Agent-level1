import numpy as np
import pytest

from level1 import lang
from level1.networks import load_model


@pytest.fixture(scope="session")
def model():
    return load_model("models/lin-a")


@pytest.fixture(scope="session")
def nets(model):
    """{network id: (weights, reference entry, all inputs, the network's outputs, forward operations)}"""
    out = {}
    x = lang.all_inputs(model.input_space)
    for n in model.config["networks"]:
        w = model.load_weights(n["id"])
        ref = model.reference_entry(w)
        answers, ops = lang.run_reference(ref, w, model.n_ctx, x)
        out[n["id"]] = (w, ref, x, answers, ops)
    return out


@pytest.fixture(scope="session")
def model4():
    """The same networks on a data set of inputs with 1 to 4 tokens."""
    return load_model("models/lin-a-1to4")


@pytest.fixture()
def toy():
    """Small weights for tests of the language itself."""
    rng = np.random.default_rng(0)
    return {"A": rng.normal(size=(3, 4)), "B": rng.normal(size=(4, 5)), "v": rng.normal(size=4),
            "T": rng.normal(size=(6, 6)), "K": rng.integers(0, 6, size=(6, 6))}

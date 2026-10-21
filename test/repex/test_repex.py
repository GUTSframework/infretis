import os
from pathlib import PosixPath
import pytest
import tomli

from infretis.classes.repex import REPEX_state, spawn_rng


@pytest.mark.parametrize("seed", [0, 2])
def test_rgen_io(tmp_path: PosixPath, seed: int) -> None:
    """Test repex rgen and rgen spawn reproducability."""
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0},
            "runner": {"workers": 1},
            "simulation": {"seed": seed, "steps":10,
                           "zeroswap": 0.5, "pick_scheme": 0},
        }
    )
    folder = tmp_path / "temp"
    folder.mkdir()
    os.chdir(folder)

    # save initial state for restart
    state.write_toml()

    # generate numbers
    save_rng = []
    save_rng_child = []
    for i in range(5):
        save_rng.append(state.rgen.random())
        child = spawn_rng(state.rgen)
        save_rng_child.append(child.random())

    # restart with the "restarted_from" keyword
    with open("restart.toml", mode="rb") as f:
        config = tomli.load(f)
        config["current"]["restarted_from"] = {}
    state = REPEX_state(config)

    # test that the numbers are the same
    for rng, child_rng in zip(save_rng, save_rng_child):
        assert state.rgen.random() == rng
        child = spawn_rng(state.rgen)
        assert child.random() == child_rng


def test_write_toml_keeps_old_file_on_failure(
    tmp_path: PosixPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that fails halfway leaves the previous restart.toml intact."""
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0},
            "runner": {"workers": 1},
            "simulation": {"seed": 0, "steps": 10,
                           "zeroswap": 0.5, "pick_scheme": 0},
        }
    )
    os.chdir(tmp_path)
    state.write_toml()
    before = (tmp_path / "restart.toml").read_bytes()

    def dump_halfway(obj, f):
        f.write(b"[current]\ncst")
        raise OSError("disk full")

    state.config["current"]["cstep"] = 1
    monkeypatch.setattr("infretis.classes.repex.tomli_w.dump", dump_halfway)
    with pytest.raises(OSError):
        state.write_toml()
    assert (tmp_path / "restart.toml").read_bytes() == before

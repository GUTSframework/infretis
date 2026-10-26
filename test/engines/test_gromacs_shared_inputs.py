"""Test that the GROMACS input files are written once, at setup."""

import copy
import pathlib
import shutil

import pytest
import tomli

from infretis.classes.engines.factory import create_engine, create_engines
from infretis.classes.engines.gromacs import GromacsEngine

HERE = pathlib.Path(__file__).resolve().parent
EXAMPLE = HERE / "../../examples/gromacs/H2"

pytestmark = pytest.mark.skipif(
    shutil.which("gmx") is None, reason="GROMACS (gmx) not found"
)


@pytest.fixture
def config(tmp_path: pathlib.Path, monkeypatch) -> dict:
    """The H2 example's config, in a copy of its input folder."""
    shutil.copytree(EXAMPLE / "gromacs_input", tmp_path / "gromacs_input")
    for leftover in ("infretis.mdp", "topol.tpr"):
        (tmp_path / "gromacs_input" / leftover).unlink(missing_ok=True)
    monkeypatch.chdir(tmp_path)
    with open(EXAMPLE / "infretis.toml", mode="rb") as f:
        config = tomli.load(f)
    config["simulation"]["ensemble_engines"] = [
        ["engine"] for _ in config["simulation"]["interfaces"]
    ]
    # Its default is the folder infretis was first imported in.
    config["engine"]["exe_path"] = str(tmp_path)
    return config


def test_inputs_written_at_setup_only(config: dict) -> None:
    """create_engines writes the .mdp and .tpr; a worker's call does not."""
    mdp = pathlib.Path("gromacs_input/infretis.mdp")
    create_engines(config, write_inputs=False)
    assert not mdp.exists()
    create_engines(config)
    assert mdp.exists() and pathlib.Path("gromacs_input/topol.tpr").exists()
    written = mdp.stat().st_mtime_ns
    create_engines(config, write_inputs=False)
    assert mdp.stat().st_mtime_ns == written


def test_engines_with_different_settings_share_no_input(config: dict) -> None:
    """Two engines on one input_path with different settings are refused."""
    other = copy.deepcopy(config["engine"])
    other["timestep"] = 2 * other["timestep"]
    config["engine1"] = other
    config["simulation"]["ensemble_engines"][-1] = ["engine1"]
    with pytest.raises(ValueError, match="'engine' and 'engine1'"):
        create_engines(config)
    assert not pathlib.Path("gromacs_input/infretis.mdp").exists()
    # The same settings under another name are fine.
    config["engine1"] = copy.deepcopy(config["engine"])
    create_engines(config)


def test_create_engine_writes_its_inputs(config: dict) -> None:
    """A single engine, as inftools' shoot creates it, writes its inputs."""
    create_engine(config)
    assert pathlib.Path("gromacs_input/infretis.mdp").exists()
    assert pathlib.Path("gromacs_input/topol.tpr").exists()


def test_shared_inputs_written_once(config: dict, monkeypatch) -> None:
    """Two engines that write the same files run grompp once."""
    config["engine1"] = copy.deepcopy(config["engine"])
    config["simulation"]["ensemble_engines"][-1] = ["engine1"]
    calls = []
    write = GromacsEngine.write_shared_inputs

    def counted(self):
        calls.append(self)
        write(self)

    monkeypatch.setattr(GromacsEngine, "write_shared_inputs", counted)
    create_engines(config)
    assert len(calls) == 1

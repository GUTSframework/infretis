import copy
import os
from pathlib import Path, PosixPath

import pytest
import tomli
import tomli_w

from infretis.setup import (
    TOMLConfigError,
    check_config,
    setup_config,
    write_header,
)

HERE = Path(__file__).resolve().parent


def test_write_header(tmp_path: PosixPath) -> None:
    """Test that we create new data file if datafile already present."""
    f1 = tmp_path / "temp"
    f1.mkdir()
    os.chdir(f1)
    config: dict = {"current": {"size": 10}, "output": {"data_dir": "./"}}

    # write the first infretis_data.txt file
    write_header(config)
    assert os.path.isfile("./infretis_data.txt")
    for i in range(1, 6):
        # create new infretis_data.txt files
        write_header(config)
        isfile = f"./infretis_data_{i}.txt"
        assert os.path.isfile(isfile)
        assert config["output"]["data_file"] == isfile


def set_nested_value(d, keys, value):
    """Set a value in a nested dictionary by following the list of keys,
    creating keys if they don't exist."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}  # Create a new dict if the key doesn't exist
        d = d[key]
    d[keys[-1]] = value


def test_check_config():
    toml_path = (
        Path(__file__).parent / "../../examples/gromacs/H2/infretis.toml"
    )
    original_config = setup_config(toml_path)
    test_cases = [
        (["runner", "workers"], 100),
        (["simulation", "tis_set", "interface_cap"], 100),
        (["simulation", "interfaces"], [0.0, 0.5, 0.2, 1.0]),
        (["simulation", "interfaces"], [0.0, 0.2, 0.2, 1.0]),
        (["simulation", "interfaces"], []),
    ]
    for keys, invalid_value in test_cases:
        config = copy.deepcopy(original_config)
        set_nested_value(config, keys, invalid_value)
        print("Testing:", keys, invalid_value)
        with pytest.raises(TOMLConfigError):
            check_config(config)


def test_multi_engine_config():
    toml_path = (
        Path(__file__).parent / "../../examples/gromacs/H2/infretis.toml"
    )
    original_config = setup_config(toml_path)
    original_config["simulation"]["tis_set"]["multi_engine"] = True
    original_config["engine0"] = original_config["engine"].copy()
    original_config["engine1"] = original_config["engine"].copy()
    original_config["engine2"] = original_config.pop("engine")
    test_cases = [
        (["engine2", "timestep"], 10.0),
    ]
    for keys, invalid_value in test_cases:
        config = copy.deepcopy(original_config)
        set_nested_value(config, keys, invalid_value)
        print("Testing:", keys, invalid_value)
        with pytest.raises(TOMLConfigError):
            check_config(config)


def finished_run(monkeypatch, folder: PosixPath) -> dict:
    """A run that has done all its steps, with its active paths on disk."""
    restart = HERE / "../simulations/data/10steps_wf/restart.toml"
    folder.mkdir()
    monkeypatch.chdir(folder)
    with open(restart, mode="rb") as f:
        config = tomli.load(f)
    for act in config["current"]["active"]:
        os.makedirs(f"load/{act}")
        Path(f"load/{act}/traj.txt").touch()
    write_toml(config)
    return config


def write_toml(config: dict) -> None:
    with open("restart.toml", "wb") as f:
        tomli_w.dump(config, f)


def test_finished_run_continues_with_more_steps(tmp_path, caplog, monkeypatch):
    """A finished run is refused without writing anything, so raising its
    steps later continues it."""
    config = finished_run(monkeypatch, tmp_path / "run")
    before = Path("restart.toml").read_bytes()
    assert setup_config("restart.toml") is None
    assert "the run is done (cstep 10, steps 10)" in caplog.text
    assert Path("restart.toml").read_bytes() == before

    config["simulation"]["steps"] = 20
    write_toml(config)
    restarted = setup_config("restart.toml")
    assert restarted["current"]["restarted_from"] == 10


def test_refusals_say_why(tmp_path, caplog, monkeypatch):
    """Each refusal of setup_config logs its reason."""
    config = finished_run(monkeypatch, tmp_path / "run")
    assert setup_config("nosuch.toml") is None
    assert "nosuch.toml file not found" in caplog.text

    config["simulation"]["steps"] = 20
    config["current"]["restarted_from"] = 10
    write_toml(config)
    assert setup_config("restart.toml") is None
    assert "the last launch from it ran no step" in caplog.text

    del config["current"]["restarted_from"]
    write_toml(config)
    gone = config["current"]["active"][0]
    os.remove(f"load/{gone}/traj.txt")
    assert setup_config("restart.toml") is None
    assert f"active path {gone} is missing" in caplog.text

"""Test stopping a run with SIGTERM and continuing it from restart.toml."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import PosixPath

import pytest
import tomli
import tomli_w

from infretis.asyncrunner import future_list
from infretis.classes.repex import REPEX_state
from infretis.scheduler import scheduler
from infretis.setup import setup_config

BASEPATH = PosixPath(__file__).parent


def prepare(monkeypatch, folder: PosixPath, workers: int, steps: int) -> None:
    """Copy the turtlemd double well example to `folder` and go there."""
    load_dir = (
        BASEPATH / "../../examples/turtlemd/double_well/load_copy"
    ).resolve()
    folder.mkdir()
    shutil.copytree(load_dir, folder / "load")
    shutil.copy(load_dir / "../orderp.py", folder)
    monkeypatch.chdir(folder)
    with open(BASEPATH / "data/wf.toml", mode="rb") as f:
        config = tomli.load(f)
    config["simulation"]["steps"] = steps
    config["runner"]["workers"] = workers
    with open("infretis.toml", "wb") as f:
        tomli_w.dump(config, f)


def cstep() -> int:
    """The step saved in restart.toml."""
    with open("restart.toml", mode="rb") as f:
        return tomli.load(f)["current"]["cstep"]


def children(pid: int) -> list:
    """The child processes of `pid` (Linux)."""
    with open(f"/proc/{pid}/task/{pid}/children") as f:
        return [int(child) for child in f.read().split()]


def alive(pid: int) -> bool:
    """Whether `pid` runs, and is not a zombie (Linux)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_as_completed_stops_waiting():
    """as_completed returns None once `stop` says so."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        futures = future_list()
        futures.add(loop.create_future())  # never done
        start = time.monotonic()
        assert futures.as_completed(stop=lambda: True) is None
        assert time.monotonic() - start < 1
    finally:
        loop.close()


def test_signal_inside_treat_output_saves_the_result(tmp_path, monkeypatch):
    """A signal while a result is recorded: the result and restart.toml
    are saved, then the scheduler stops and returns the signal."""
    prepare(monkeypatch, tmp_path / "run", workers=2, steps=1000)
    original = REPEX_state.treat_output
    calls = []

    def treat_output(state, md_items):
        calls.append(state.cstep)
        os.kill(os.getpid(), signal.SIGTERM)  # handled between bytecodes
        return original(state, md_items)

    monkeypatch.setattr(REPEX_state, "treat_output", treat_output)
    config = setup_config("infretis.toml")
    assert scheduler(config) == signal.SIGTERM
    assert len(calls) == 1
    assert cstep() == calls[0]
    # The handlers are given back.
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="reads /proc")
def test_sigterm_stops_and_the_run_continues(tmp_path, monkeypatch):
    """infretisrun stopped by SIGTERM exits with 128 + 15, leaves a
    restart.toml it continues from, and ends its worker processes."""
    prepare(monkeypatch, tmp_path / "run", workers=4, steps=1000000)
    exe = shutil.which("infretisrun") or os.path.join(
        os.path.dirname(sys.executable), "infretisrun"
    )
    proc = subprocess.Popen([exe, "-i", "infretis.toml"])
    deadline = time.monotonic() + 60
    while not os.path.isfile("restart.toml") or cstep() < 10:
        assert time.monotonic() < deadline, "no progress"
        time.sleep(0.2)
    workers = children(proc.pid)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    assert not [pid for pid in workers if alive(pid)]
    stopped_at = cstep()
    assert stopped_at >= 10

    with open("restart.toml", mode="rb") as f:
        config = tomli.load(f)
    config["simulation"]["steps"] = stopped_at + 5
    with open("restart.toml", "wb") as f:
        tomli_w.dump(config, f)
    done = subprocess.run([exe, "-i", "restart.toml"], timeout=120)
    assert done.returncode == 0
    assert cstep() == stopped_at + 5

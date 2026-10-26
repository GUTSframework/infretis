"""Test the per-pin worker log of infretis.core.tis."""
import logging

import pytest

from infretis.core.tis import hold_early_log, worker_log


def test_worker_log_per_pin(tmp_path, monkeypatch):
    """Each move logs to the file of its pin; the header comes once."""
    monkeypatch.chdir(tmp_path)
    log = logging.getLogger("infretis.test")
    log.setLevel(logging.INFO)
    handlers = list(logging.getLogger("").handlers)
    for pin, text in [(1, "first"), (0, "other"), (1, "second")]:
        with worker_log(pin):
            log.info(text)
    assert logging.getLogger("").handlers == handlers
    one = (tmp_path / "worker1.log").read_text()
    assert one.count("Logging file for worker 1") == 1
    assert "first" in one and "second" in one and "other" not in one
    assert "other" in (tmp_path / "worker0.log").read_text()


def test_worker_log_of_a_failing_move(tmp_path, monkeypatch):
    """A move that raises leaves no handler behind."""
    monkeypatch.chdir(tmp_path)
    handlers = list(logging.getLogger("").handlers)
    with pytest.raises(RuntimeError):
        with worker_log(0):
            raise RuntimeError("move failed")
    assert logging.getLogger("").handlers == handlers


def test_early_log_goes_to_the_first_move(tmp_path, monkeypatch):
    """What the worker logs before its first move (its initializer) is
    written to the log of that move, and not to later ones."""
    monkeypatch.chdir(tmp_path)
    log = logging.getLogger("infretis.test")
    log.setLevel(logging.INFO)
    handlers = list(logging.getLogger("").handlers)
    hold_early_log()
    log.info("engines created")
    for pin in (2, 3):
        with worker_log(pin):
            log.info("move")
    assert logging.getLogger("").handlers == handlers
    two = (tmp_path / "worker2.log").read_text()
    assert two.index("Logging file") < two.index("engines created")
    assert two.index("engines created") < two.index("move")
    assert "engines created" not in (tmp_path / "worker3.log").read_text()

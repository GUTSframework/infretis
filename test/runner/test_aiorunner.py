import concurrent.futures
import threading
import time
from pathlib import PosixPath
from time import sleep
from typing import Dict, Any
import os

import pytest
import asyncio
from infretis.asyncrunner import aiorunner, future_list

class TaskError(Exception):
    """Exception class for the test task."""

    pass


def sleeping(sleep_dict: Dict[str, Any]) -> Dict[str, Any]:
    """A simple sleeping task.

    Raising an error if triggered

    Args
        sleep_dict: a dictionary of task params

    Return
        A dictionary containing the task results
    """
    raise_error = sleep_dict.get("raise_err", False)
    if raise_error:
        raise TaskError("Task raise error on purpose")
    else:
        sleeping_time = sleep_dict.get("duration", 0.5)
        sleep(sleeping_time)
        return {"Time slept": sleeping_time}


def test_runner_init():
    """Test bare runner init."""
    runner = aiorunner({})
    assert runner.n_workers() == 1
    runner.stop()


def test_runner_attach_callable():
    """Test attach callable to runner."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        runner = aiorunner({}, 2)
        runner.set_task(sleeping)
        assert runner.n_workers() == 2
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)

def test_runner_fail_start():
    """Test fail runner start missing callable."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        runner = aiorunner({}, 1)
        with pytest.raises(Exception):
            runner.start()
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)

def test_runner_start_and_stop():
    """Test runner start and stop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        runner = aiorunner({}, 1)
        runner.set_task(sleeping)
        runner.start()
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)

def test_runner_fail_submit():
    """Test fail runner submit work, need start."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        runner = aiorunner({}, 2)
        with pytest.raises(Exception):
            runner.submit_work({"duration": 1.0})
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)

def test_runner_check_return_success():
    """Test runner return task result."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        runner = aiorunner({}, 1)
        futlist = future_list()
        runner.set_task(sleeping)
        runner.start()
        futlist.add(runner.submit_work({"duration": 0.3}))
        fut = futlist.as_completed()
        assert fut.result().get("Time slept") == 0.3
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)

def test_runner_task_error():
    """Test fail runner task raised error."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        runner = aiorunner({}, 1)
        futlist = future_list()
        runner.set_task(sleeping)
        runner.start()
        futlist.add(runner.submit_work({"raise_err": True}))
        fut = futlist.as_completed()
        with pytest.raises(TaskError):
            fut.result()
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)

def test_runner_infretis_mode(tmp_path: PosixPath):
    """Test runner operating in infretis mode."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    os.chdir(tmp_path)
    try:
        n_workers = 2
        runner = aiorunner({}, n_workers)
        futlist = future_list()
        runner.set_task(sleeping)
        runner.start()
        loop_cnt = 0
        n_loops = 10
        res_dict = {}
        while loop_cnt < n_loops + 2:
            if loop_cnt < n_workers:
                futlist.add(runner.submit_work({}))
                loop_cnt = loop_cnt + 1
            else:
                future = futlist.as_completed()
                if future:
                    res_dict[f"l{loop_cnt}"] = future.result()
                if loop_cnt <= n_loops:
                    futlist.add(runner.submit_work({}))
                loop_cnt = loop_cnt + 1
        assert len(res_dict) == 10
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)


def test_future_list_waits_without_spinning():
    """as_completed returns futures as they finish and None when the list
    is empty, and uses little CPU while it waits."""
    futlist = future_list()
    assert futlist.as_completed() is None
    fut = concurrent.futures.Future()
    futlist.add(fut)
    timer = threading.Timer(0.5, fut.set_result, ("done",))
    timer.start()
    cpu0 = time.thread_time()
    assert futlist.as_completed() is fut
    cpu = time.thread_time() - cpu0
    timer.join()
    assert futlist.as_completed() is None
    # A busy loop uses about the full 0.5 s.
    assert cpu < 0.1


def test_runner_executor_factory():
    """The runner builds its executor with the given factory."""
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=kwargs["max_workers"],
            initializer=kwargs["initializer"],
            initargs=kwargs["initargs"],
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        runner = aiorunner({}, 1, executor_factory=factory)
        futlist = future_list()
        runner.set_task(sleeping)
        runner.start()
        futlist.add(runner.submit_work({"duration": 0.1}))
        fut = futlist.as_completed()
        assert fut.result().get("Time slept") == 0.1
    finally:
        runner.stop()
        loop.close()
        asyncio.set_event_loop(None)
    assert len(calls) == 1
    assert calls[0]["max_workers"] == 1
    assert calls[0]["initargs"] == ({},)


def md_program(folder: str) -> None:
    """Start a program in a session of its own and wait for it, as the
    engines do; stop it when the wait ends with an exception."""
    import signal
    import subprocess

    proc = subprocess.Popen(["sleep", "60"], preexec_fn=os.setsid)
    try:
        with open(os.path.join(folder, "pids"), "w") as f:
            f.write(f"{os.getpid()} {proc.pid}")
        proc.wait()
    finally:
        if proc.returncode is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()


def test_sigterm_to_a_worker_stops_its_md_program(tmp_path, monkeypatch):
    """A worker ended by SIGTERM unwinds its task, so the program the
    task started in its own session does not outlive it."""
    import multiprocessing
    import signal

    from infretis.asyncrunner import WorkerTerminated, worker_initializer

    monkeypatch.chdir(tmp_path)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        initializer=worker_initializer,
        initargs=({},),
        mp_context=context,
    ) as executor:
        future = executor.submit(md_program, str(tmp_path))
        pids = tmp_path / "pids"
        for _ in range(200):
            if pids.is_file() and pids.read_text():
                break
            sleep(0.05)
        worker, program = map(int, pids.read_text().split())
        try:
            os.kill(worker, signal.SIGTERM)
            with pytest.raises(WorkerTerminated):
                future.result(timeout=30)
            with pytest.raises(ProcessLookupError):
                os.kill(program, 0)
        finally:
            try:
                os.kill(program, signal.SIGKILL)
            except ProcessLookupError:
                pass

"""An asyncio-based task runner for infRETIS."""

import asyncio
import concurrent.futures
import functools
import logging
import multiprocessing
import signal
import threading
import time
from asyncio import Future
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import infretis.core.tis
from infretis.classes.engines.factory import create_engines
from infretis.classes.orderparameter import create_orderparameters

logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)


class RunnerError(Exception):
    """Exception class for the runner."""

    pass


class WorkerTerminated(Exception):
    """Raised in a worker that got SIGTERM, to end the task it runs."""

    pass


class aiorunner:
    """A light asynchronuous runner based on asyncio.

    The runner manage an asyncio.queue with a pool of workers.
    Upon instanciation, a dedicated event loop
    is launched in a separate thread. The user can then
    attach a worker function to the runner and start multiple
    instances of that function in the background.
    As work is submitted to the runner, it is picked up by
    workers on-the-fly.
    """

    def __init__(
        self,
        config: Dict,
        n_workers: int = 1,
        executor_factory: Optional[
            Callable[..., concurrent.futures.Executor]
        ] = None,
    ) -> None:
        """Init function of runner.

        Args:
            config: the configuration dictionary
            n_workers: number of workers active in the runner
            executor_factory: a callable that takes the arguments of
                concurrent.futures.ProcessPoolExecutor and returns the
                executor that runs the tasks (default: ProcessPoolExecutor)
        """
        if executor_factory is None:
            executor_factory = concurrent.futures.ProcessPoolExecutor
        self._n_workers: int = n_workers
        self._executor: concurrent.futures.Executor = executor_factory(
            max_workers=n_workers,
            initializer=worker_initializer,
            initargs=(config,),
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._stop_event = asyncio.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._start_event_loop, daemon=True
        )
        self._thread.start()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task_f: Optional[Callable] = None
        self._tasks: Optional[List[asyncio.Task[Any]]] = None

    def start(self) -> None:
        """Launch background tasks."""
        future = asyncio.run_coroutine_threadsafe(
            self._start_tasks(), self._loop
        )
        try:
            # Task startup should be fast
            future.result(5.0)
        except TimeoutError:
            raise RunnerError("Launching background tasks took too long")
        except Exception as e:
            raise (e)

    def _start_event_loop(self) -> None:
        """Start the event loop in a separate thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def set_task(self, task_f: Callable) -> None:
        """Attach the task function to the runner.

        Args:
            task_f: a callable function
        """
        self._task_f = task_f

    async def _task_wrapper(
        self,
        stop_event: asyncio.Event,
        queue: asyncio.Queue,
        executor: concurrent.futures.Executor,
        taskID: int,
    ) -> None:
        """Wrap the sync task.

        To enable running the sync task_f
        from a dynamic list of tasks.

        Args:
            stop_event: a asyncio event to stop the worker
            queue: an asyncio queue to get work from
            executor: an executor
            taskID : an ID for the long running task
        """
        while not stop_event.is_set():
            try:
                # Unpack queue element
                md_item, future = queue.get_nowait()

                # Run the task in the event loop
                assert self._task_f
                loop = asyncio.get_running_loop()
                try:
                    md_item = await loop.run_in_executor(
                        executor, functools.partial(self._task_f, md_item)
                    )
                    future.set_result(md_item)
                except Exception as e:
                    # Pass the exception up in the future
                    future.set_exception(e)

                # Mask the task as done
                queue.task_done()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)

    async def _add_work_to_queue(
        self, work_unit: Dict[str, Any]
    ) -> asyncio.Future:
        """Async function adding work to queue, returns a future.

        Args
            work_unit: a unit of work encapsulated in a dict

        Return:
            A future wih the results of the work
        """
        future: asyncio.Future = asyncio.Future()
        await self._queue.put((work_unit, future))
        return future

    def submit_work(self, work_unit: Dict[str, Any]) -> Future:
        """Submit work to the runner.

        Args:
            task: a unit of work encapsulated in a dict

        Return:
            A future wih the results of the work
        """
        if not self._tasks:
            raise RunnerError(
                "Unable to submit work if the tasks haven't been initiated"
            )
        future = asyncio.run(self._add_work_to_queue(work_unit))
        # Need to wait otherwise some race condition can occur
        time.sleep(0.05)
        return future

    async def _start_tasks(self) -> None:
        """Launch the background tasks."""
        if not self._task_f:
            raise RunnerError("Can't start task(s) without a task function.")
        try:
            self._tasks = [
                asyncio.create_task(
                    self._task_wrapper(
                        self._stop_event, self._queue, self._executor, i
                    )
                )
                for i in range(self._n_workers)
            ]
        except Exception as e:
            raise e

    async def wait_for_tasks_to_end(self) -> None:
        """Async function waiting for tasks to end."""
        while len(asyncio.all_tasks(self._loop)) > 0:
            await asyncio.sleep(0.1)

    def n_workers(self) -> int:
        """Return runner number of workers."""
        return self._n_workers

    def stop(self) -> None:
        """Terminate the runner."""
        # Make sure there is no more work in the queue
        # before dispatching the task stopping event
        while self._queue.qsize() > 0:
            time.sleep(0.1)

        # Stop ongoing tasks
        self._stop_event.set()

        # Wait until all tasks are done
        asyncio.run(self.wait_for_tasks_to_end())

        # Close the event loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def cancel(self) -> None:
        """Terminate the runner now, without waiting for the tasks.

        Queued tasks are cancelled and the worker processes of a
        ProcessPoolExecutor are terminated; another executor is asked to
        cancel its futures. Results of tasks that were running are lost.
        """
        # shutdown() forgets the processes, so collect them first.
        processes = list(
            (getattr(self._executor, "_processes", None) or {}).values()
        )
        self._executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()
        asyncio.run_coroutine_threadsafe(
            self._cancel_tasks(), self._loop
        ).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    async def _cancel_tasks(self) -> None:
        """Cancel the background tasks and wait until they have ended."""
        self._stop_event.set()
        for task in self._tasks or []:
            task.cancel()
        await asyncio.gather(*(self._tasks or []), return_exceptions=True)


def worker_initializer(config):
    """Initialize function for each worker process.

    Creates the engines and order parameters. Each move logs to the
    worker{pin}.log of the pin it runs on (see core.tis.run_md), so the
    initializer needs no shared state: it also works with an executor
    whose workers share no memory with this process.

    On SIGTERM, the worker raises WorkerTerminated, so the move it runs
    unwinds: the engines then stop the MD programs they started in
    sessions of their own, which the signal does not reach.
    """
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _raise_terminated)
    infretis.core.tis.hold_early_log()
    # load engines if infretis simulation
    if "simulation" in config:
        # The parent wrote the engines' input files at setup.
        engines, _ = create_engines(config, write_inputs=False)
        create_orderparameters(engines, config)

        # set the ENGINES variable in the tis file for each worker
        infretis.core.tis.ENGINES = engines


def _raise_terminated(signum, frame):
    """Raise WorkerTerminated; a second SIGTERM ends the worker at once."""
    signal.signal(signum, signal.SIG_DFL)
    raise WorkerTerminated(f"worker got {signal.Signals(signum).name}")


class future_list:
    """A managed list of future."""

    def __init__(self) -> None:
        """Initialize future list."""
        self._futures: List[asyncio.Future] = []

    def add(self, future: asyncio.Future) -> None:
        """Add a future to list."""
        self._futures.append(future)

    def as_completed(
        self, stop: Optional[Callable[[], bool]] = None
    ) -> Optional[asyncio.Future]:
        """Get future as they are done.

        Args:
            stop: if given, stop waiting once it returns True

        Return:
            return a future from the list, whenever it is done
            or return None when the list is empty, or when `stop`
            returns True.
        """
        while self._futures:
            if stop is not None and stop():
                return None
            for fut in list(self._futures):
                if fut.done():
                    self._futures.remove(fut)
                    return fut
            # Sleep between checks, or the loop keeps a core busy.
            time.sleep(0.005)
        return None

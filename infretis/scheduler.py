"""The main infretis loop."""

import copy
import logging
import signal
import threading
from types import FrameType
from typing import Optional

from infretis.setup import setup_internal, setup_runner

logger = logging.getLogger("main")

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class StopRequest:
    """Note SIGTERM and SIGINT instead of dying of them.

    While installed (as a context manager, in the main thread only), a
    signal is recorded and the scheduler stops between moves: a result
    being recorded is saved with restart.toml first. A batch system's
    walltime warning is usually SIGTERM.
    """

    def __init__(self) -> None:
        """Initialize: no signal yet."""
        self.signum: Optional[int] = None
        self._previous: dict = {}

    def __enter__(self) -> "StopRequest":
        """Take over the stop signals."""
        if threading.current_thread() is threading.main_thread():
            for signum in STOP_SIGNALS:
                self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Give the stop signals back."""
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def _handle(self, signum: int, _frame: Optional[FrameType]) -> None:
        if self.signum is None:
            logger.info(
                "%s received: stopping. Moves not yet saved are redone "
                "after a restart from restart.toml.",
                signal.Signals(signum).name,
            )
        self.signum = signum

    def requested(self) -> bool:
        """Whether a stop signal arrived."""
        return self.signum is not None


def scheduler(config, executor_factory=None) -> Optional[int]:
    """Run infretis loop.

    Args:
        config: the configuration dictionary
        executor_factory: a callable that takes the arguments of
            concurrent.futures.ProcessPoolExecutor and returns the
            executor that runs the MD moves (default: ProcessPoolExecutor)

    Returns:
        None when the run is done, or the number of the signal (SIGTERM
        or SIGINT) that stopped it. A stopped run continues from
        restart.toml.
    """
    with StopRequest() as stop:
        # setup repex, runner and futures
        md_items, state = setup_internal(config)
        runner, futures = setup_runner(state, executor_factory)

        # submit the first number of workers
        while not stop.requested() and state.initiate():
            # give each worker its own md_items
            worker_md_items = copy.deepcopy(md_items)
            # pick and prep ens and path for the next job
            worker_md_items = state.prep_md_items(worker_md_items)

            # submit job to scheduler
            futures.add(runner.submit_work(worker_md_items))

        # main step loop
        while not stop.requested() and state.loop():
            # Get futures as they are completed
            future = futures.as_completed(stop=stop.requested)

            if future:
                worker_md_items = state.treat_output(future.result())

            if stop.requested():
                break

            # submit new job
            if state.cstep + state.workers <= state.tsteps:
                # chose ens and path for the next job
                worker_md_items = state.prep_md_items(worker_md_items)

                # submit job to scheduler
                futures.add(runner.submit_work(worker_md_items))

        if stop.requested():
            runner.cancel()
            return stop.signum

        # end client; a signal from here on comes after the run is done,
        # and is ignored
        runner.stop()
    return None

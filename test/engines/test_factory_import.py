"""Test that the engine factory imports on its own."""

import subprocess
import sys


def test_factory_imports_first() -> None:
    """The factory imports in a fresh interpreter as the first import.

    It uses importlib.util, which `import importlib` alone does not
    provide unless another module imported importlib.util before.
    """
    subprocess.run(
        [sys.executable, "-c", "import infretis.classes.engines.factory"],
        check=True,
    )

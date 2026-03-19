"""pytest configuration: path setup runs before test collection.

The project has a naming collision:
  - /home/user/dife/dife.py          — root-level DIFE equation module
  - memory-vortex-dife-lab/dife/     — dife package (exports same functions + controller)

pytest adds the project root to sys.path early, causing dife.py to shadow the
package. The pytest_configure hook runs before any test file is imported, so we
can forcibly ensure memory-vortex-dife-lab is at sys.path[0].

The memory-vortex dife package exports the same dife() function as dife.py
(via dife/__init__.py → dife/core.py), so all existing code works correctly.
"""

import os
import sys


def pytest_configure(config):
    """Insert memory-vortex-dife-lab at sys.path[0] before any imports happen.

    Also clears any stale dife module cache so the package is found first.
    """
    mv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "memory-vortex-dife-lab")
    )

    # Remove existing entry if present (to force position 0)
    if mv_path in sys.path:
        sys.path.remove(mv_path)
    sys.path.insert(0, mv_path)

    # If dife was already cached as the root .py module (not the package),
    # evict it so the re-import picks up the package version.
    if "dife" in sys.modules:
        cached = sys.modules["dife"]
        cached_file = getattr(cached, "__file__", "") or ""
        if not cached_file.endswith("__init__.py"):
            keys_to_del = [k for k in sys.modules if k == "dife" or k.startswith("dife.")]
            for k in keys_to_del:
                del sys.modules[k]

"""Native-library import-order shim.  Import this before ``faiss``.

Why this exists
---------------
On macOS arm64 (and occasionally on Linux), ``faiss-cpu`` and PyTorch each bundle
their own OpenMP runtime.  If ``faiss`` is imported *first*, the process aborts
with a bus error / ``SIGABRT`` as soon as PyTorch's encoder runs — a hard crash
with no Python traceback, which is exactly the kind of failure that is
miserable to debug.

Importing ``torch`` before ``faiss`` lets PyTorch claim its OpenMP runtime
first, after which both libraries coexist.  That is all this module does.

It is safe and cheap when PyTorch is absent: the import simply fails and we
carry on.  ``KMP_DUPLICATE_LIB_OK`` is set as a belt-and-braces measure for the
Intel-Mac and Linux cases where the duplicate-runtime check is fatal rather
than merely noisy; it is scoped to this process and has no effect on results.
"""

from __future__ import annotations

import os
import sys

#: True when PyTorch was successfully preloaded.
TORCH_PRELOADED = False

#: True when this shim ran on a platform where the ordering matters.
ORDERING_REQUIRED = sys.platform == "darwin"


def preload_native_libraries() -> None:
    """Import PyTorch (if present) before anything imports FAISS."""
    global TORCH_PRELOADED

    if TORCH_PRELOADED or "torch" in sys.modules:
        TORCH_PRELOADED = True
        return

    # Allow duplicate OpenMP runtimes rather than aborting.  Must be set before
    # the offending library is loaded.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Keep thread pools modest; this app is latency-sensitive, not throughput-bound.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch  # noqa: F401

        TORCH_PRELOADED = True
    except ImportError:
        # No PyTorch installed — the hashing fallback is in use, so there is no
        # conflicting runtime to worry about.
        TORCH_PRELOADED = False
    except Exception:
        # A broken torch install must not take the whole app down; the local
        # embedding provider reports its own load failure with a clear remedy.
        TORCH_PRELOADED = False


# Run on import so that `import hiring_assistant._compat` is sufficient.
preload_native_libraries()


__all__ = ["ORDERING_REQUIRED", "TORCH_PRELOADED", "preload_native_libraries"]

"""Small, safe adapters around the pybind11 ``redmax_py`` extension."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional, Tuple

from api.errors import NativeBackendUnavailable

logger = logging.getLogger(__name__)


def native_status() -> Tuple[bool, Optional[str]]:
    try:
        importlib.import_module("redmax_py")
        return True, None
    except Exception as exc:  # ImportError is common; broken shared libraries also arrive here.
        return False, str(exc)


def require_native() -> Any:
    try:
        return importlib.import_module("redmax_py")
    except Exception as exc:
        raise NativeBackendUnavailable(
            "redmax_py could not be loaded. Build it with `pip install ./simulation` and check native runtime libraries."
        ) from exc

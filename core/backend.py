"""Rust backend detection and feature flag.

Set _USE_RUST_CORE = True to use the Rust-accelerated hot path.
Falls back to pure Python automatically if the Rust extension is not built.

Build the Rust extension:
    cd rust && maturin develop --release

Force Python fallback (even if Rust is available):
    export MICROSTRUCTURE_FORCE_PYTHON=1
"""

import os
import logging

logger = logging.getLogger(__name__)

_USE_RUST_CORE: bool = False
_rust_core = None

if not os.environ.get("MICROSTRUCTURE_FORCE_PYTHON"):
    try:
        import _microstructure_core as _rust_core
        _USE_RUST_CORE = True
        logger.info("Rust backend loaded: _microstructure_core")
    except ImportError:
        logger.info("Rust backend not available, using pure Python")
else:
    logger.info("Rust backend disabled by MICROSTRUCTURE_FORCE_PYTHON")


def get_rust_core():
    """Get the Rust module, or None if not available."""
    return _rust_core


def is_rust_available() -> bool:
    """Check if the Rust backend is loaded."""
    return _USE_RUST_CORE

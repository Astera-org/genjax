"""Compatibility helpers for third-party API transitions.

These shims are kept minimal and can be removed once upstream dependencies
(namely TensorFlow Probability) support newer JAX APIs directly.
"""

from __future__ import annotations

import contextlib
import warnings

import jax


@contextlib.contextmanager
def suppress_tfp_dtype_warning():
    """Silence TFP's benign float64-truncation ``UserWarning`` while sampling.

    Some TFP samplers (e.g. ``Poisson``) cast their parameters to the backend's
    ``internal_dtype`` (float64), which JAX harmlessly truncates back to float32
    when x64 is disabled. This emits a ``UserWarning`` that a consumer's
    ``error::UserWarning`` filter would otherwise escalate into a hard failure.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype.*is not available.*truncated",
            category=UserWarning,
        )
        yield


def ensure_jax_tfp_compat() -> None:
    """Install small compatibility shims needed by current TFP on JAX >= 0.7.

    TFP 0.25 still references ``jax.interpreters.xla.pytype_aval_mappings``,
    which was removed in JAX 0.7 in favor of ``jax.core.pytype_aval_mappings``.
    """

    xla_interpreter = jax.interpreters.xla
    if not hasattr(xla_interpreter, "pytype_aval_mappings"):
        xla_interpreter.pytype_aval_mappings = jax.core.pytype_aval_mappings

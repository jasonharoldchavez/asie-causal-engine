"""ASIE experimental predictive engine package.

The pipeline is imported lazily so low-level predictor modules remain usable
while the separately versioned Lumen integration is being reconciled.
"""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "VerifiableASIEPipeline":
        from .pipeline import VerifiableASIEPipeline
        return VerifiableASIEPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VerifiableASIEPipeline"]

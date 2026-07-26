import numpy as np
import time

from lumen.ledger import Ledger
from lumen.verifier import Verifier
from .core import NonLinearInvariantCausalEngine

class VerifiableASIEPipeline:
    def __init__(self, num_signals: int, ledger_path: str = "asie_ledger.json"):
        self.num_signals = num_signals
        self.engine = NonLinearInvariantCausalEngine(num_signals=num_signals)
        self.ledger = Ledger(storage_path=ledger_path)
        self.verifier = Verifier()

    def ingest_tick(self, signal_vector: np.ndarray):
        step_result = self.engine.step(signal_vector)
        if step_result is None:
            return None
            
        causal_matrix, predictions = step_result
        invariance_score = self.engine._compute_structural_invariance()

        event_payload = {
            "timestamp": time.time(),
            "signals": signal_vector.tolist(),
            "causal_matrix": causal_matrix.tolist(),
            "predictions": predictions.tolist(),
            "invariance_score": float(invariance_score)
        }

        entry_hash = self.ledger.append_entry(event_payload)

        return {
            "causal_matrix": causal_matrix,
            "predictions": predictions,
            "invariance_score": invariance_score,
            "merkle_hash": entry_hash
        }

    def verify_audit_ledger(self) -> bool:
        return self.verifier.verify_ledger(self.ledger.storage_path)

"""Async bridge between the ASIE predictor and the Lumen SQLite ledger.

The model output is a predictive-weight matrix, not a validated causal graph.
A caller must supply a key provider backed by its own secure key management.
"""

import hashlib
import json
import time
import uuid

from lumen.ledger import SQLiteLedgerEngine
from lumen.merkle import MerkleTreeEngine
from lumen.verifier import ForensicAuditVerifier

from .core import NonLinearInvariantCausalEngine


class VerifiableASIEPipeline:
    """Predictive ASIE pipeline with explicit async Lumen persistence."""

    def __init__(
        self,
        num_signals: int,
        key_provider,
        key_id: str,
        ledger_path: str = "asie_ledger.sqlite3",
        model_options=None,
    ):
        if not isinstance(key_id, str) or not key_id.strip():
            raise ValueError("key_id must be a non-empty string.")
        if key_provider is None or not callable(getattr(key_provider, "get_key_material", None)):
            raise ValueError("key_provider must implement get_key_material(key_id).")
        self.num_signals = num_signals
        self.key_provider = key_provider
        self.key_id = key_id
        self.ledger_path = ledger_path
        self.engine = NonLinearInvariantCausalEngine(
            num_signals=num_signals, **(model_options or {})
        )
        self.storage = SQLiteLedgerEngine(path=ledger_path, key_provider=key_provider)
        self.verifier = ForensicAuditVerifier(self.storage)
        self._initialized = False
        self._record_version = 0
        self._last_timestamp_nanos = 0

    async def initialize(self):
        """Open/migrate the ledger and restore monotonic record counters."""
        if self._initialized:
            return
        await self.storage.initialize()
        async with self.storage._read_conn.execute(
            "SELECT version FROM event_journal WHERE node=? AND metric=? ORDER BY version DESC LIMIT 1",
            ("asie_engine", "predictive_output"),
        ) as cursor:
            row = await cursor.fetchone()
            self._record_version = int(row[0]) if row else 0
        async with self.storage._read_conn.execute(
            "SELECT timestamp_nanos FROM transaction_manifest ORDER BY logical_epoch DESC LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
            self._last_timestamp_nanos = int(row[0]) if row else 0
        self._initialized = True

    async def ingest_tick(self, signal_vector):
        """Ingest one observation and persist a signed event when prediction is ready.

        Returns None during model warm-up. Otherwise returns the prediction,
        predictive-weight matrix, invariance score, transaction ID, and manifest
        hash after the event has been committed.
        """
        if not self._initialized:
            raise RuntimeError("Call and await initialize() before ingest_tick().")
        step_result = self.engine.step(signal_vector)
        if step_result is None:
            return None

        predictive_weight_matrix, predictions = step_result
        invariance_score = self.engine._compute_structural_invariance()
        timestamp = time.time()
        timestamp_nanos = max(time.time_ns(), self._last_timestamp_nanos + 1)
        self._last_timestamp_nanos = timestamp_nanos

        event_payload = {
            "timestamp": timestamp,
            "signals": self.engine.buffer[-1].tolist(),
            "predictive_weight_matrix": predictive_weight_matrix.tolist(),
            "predictions": predictions.tolist(),
            "invariance_score": float(invariance_score),
            "matrix_semantics": "predictive_associations_not_identified_causal_effects",
            "matrix_orientation": "C[source, target] (legacy orientation until the predictor patch is merged)",
        }
        value_bytes = json.dumps(
            event_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        metadata = json.dumps(
            {"record_type": "asie_predictive_output", "schema_version": 1},
            sort_keys=True, separators=(",", ":"),
        )

        node = "asie_engine"
        metric = "predictive_output"
        self._record_version += 1
        version = self._record_version
        tx_id = uuid.uuid4().hex
        parent_hash = await self.storage.get_last_journal_hash(node, metric)
        raw_journal_payload = (
            f"{tx_id}:{node}:{metric}:{version}:{parent_hash}".encode("utf-8")
            + value_bytes
        )
        journal_hash = hashlib.sha256(raw_journal_payload).hexdigest()

        last_epoch, previous_manifest_hash = await self.storage.get_last_manifest_metadata()
        epoch = int(last_epoch) + 1
        merkle_root = MerkleTreeEngine.compute_root([journal_hash])
        manifest_hash, manifest_hmac = self.storage.sign_manifest(
            tx_id, epoch, timestamp_nanos, 1, 0, previous_manifest_hash,
            self.key_id, merkle_root, "SHA256", "v1", "v1",
        )
        journal_hmac = self.storage.sign_journal_envelope(
            tx_id, epoch, 0, timestamp, node, metric, version, value_bytes,
            "ASIEPredictiveEngine", "0.1.0", metadata, parent_hash,
            journal_hash, manifest_hmac, self.key_id,
        )
        journal_row = (
            tx_id, timestamp, node, metric, version, value_bytes,
            "ASIEPredictiveEngine", "0.1.0", metadata, parent_hash,
            journal_hash, journal_hmac, self.key_id, 0,
        )
        await self.storage.save_compliance_batch_explicit(
            tx_id, epoch, timestamp_nanos, 1, 0, manifest_hash, manifest_hmac,
            self.key_id, previous_manifest_hash, merkle_root, [journal_row], [], [],
        )
        return {
            "predictive_weight_matrix": predictive_weight_matrix,
            "predictions": predictions,
            "invariance_score": invariance_score,
            "transaction_id": tx_id,
            "manifest_hash": manifest_hash,
        }

    async def verify_audit_ledger(self):
        """Return the full Lumen audit report; EMPTY is not a successful verdict."""
        if not self._initialized:
            raise RuntimeError("Call and await initialize() before verification.")
        return await self.verifier.verify_complete_ledger_chain()

    async def close(self):
        if self._initialized:
            await self.storage.close()
            self._initialized = False

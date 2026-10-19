import os
import tempfile
import unittest

import numpy as np

from asie_engine.pipeline import VerifiableASIEPipeline


class TestKeyProvider:
    def get_key_material(self, key_id):
        if key_id != "test-key":
            raise KeyError("unknown key")
        return b"pipeline-integration-test-key"


class PipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        self.pipeline = VerifiableASIEPipeline(
            num_signals=2,
            key_provider=TestKeyProvider(),
            key_id="test-key",
            ledger_path=self.db_path,
            model_options={
                "train_window": 5,
                "max_lags": 2,
                "latent_factors": 0,
                "epochs": 1,
                "random_state": 17,
            },
        )
        await self.pipeline.initialize()

    async def asyncTearDown(self):
        await self.pipeline.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(self.db_path + suffix)
            except FileNotFoundError:
                pass

    async def test_empty_ledger_is_not_reported_verified(self):
        report = await self.pipeline.verify_audit_ledger()
        self.assertEqual(report["status"], "EMPTY")

    async def test_prediction_is_written_and_audit_verifies(self):
        observations = np.random.default_rng(10).normal(size=(7, 2))
        last_result = None
        for observation in observations:
            last_result = await self.pipeline.ingest_tick(observation)

        self.assertIsNotNone(last_result)
        self.assertIn("predictive_weight_matrix", last_result)
        self.assertIn("predictions", last_result)
        report = await self.pipeline.verify_audit_ledger()
        self.assertEqual(report["status"], "VERIFIED", report["errors"])
        self.assertEqual(report["manifests_processed"], 1)


if __name__ == "__main__":
    unittest.main()

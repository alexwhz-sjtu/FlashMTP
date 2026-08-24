import argparse
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from specforge import tracker


class WandbOfflineTrackerTest(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            wandb_key=None,
            wandb_project="flashmtp-test",
            wandb_name="offline-run",
            wandb_run_id="offline-id",
        )

    def test_offline_mode_needs_no_key_and_skips_login(self):
        args = self._args()
        wandb = mock.Mock()
        with (
            mock.patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=True),
            mock.patch.object(tracker, "wandb", wandb),
            mock.patch.object(tracker.dist, "get_rank", return_value=0),
        ):
            tracker.WandbTracker.validate_args(argparse.ArgumentParser(), args)
            instance = tracker.WandbTracker(args, "/tmp/output")

        wandb.login.assert_not_called()
        wandb.init.assert_called_once()
        self.assertTrue(instance.is_initialized)

    def test_online_mode_still_requires_credentials(self):
        args = self._args()
        with (
            mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=True),
            mock.patch.object(tracker, "wandb", mock.Mock()),
            mock.patch.object(tracker.os.path, "exists", return_value=False),
        ):
            with self.assertRaises(SystemExit):
                tracker.WandbTracker.validate_args(
                    argparse.ArgumentParser(), args
                )


if __name__ == "__main__":
    unittest.main()

"""Diagnostic ablation: replace trained draft convolutions with identity maps."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from specforge.modeling.draft.flashmtp import FlashMTPGroupedConv


def identity_prepare(self, hidden_states):
    return hidden_states, None


def identity_finish(self, hidden_states, coefficients):
    return hidden_states


FlashMTPGroupedConv.prepare = identity_prepare
FlashMTPGroupedConv.finish = identity_finish

from evaluation.benchmark import main

main()

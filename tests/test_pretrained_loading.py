"""Tests for loading pretrained model weights."""

import os
import pickle
import unittest

import torch
import torch.nn.functional as F

from pyengine.utils.pretrained_loading import (
    PRETRAINED_RL_CONFIG,
    PRETRAINED_ARRANGEMENT_CONFIG,
    PRETRAINED_RL_PTHW,
    PRETRAINED_INIT_PTHW,
    PRETRAINED_ARRANGEMENTS_PKL,
    load_pretrained_rl_model,
    load_pretrained_arrangement_model,
)
from pyengine.utils.loading import load_rl_model, load_arrangement_model
from pyengine.networks.legacy_rl import TransformerRL
from pyengine.networks.legacy_init import TransformerInitialization
from pyengine.arrangement.utils import to_string
from pyengine.utils.constants import ARRANGEMENT_SIZE, N_PIECE_TYPE
from pyengine.utils import get_pystratego

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
RL_TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, "rl_test_data.pkl")

pystratego = get_pystratego()


class TestPretrainedRLModelConfig(unittest.TestCase):
    """Test that pretrained RL config matches expected values."""

    def test_rl_config_depth(self):
        self.assertEqual(PRETRAINED_RL_CONFIG["rl_transformer"]["depth"], 8)

    def test_rl_config_n_head(self):
        self.assertEqual(PRETRAINED_RL_CONFIG["rl_transformer"]["n_head"], 8)

    def test_rl_config_embed_dim(self):
        self.assertEqual(PRETRAINED_RL_CONFIG["rl_transformer"]["embed_dim_per_head_over8"], 6)

    def test_rl_config_plane_history_len(self):
        self.assertEqual(PRETRAINED_RL_CONFIG["rl_transformer"]["plane_history_len"], 32)


class TestPretrainedInitModelConfig(unittest.TestCase):
    """Test that pretrained init config matches expected values."""

    def test_init_config_depth(self):
        self.assertEqual(PRETRAINED_ARRANGEMENT_CONFIG["init_transformer"]["depth"], 4)

    def test_init_config_n_head(self):
        self.assertEqual(PRETRAINED_ARRANGEMENT_CONFIG["init_transformer"]["n_head"], 8)

    def test_init_config_embed_dim(self):
        self.assertEqual(
            PRETRAINED_ARRANGEMENT_CONFIG["init_transformer"]["embed_dim_per_head_over8"], 8
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@unittest.skipUnless(os.path.exists(RL_TEST_DATA_PATH), "RL test data not found")
class TestPretrainedRLModelOutput(unittest.TestCase):
    """Test that RL model outputs match microstratego outputs."""

    @classmethod
    def setUpClass(cls):
        with open(RL_TEST_DATA_PATH, "rb") as f:
            cls.test_data = pickle.load(f)

    def test_pthw_outputs_match(self):
        model, _arrangements = load_pretrained_rl_model(str(PRETRAINED_RL_PTHW))
        model.eval()
        self.assertIsInstance(model, TransformerRL)

        observations = self.test_data["observations"].cuda()
        piece_ids = self.test_data["piece_ids"].cuda()
        legal_actions = self.test_data["legal_actions"].cuda()
        expected_logits = self.test_data["logits"].cuda()
        expected_value = self.test_data["value"].cuda()

        with torch.no_grad():
            logits, value = model.forward_main(observations, piece_ids, legal_actions)

        torch.testing.assert_close(
            logits,
            expected_logits,
            rtol=1e-4,
            atol=1e-4,
            msg="RL model logits do not match microstratego outputs",
        )
        torch.testing.assert_close(
            value,
            expected_value,
            rtol=1e-4,
            atol=1e-4,
            msg="RL model value does not match microstratego outputs",
        )

    def test_pthw_load_via_generic_loader(self):
        model, arrangements = load_rl_model(str(PRETRAINED_RL_PTHW))
        self.assertIsInstance(model, TransformerRL)
        with open(PRETRAINED_ARRANGEMENTS_PKL, "rb") as f:
            expected_arrangements = pickle.load(f)
        self.assertEqual(arrangements, expected_arrangements)


def _sample_arrangements(model: TransformerInitialization, num_samples: int) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    seq = torch.zeros((num_samples, 0, N_PIECE_TYPE), device=device)
    for t in range(model.seq_len):
        logits, _value, _reg = model(seq)
        next_logits = logits[:, t]
        probs = torch.softmax(next_logits, dim=-1)
        idx = torch.multinomial(probs, 1).squeeze(-1)
        one_hot = F.one_hot(idx, num_classes=N_PIECE_TYPE).float().unsqueeze(1)
        seq = torch.cat([seq, one_hot], dim=1)
    return seq


def _arrangements_to_onehot(arrangements: list[str], device: torch.device) -> torch.Tensor:
    ids = pystratego.util.arrangement_tensor_from_strings(arrangements)
    ids = torch.as_tensor(ids, device=device, dtype=torch.long)
    return F.one_hot(ids, num_classes=N_PIECE_TYPE).float()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestPretrainedInitModelOutput(unittest.TestCase):
    """Test that init model outputs are similar to saved arrangements."""

    def _assert_similarity(self, weight_path: str, arrangements_path: str):
        model = load_pretrained_arrangement_model(weight_path)
        self.assertIsInstance(model, TransformerInitialization)

        with open(arrangements_path, "rb") as f:
            ref_arrangements = pickle.load(f)

        # Use a fixed subset for determinism
        torch.manual_seed(0)
        ref_subset = ref_arrangements[0][:256]

        generated = _sample_arrangements(model, num_samples=len(ref_subset))
        gen_onehot = generated.detach()
        ref_onehot = _arrangements_to_onehot(ref_subset, gen_onehot.device)

        self.assertEqual(gen_onehot.shape, (len(ref_subset), ARRANGEMENT_SIZE, N_PIECE_TYPE))
        self.assertEqual(ref_onehot.shape, (len(ref_subset), ARRANGEMENT_SIZE, N_PIECE_TYPE))

        freq_gen = gen_onehot.mean(dim=0)
        freq_ref = ref_onehot.mean(dim=0)
        l1 = (freq_gen - freq_ref).abs().mean().item()
        self.assertLess(l1, 0.1, f"Arrangement distribution drift too large: {l1}")

        # Basic sanity: generated arrangements are valid strings
        gen_strings = to_string(gen_onehot)
        self.assertEqual(len(gen_strings), len(ref_subset))

    def test_pthw_similarity(self):
        self._assert_similarity(str(PRETRAINED_INIT_PTHW), str(PRETRAINED_ARRANGEMENTS_PKL))

    def test_pthw_load_via_generic_loader(self):
        model = load_arrangement_model(str(PRETRAINED_INIT_PTHW))
        self.assertIsInstance(model, TransformerInitialization)


if __name__ == "__main__":
    unittest.main()

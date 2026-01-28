# Utilities for loading Ataraxos July 2025 checkpoint weights.

import os
from pathlib import Path
from typing import Any, Optional
import pickle
from collections import OrderedDict

import torch

from pyengine.networks.legacy_belief import ARBelief, ARBeliefConfig
from pyengine.networks.legacy_init import TransformerInitConfig, TransformerInitialization
from pyengine.networks.legacy_rl import TransformerRLConfig, TransformerRL
from pyengine.utils.init_helpers import COUNTERS

# Paths to the shipped pretrained artifacts.
PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "pretrained" / "final_run"
PRETRAINED_RL_PTHW = PRETRAINED_DIR / "model42400.pthw"
PRETRAINED_RL_PTHM = PRETRAINED_DIR / "model42400.pthm"
PRETRAINED_INIT_PTHW = PRETRAINED_DIR / "init_model42400.pthw"
PRETRAINED_INIT_PTHM = PRETRAINED_DIR / "init_model42400.pthm"
PRETRAINED_ARRANGEMENTS_PKL = PRETRAINED_DIR / "arrangements42400.pkl"
PRETRAINED_ARRANGEMENTS_EMA_PKL = PRETRAINED_DIR / "ema_arrangements42400.pkl"

# Hardcoded configs for the pretrained models (from microstratego training)
PRETRAINED_RL_CONFIG = {
    "barrage": 0,
    "rl_transformer": {
        "depth": 8,
        "embed_dim_per_head_over8": 6,
        "n_head": 8,
        "dropout": 0.0,
        "pos_emb_std": 0.1,
        "ff_factor": 4,
        "plane_history_len": 32,
        "use_piece_ids": 1,
        "legacy": 0,
        "protect_legacy": 0,
        "use_threaten": 1,
        "use_evade": 1,
        "use_actadj": 1,
        "use_battle": 1,
        "use_cemetery": 1,
        "use_protect": 1,
    },
}

PRETRAINED_BELIEF_CONFIG = {
    "ar_belief": {
        "depth": 6,
        "num_head": 8,
        "embed_dim": 512,
        "dropout": 0.2,
        "mask": 0,
        "plane_history_len": 86,
        "decoder_depth": 4,
    },
}

PRETRAINED_ARRANGEMENT_CONFIG = {
    "barrage": 0,
    "init_transformer": {
        "depth": 4,
        "embed_dim_per_head_over8": 8,
        "n_head": 8,
        "dropout": 0.0,
        "pos_emb_std": 0.1,
        "force_handedness": 1,
        "use_value_net": 1,
        "weight_counts": 1,
    },
}


def get_checkpoint_step(checkpoint: str) -> int:
    """Extract the checkpoint step number from a checkpoint filename."""
    filename = Path(checkpoint).stem
    for prefix in ["model", "belief", "init_model"]:
        if filename.startswith(prefix):
            return int(filename[len(prefix):])
    raise ValueError(f"Could not extract checkpoint step from {checkpoint}")


def load_state_dict(model, state_dict):
    """Load state dict with handling for DDP and torch.compile prefixes."""
    model_keys = list(model.state_dict().keys())
    if model_keys and "_orig_mod." not in model_keys[0]:
        state_dict = remove_string(state_dict, "_orig_mod.")
    if model_keys and "module." not in model_keys[0]:
        state_dict = remove_string(state_dict, "module.")
    model.load_state_dict(state_dict)


def remove_string(dictionary: OrderedDict[str, Any], string: str) -> dict[str, Any]:
    new_dict = OrderedDict()
    for k, v in dictionary.items():
        if string in k:
            new_key = k.replace(string, "")
            new_dict[new_key] = v
        else:
            new_dict[k] = v
    return new_dict


def load_pretrained_rl_model(
    fn: str,
    rank: Optional[int] = None,
) -> tuple[TransformerRL, tuple[list[str], list[str]]]:
    """Load a pretrained RL model from a known checkpoint.

    Args:
        fn: Path to the model weights file.
        rank: CUDA device rank. If None, uses 'cuda'.

    Returns:
        Tuple of (model, arrangements) where arrangements is a tuple of two lists.
    """
    if rank is not None:
        device = f"cuda:{rank}"
    else:
        device = "cuda"

    config = PRETRAINED_RL_CONFIG
    rl_cfg = config["rl_transformer"]

    if config.get("barrage", 0):
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device=device)
    else:
        piece_counts = torch.tensor(COUNTERS["classic"] + [0, 0], device=device)

    net = TransformerRL(
        piece_counts=piece_counts,
        cfg=TransformerRLConfig(
            depth=rl_cfg["depth"],
            embed_dim_per_head_over8=rl_cfg["embed_dim_per_head_over8"],
            n_head=rl_cfg["n_head"],
            dropout=rl_cfg["dropout"],
            pos_emb_std=rl_cfg["pos_emb_std"],
            ff_factor=rl_cfg["ff_factor"],
            plane_history_len=rl_cfg["plane_history_len"],
            use_piece_ids=bool(rl_cfg["use_piece_ids"]),
            legacy=bool(rl_cfg["legacy"]),
            protect_legacy=bool(rl_cfg["protect_legacy"]),
            use_threaten=bool(rl_cfg.get("use_threaten", True)),
            use_evade=bool(rl_cfg.get("use_evade", True)),
            use_actadj=bool(rl_cfg.get("use_actadj", True)),
            use_battle=bool(rl_cfg.get("use_battle", True)),
            use_cemetery=bool(rl_cfg.get("use_cemetery", True)),
            use_protect=bool(rl_cfg.get("use_protect", True)),
        ),
    )
    net.to(device)
    load_state_dict(net, torch.load(fn, map_location=device, weights_only=True))

    # Load arrangements if available
    log_dir = str(Path(fn).parent)
    ema_prefix = "ema_" if "pthm" in fn else ""
    checkpoint = get_checkpoint_step(fn)
    arrangements_path = f"{log_dir}/{ema_prefix}arrangements{checkpoint}.pkl"

    if os.path.exists(arrangements_path):
        with open(arrangements_path, "rb") as f:
            arrangements = pickle.load(f)
    else:
        # Return empty arrangements if not found
        arrangements = ([], [])

    return net, arrangements


def load_pretrained_arrangement_model(
    fn: str,
    rank: Optional[int] = None,
) -> TransformerInitialization:
    """Load a pretrained arrangement/init model from a known checkpoint.

    Args:
        fn: Path to the model weights file.
        rank: CUDA device rank. If None, uses 'cuda'.

    Returns:
        The loaded TransformerInitialization model.
    """
    if rank is not None:
        device = f"cuda:{rank}"
    else:
        device = "cuda"

    config = PRETRAINED_ARRANGEMENT_CONFIG
    init_cfg = config["init_transformer"]

    if config.get("barrage", 0):
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device=device)
    else:
        piece_counts = torch.tensor(COUNTERS["classic"] + [0, 0], device=device)

    net = TransformerInitialization(
        piece_counts=piece_counts,
        cfg=TransformerInitConfig(
            embed_dim_per_head_over8=init_cfg["embed_dim_per_head_over8"],
            depth=init_cfg["depth"],
            n_head=init_cfg["n_head"],
        ),
    )
    net.to(device)
    load_state_dict(net, torch.load(fn, map_location=device, weights_only=True))

    return net

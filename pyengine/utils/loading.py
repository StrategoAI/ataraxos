import os
from pathlib import Path
from typing import Any, Optional
import pickle
from collections import OrderedDict
import yaml

import torch

from pyengine.core.env import Stratego
from pyengine.networks import legacy_belief
from pyengine.networks.temporal_belief_transformer import (
    TemporalBeliefTransformer,
    TemporalBeliefConfig,
)
from pyengine.networks.belief_transformer import BeliefTransformer, BeliefTransformerConfig
from pyengine.networks.legacy_init import TransformerInitConfig, TransformerInitialization
from pyengine.networks.arrangement_transformer import (
    ArrangementTransformer,
    ArrangementTransformerConfig,
)
from pyengine.networks.legacy_rl import TransformerRLConfig, TransformerRL
from pyengine.utils import get_pystratego
from pyengine.utils.init_helpers import is_valid_line, CHAR_TO_VAL, COUNTERS
from pyengine.utils.pretrained_loading import (
    PRETRAINED_RL_PTHW,
    PRETRAINED_RL_PTHM,
    PRETRAINED_INIT_PTHW,
    PRETRAINED_INIT_PTHM,
    load_pretrained_rl_model,
    load_pretrained_arrangement_model,
)

pystratego = get_pystratego()

# These are training keys whose values need to be evaluated.
EVAL_KEYS = ["obs_shape", "arrangement_rng_state", "eval_against"]


def is_castable_to_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_castable_to_float(s: str) -> bool:
    try:
        float(s)  # Attempt to convert the string to an integer
        return True
    except ValueError:
        return False  # If a ValueError is raised, it means the conversion failed


def extract_train_info(log_content):
    """Extracts the training information between the specified delimiters."""
    start_delimiter = "===============train_info==============="
    end_delimiter = "========================================"
    start_index = log_content.find(start_delimiter) + len(start_delimiter)
    end_index = log_content.find(end_delimiter)
    return log_content[start_index:end_index].strip()


def convert_value(value):
    """Convert the value to the appropriate type."""
    if isinstance(value, str):
        return convert_string(value)
    elif isinstance(value, list):
        # Recursively convert list elements
        return [convert_value(v) for v in value]
    elif isinstance(value, dict):
        # Recursively convert dictionary elements
        return {k: convert_value(v) for k, v in value.items()}
    return value


def convert_string(value: str) -> int | float | str:
    for converter in [int, float]:
        try:
            return converter(value)
        except ValueError:
            continue
    return value


def get_train_info(log_dir: str) -> dict[str, Any]:
    """Load the log content and convert it into a dictionary with appropriate types."""
    with open(f"{log_dir}/train.log", "r") as file:
        log_content = file.read()
    train_info_content = extract_train_info(log_content)
    data = yaml.safe_load(train_info_content)
    return {k: convert_value(v) for k, v in data.items()}


def log_dir_from_fn(fn: str) -> str:
    return fn[: fn.rfind("/")]


def get_checkpoint_step(checkpoint: str) -> int:
    idx = checkpoint.split("/")[-1].split(".")[0][len("model") :]
    return int(idx)


def load_rl_model(
    fn: str, rank: Optional[int] = None
) -> tuple[TransformerRL, tuple[list[str], list[str]]]:
    pretrained_paths = {
        str(PRETRAINED_RL_PTHW.resolve()),
        str(PRETRAINED_RL_PTHM.resolve()),
    }
    if str(Path(fn).resolve()) in pretrained_paths:
        return load_pretrained_rl_model(fn, rank)

    log_dir = log_dir_from_fn(fn)
    train_info = get_train_info(log_dir)
    if rank is not None:
        device = f"cuda:{rank}"
    else:
        device = "cuda"
    if train_info["barrage"]:
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device=device)
    else:
        piece_counts = torch.tensor(COUNTERS["classic"] + [0, 0], device=device)
    if "legacy" not in train_info["rl"]["rl_transformer"]:
        train_info["rl"]["rl_transformer"]["legacy"] = 1
    if "protect_legacy" not in train_info["rl"]["rl_transformer"]:
        train_info["rl"]["rl_transformer"]["protect_legacy"] = 1
    net = TransformerRL(
        piece_counts=piece_counts,
        cfg=TransformerRLConfig(
            depth=train_info["rl"]["rl_transformer"]["depth"],
            embed_dim_per_head_over8=train_info["rl"]["rl_transformer"]["embed_dim_per_head_over8"],
            n_head=train_info["rl"]["rl_transformer"]["n_head"],
            dropout=train_info["rl"]["rl_transformer"]["dropout"],
            pos_emb_std=train_info["rl"]["rl_transformer"]["pos_emb_std"],
            ff_factor=train_info["rl"]["rl_transformer"]["ff_factor"],
            plane_history_len=train_info["rl"]["rl_transformer"]["plane_history_len"],
            use_piece_ids=train_info["rl"]["rl_transformer"]["use_piece_ids"],
            legacy=train_info["rl"]["rl_transformer"]["legacy"],
            protect_legacy=train_info["rl"]["rl_transformer"]["protect_legacy"],
        ),
    )
    net.to(device)
    load_state_dict(net, torch.load(fn, map_location=device))
    ema = "ema_" if "pthm" in fn else ""
    checkpoint = get_checkpoint_step(fn)
    with open(f"{log_dir}/{ema}arrangements{checkpoint}.pkl", "rb") as f:
        arrangements = pickle.load(f)
    return net, arrangements


def load_belief_model(fn: str, rank: Optional[int] = None):
    log_dir = log_dir_from_fn(fn)
    train_info = get_train_info(log_dir)
    if rank is not None:
        device = f"cuda:{rank}"
    else:
        device = "cuda"

    if "kl_coef" in train_info["belief"]["ar_belief"]:
        net = legacy_belief.ARBelief(
            cfg=legacy_belief.ARBeliefConfig(
                depth=train_info["belief"]["ar_belief"]["depth"],
                num_head=train_info["belief"]["ar_belief"]["num_head"],
                embed_dim=train_info["belief"]["ar_belief"]["embed_dim"],
                dropout=train_info["belief"]["ar_belief"]["dropout"],
                plane_history_len=train_info["belief"]["ar_belief"]["plane_history_len"],
            ),
        )
    elif "use_temporal_model" in train_info["belief"]:
        net = TemporalBeliefTransformer(
            max_num_moves=train_info["max_num_moves"],
            cfg=TemporalBeliefConfig(
                n_encoder_block=train_info["belief"]["temporal_belief"]["n_encoder_block"],
                n_decoder_block=train_info["belief"]["temporal_belief"]["n_decoder_block"],
                embed_dim_per_head_over8=train_info["belief"]["temporal_belief"][
                    "embed_dim_per_head_over8"
                ],
                n_head=train_info["belief"]["temporal_belief"]["n_head"],
                dropout=train_info["belief"]["temporal_belief"]["dropout"],
                pos_emb_std=train_info["belief"]["temporal_belief"]["pos_emb_std"],
                ff_factor=train_info["belief"]["temporal_belief"]["ff_factor"],
                only_grounded_features=train_info["belief"]["temporal_belief"][
                    "only_grounded_features"
                ],
            ),
        )
    else:
        net = BeliefTransformer(cfg=BeliefTransformerConfig())
    net.to(device)
    load_state_dict(net, torch.load(fn, map_location=device))
    net.eval()
    return net


def load_arrangement_model(fn: str, rank: Optional[int] = None) -> ArrangementTransformer:
    pretrained_paths = {
        str(PRETRAINED_INIT_PTHW.resolve()),
        str(PRETRAINED_INIT_PTHM.resolve()),
    }
    if str(Path(fn).resolve()) in pretrained_paths:
        return load_pretrained_arrangement_model(fn, rank)

    log_dir = log_dir_from_fn(fn)
    train_info = get_train_info(log_dir)
    if rank is not None:
        device = f"cuda:{rank}"
    else:
        device = "cuda"
    if train_info["barrage"]:
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device=device)
    else:
        piece_counts = torch.tensor(COUNTERS["classic"] + [0, 0], device=device)
    if "arr_transformer" in train_info["rl"]:
        net = ArrangementTransformer(
            piece_counts=piece_counts,
            cfg=ArrangementTransformerConfig(
                embed_dim_per_head_over8=train_info["rl"]["arr_transformer"][
                    "embed_dim_per_head_over8"
                ],
                use_cat_vf=train_info["rl"]["arr_transformer"]["use_cat_vf"],
                depth=train_info["rl"]["arr_transformer"]["depth"],
                n_head=train_info["rl"]["arr_transformer"]["n_head"],
            ),
        )
    else:
        net = TransformerInitialization(
            piece_counts=piece_counts,
            cfg=TransformerInitConfig(
                embed_dim_per_head_over8=train_info["rl"]["init_transformer"][
                    "embed_dim_per_head_over8"
                ],
                depth=train_info["rl"]["init_transformer"]["depth"],
                n_head=train_info["rl"]["init_transformer"]["n_head"],
            ),
        )
    net.to(device)
    load_state_dict(net, torch.load(fn, map_location=device))
    return net


def load_env(
    fn: str,
    num_envs: int,
    traj_len_per_player: Optional[int] = None,
    move_memory: Optional[int] = None,
    max_num_moves: Optional[int] = None,
    **kwargs,
) -> Stratego:
    log_dir = log_dir_from_fn(fn)
    train_info = get_train_info(log_dir)
    if traj_len_per_player is not None:
        train_info["rl"]["traj_len_per_player"] = traj_len_per_player
    if move_memory is not None:
        train_info["move_memory"] = move_memory
    if max_num_moves is not None:
        train_info["max_num_moves"] = max_num_moves
    return Stratego(
        num_envs=num_envs,
        traj_len_per_player=train_info["rl"]["traj_len_per_player"],
        full_info=train_info["full_info"],
        barrage=train_info["barrage"],
        move_memory=train_info["move_memory"],
        max_num_moves=train_info["max_num_moves"],
        **kwargs,
    )


def get_arrangements(train_info: dict[str, Any], log_dir: str) -> torch.Tensor:
    if ("random_inits" in train_info) and (train_info["random_inits"] > 0):
        if train_info["barrage"]:
            board_variant = pystratego.BoardVariant.BARRAGE
        else:
            board_variant = pystratego.BoardVariant.CLASSIC
        gen = pystratego.PieceArrangementGenerator(board_variant)
        with open(log_dir + "/" + "arrangement_ids.pkl", "rb") as f:
            ids = pickle.load(f)
        arrangements = gen.generate_arrangements(ids)
        return arrangements
    # Else use human inits
    if train_info["barrage"]:
        prefix = "barrage"
    else:
        prefix = "classic"
    init_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "src",
        "env",
        "inits",
        f"{prefix}_human_inits.dat",
    )
    with open(init_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip()[::-1] for line in lines if is_valid_line(line)]
    numerical_lines = [[CHAR_TO_VAL[c] for c in line] for line in lines]
    arrangements = torch.tensor(numerical_lines, dtype=torch.uint8, device="cuda")
    return arrangements


def tile_and_truncate_tensors(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x_batch, z_x = x.shape
    y_batch, z_y = y.shape

    # Ensure the second dimension (z) is the same for both tensors
    if z_x != z_y:
        raise ValueError("The second dimension of both tensors must be the same.")

    # Determine the larger batch size
    larger_batch_size = max(x_batch, y_batch)

    # Tile the tensor with the smaller batch size
    if x_batch < y_batch:
        x = x.repeat((y_batch // x_batch + 1, 1))[:larger_batch_size]
    elif y_batch < x_batch:
        y = y.repeat((x_batch // y_batch + 1, 1))[:larger_batch_size]

    return x, y


def load_state_dict(model, state_dict):
    if "_orig_mod." not in list(model.state_dict().keys())[0]:
        state_dict = remove_string(state_dict, "_orig_mod.")
    if "module." not in list(model.state_dict().keys())[0]:
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

import os
import socket
from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from ptflops import get_model_complexity_info

from .archs import create_model
from .options.options import parse
from .utils.test_utils import cleanup, setup

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, "options", "inference_video", "Baseline.yml")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    opt: dict
    device: torch.device


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def download_model(repo_id: str = "Cidaut/DarkIR", local_dir: str | None = None) -> None:
    if local_dir is None:
        local_dir = os.path.join(_THIS_DIR, "models")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)


def resolve_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    opt = parse(config_path)
    save_cfg = opt.get("save", {}) if isinstance(opt, dict) else {}
    for key in ("path", "best"):
        value = save_cfg.get(key)
        if isinstance(value, str) and not os.path.isabs(value):
            save_cfg[key] = os.path.join(_THIS_DIR, value.lstrip("./"))
    if isinstance(opt, dict):
        opt["save"] = save_cfg

    weights_path = save_cfg.get("path")
    if isinstance(weights_path, str) and not os.path.isfile(weights_path):
        download_model(local_dir=os.path.join(_THIS_DIR, "models"))

    return opt


def load_model(model: torch.nn.Module, path_weights: str) -> torch.nn.Module:
    checkpoints = torch.load(path_weights, map_location="cpu", weights_only=False)
    weights = checkpoints["params"]
    weights = {"module." + key: value for key, value in weights.items()}

    macs, params = get_model_complexity_info(
        model, (3, 256, 256), print_per_layer_stat=False, verbose=False
    )
    print("Complexity information of the model:", macs, params)
    model.load_state_dict(weights)
    print("Loaded weights correctly")
    return model


def load_ready_model(
    config_path: str = DEFAULT_CONFIG_PATH,
    cuda_visible_devices: str = "0",
    master_port: str | None = None,
) -> LoadedModel:
    """Download weights if needed, initialize DDP, and return a model ready for inference."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    opt = resolve_config(config_path)
    rank = 0
    setup(rank, world_size=1, Master_port=str(master_port or _find_free_port()))

    model, _, _ = create_model(opt["network"], rank=rank)
    model = load_model(model, path_weights=opt["save"]["path"])
    model.eval()

    device = (
        torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    )
    return LoadedModel(model=model, opt=opt, device=device)


def shutdown_ready_model() -> None:
    cleanup()

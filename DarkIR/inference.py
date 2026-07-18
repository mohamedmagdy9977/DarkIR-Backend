import os
import socket

from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, "options", "inference", "LOLBlur.yml")

import torch
import torch.multiprocessing as mp
from torchvision.transforms import Resize

from .data.dataset_reader.datapipeline import *
from .archs import *
from .utils.test_utils import *
from .download_model import LoadedModel, load_model, resolve_config

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def path_to_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return pil_to_tensor(img).unsqueeze(0)


def save_tensor(tensor: torch.Tensor, path: str) -> None:
    tensor = tensor.squeeze(0)
    img = tensor_to_pil(tensor)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)


def pad_tensor(tensor: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=0)


def apply_model(model, tensor: torch.Tensor, resize: bool = False) -> torch.Tensor:
    _, _, H, W = tensor.shape

    if resize and (H >= 1500 or W >= 1500):
        new_size = [int(dim // 2) for dim in (H, W)]
        downsample = Resize(new_size)
    else:
        downsample = torch.nn.Identity()

    tensor = downsample(tensor)
    tensor = pad_tensor(tensor)

    with torch.no_grad():
        output = model(tensor, side_loss=False)

    if resize:
        upsample = Resize((H, W))
    else:
        upsample = torch.nn.Identity()

    output = upsample(output)
    output = torch.clamp(output, 0.0, 1.0)
    return output[:, :, :H, :W]


def inference_image(
    rank,
    world_size,
    src_image: str,
    trgt_image: str,
    opt,
    model=None,
    master_port=None,
):
    """
    Run inference on a single image.
    Pass a pre-loaded `model` to skip loading weights on each call.
    """
    own_runtime = model is None
    if own_runtime:
        setup(rank, world_size=world_size, Master_port=str(master_port or 12354))

    try:
        if not os.path.isfile(src_image):
            raise FileNotFoundError(f"Source image not found: {src_image}")

        resize = opt["Resize"]

        if model is None:
            model, _, _ = create_model(opt["network"], rank=rank)
            model = load_model(model, path_weights=opt["save"]["path"])

        model.eval()

        tensor = path_to_tensor(src_image).to(device)
        output = apply_model(model, tensor, resize=resize)
        save_tensor(output, trgt_image)
        print(f"Finished inference! Saved to {trgt_image}")
        return trgt_image
    finally:
        if own_runtime:
            cleanup()


def run_low_light_img_inference(
    src_image: str,
    trgt_image: str,
    loaded_model: LoadedModel | None = None,
    config_path: str = DEFAULT_CONFIG_PATH,
    cuda_visible_devices: str = "0",
    world_size: int = 1,
) -> str:
    """
    Programmatic entrypoint (no CLI args).
    Pass `loaded_model` from `load_ready_model()` to reuse a model loaded at server startup.
    """
    if loaded_model is not None:
        return inference_image(
            rank=0,
            world_size=1,
            src_image=src_image,
            trgt_image=trgt_image,
            opt=loaded_model.opt,
            model=loaded_model.model,
        )

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    opt = resolve_config(config_path)
    master_port = _find_free_port()

    if world_size == 1:
        return inference_image(
            rank=0,
            world_size=1,
            src_image=src_image,
            trgt_image=trgt_image,
            opt=opt,
            master_port=master_port,
        )

    mp.spawn(
        inference_image,
        args=(world_size, src_image, trgt_image, opt, None, master_port),
        nprocs=world_size,
        join=True,
    )
    return trgt_image


if __name__ == "__main__":
    raise SystemExit(
        "No CLI arguments supported.\n"
        "Import and call `run_inference(src_image, trgt_image, loaded_model=...)` instead."
    )

'''
This script works as an inference video recorder.
'''
import os
import numpy as np
import cv2 as cv
import socket

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, 'options', 'inference_video', 'Baseline.yml')

# PyTorch library
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from torchvision.transforms import Resize

from .data.dataset_reader.datapipeline import *
from .archs import *
from .utils.test_utils import *
from .download_model import LoadedModel, load_model, resolve_config

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port

#define some transforms
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def array_to_tensor(frame):
    '''
    Transform from numpy array [H,W,C] to torch tensor [B,C,H,W]
    '''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() 
    return tensor_frame 

def tensor_to_array(tensor):
    '''
    Transform from torch tensor [B,C,H,W] to numpy array [H,W,C].
    '''
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    frame = (array * 255).astype(np.uint8)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # flip red and blue channels
    return frame

def normalize_tensor(tensor):
    '''
    Normalize tensor to the range [0,1]
    '''
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    '''
    Save tensor as PIL image.
    '''
    tensor = tensor.squeeze(0)
    # tensor = normalize_tensor(tensor)
    print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor))
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''
    Pad the tensor to be multiple of some number (its size).
    '''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    return tensor

def apply_model(model, tensor, resize = False):
    '''
    Apply the inference over each specific frame. If resize = True, resizes before inference.
    '''
    _, _, H, W = tensor.shape
    if resize:
        new_size = [720, 1080]
        downsample = Resize(new_size)
    else:
        downsample = torch.nn.Identity()
    tensor = downsample(tensor)
    tensor = pad_tensor(tensor)

    with torch.no_grad():
        output = model(tensor, side_loss=False)
    if resize:
        upsample = Resize((H, W))
    else: upsample = torch.nn.Identity()

    output = upsample(output)
    output = torch.clamp(output, 0., 1.)
    output = output[:,:, :H, :W]
    return output

def inference_video(rank, world_size, inp_path, out_path, opt, model=None, master_port=None):
    '''
    Inferences the video frames and constructs a new video.
    Pass a pre-loaded `model` to skip loading weights on each call.
    '''
    own_runtime = model is None
    if own_runtime:
        setup(rank, world_size=world_size, Master_port=str(master_port or 12355))
    cap = None
    out = None
    pbar = None
    try:
        resize = opt['Resize']

        if not os.path.isfile(inp_path):
            raise FileNotFoundError(f'Input video not found: {inp_path}')

        # Open the video file
        cap = cv.VideoCapture(inp_path)
        if not cap.isOpened():
            raise RuntimeError(f'Could not open video: {inp_path}')

        # Get video properties
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        # Buffer all enhanced frames, then write the video once at the end.
        # This is more robust for some OpenCV builds, but uses more RAM.
        enhanced_frames: list[np.ndarray] = []

        # Instantiate model and load weights unless a ready model was provided
        if model is None:
            model, _, _ = create_model(opt['network'], rank=rank)
            model = load_model(model, path_weights=opt['save']['path'])

        model.eval()

        if rank == 0:
            pbar = tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tensor = array_to_tensor(frame)
            tensor = normalize_tensor(tensor)

            output = apply_model(model, tensor, resize = resize)

            frame = tensor_to_array(output)
            enhanced_frames.append(frame)
            if rank == 0:
                pbar.update(1)

        # Write the buffered frames to disk.
        out = cv.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise RuntimeError(f'Could not create output video: {out_path}')
        for f in enhanced_frames:
            out.write(f)

        print(f'Finished inference! Saved to {out_path}')
        return out_path
    finally:
        try:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if pbar is not None:
                pbar.close()
        finally:
            if own_runtime:
                cleanup()

def run_low_light_video_inference(
    inp_path,
    out_path,
    loaded_model: LoadedModel | None = None,
    config_path=DEFAULT_CONFIG_PATH,
    cuda_visible_devices="0",
    world_size=1,
):
    """
    Programmatic entrypoint (no CLI args).
    Pass `loaded_model` from `load_ready_model()` to reuse a model loaded at server startup.
    """
    if loaded_model is not None:
        os.makedirs('./videos/results', exist_ok=True)
        return inference_video(
            rank=0,
            world_size=1,
            inp_path=inp_path,
            out_path=out_path,
            opt=loaded_model.opt,
            model=loaded_model.model,
        )

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    opt = resolve_config(config_path)

    master_port = _find_free_port()
    if world_size == 1:
        return inference_video(
            rank=0,
            world_size=1,
            inp_path=inp_path,
            out_path=out_path,
            opt=opt,
            master_port=master_port,
        )

    mp.spawn(
        inference_video,
        args=(world_size, inp_path, out_path, opt, None, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == '__main__':
    raise SystemExit(
        "No CLI arguments supported.\n"
        "Import and call `run_video_inference(inp_path, config_path=...)` instead."
    )

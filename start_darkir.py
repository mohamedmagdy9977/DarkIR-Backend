from DarkIR.download_model import (
    LoadedModel,
    load_ready_model,
    shutdown_ready_model,
)
from DarkIR.inference import run_low_light_img_inference
from DarkIR.inference_video import run_low_light_video_inference


# Load the model
model: LoadedModel = load_ready_model()

# Run the inference
run_low_light_img_inference("Source/input.png", "Target/output.png",model)
run_low_light_video_inference("Source/input.mp4", "Target/output.mp4",model)

# Shutdown the model
shutdown_ready_model()
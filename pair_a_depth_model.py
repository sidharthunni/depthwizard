"""
Pair A — Depth Model Inference
SIH26175 DepthWizard
"""

import argparse
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def load_depth_model(model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()
    return processor, model


def run_depth_estimation(image_path: str, processor, model):
    image = Image.open(image_path).convert("RGB")
    image_size = image.size

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(image_size[1], image_size[0]),
        mode="bicubic",
        align_corners=False,
    )
    depth_array = prediction.squeeze().cpu().numpy().astype(np.float32)
    return depth_array, image_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="depth_output.npy")
    args = parser.parse_args()

    print("Loading depth model...")
    processor, model = load_depth_model()

    print(f"Running inference on {args.input}...")
    depth_array, image_size = run_depth_estimation(args.input, processor, model)

    print(f"Depth array shape: {depth_array.shape}")
    print(f"Original image size (W x H): {image_size}")
    print(f"Depth value range: {depth_array.min():.3f} to {depth_array.max():.3f}")

    np.save(args.output, depth_array)
    print(f"Saved depth output to {args.output}")

    import json
    meta_path = args.output.replace(".npy", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"width": image_size[0], "height": image_size[1]}, f)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

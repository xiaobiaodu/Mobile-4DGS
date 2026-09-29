#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from lpipsPyTorch import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


def read_image(path, device):
    """Load one RGB image and transfer only that image to the target device."""
    with Image.open(path) as image:
        tensor = tf.to_tensor(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device)


def image_names_for_method(renders_dir, gt_dir):
    render_names = {
        path.name for path in renders_dir.iterdir() if path.is_file()
    }
    gt_names = {
        path.name for path in gt_dir.iterdir() if path.is_file()
    }
    if render_names != gt_names:
        missing_gt = sorted(render_names - gt_names)
        missing_render = sorted(gt_names - render_names)
        raise ValueError(
            "Render/GT image sets differ: "
            f"missing_gt={missing_gt[:5]}, missing_render={missing_render[:5]}"
        )
    if not render_names:
        raise ValueError(f"No metric images found in {renders_dir}")
    return sorted(render_names)


def load_json_dict(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}


def evaluate(model_paths, methods=None, device=None):
    device = torch.device("cuda:0") if device is None else torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Construct VGG once. The previous implementation rebuilt and transferred
    # the network for every frame, causing allocator churn and excessive work.
    lpips_metric = LPIPS("vgg", "0.1").to(device).eval()
    requested_methods = None if methods is None else set(methods)
    print("")

    for scene_dir_value in model_paths:
        scene_dir = Path(scene_dir_value)
        try:
            print("Scene:", scene_dir)
            results_path = scene_dir / "results.json"
            per_view_path = scene_dir / "per_view.json"
            scene_results = load_json_dict(results_path)
            scene_per_view = load_json_dict(per_view_path)
            test_dir = scene_dir / "test"
            available_methods = sorted(
                path.name for path in test_dir.iterdir() if path.is_dir()
            )
            if requested_methods is None:
                selected_methods = available_methods
            else:
                missing_methods = sorted(requested_methods - set(available_methods))
                if missing_methods:
                    raise FileNotFoundError(
                        f"Requested metric methods do not exist: {missing_methods}; "
                        f"available={available_methods}"
                    )
                selected_methods = [
                    method for method in available_methods
                    if method in requested_methods
                ]

            for method in selected_methods:
                print("Method:", method)
                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                image_names = image_names_for_method(renders_dir, gt_dir)
                ssims = []
                psnrs = []
                lpipss = []

                with torch.inference_mode():
                    for image_name in tqdm(
                        image_names,
                        desc="Metric evaluation progress",
                    ):
                        render = read_image(renders_dir / image_name, device)
                        gt = read_image(gt_dir / image_name, device)
                        ssims.append(float(ssim(render, gt).item()))
                        psnrs.append(float(psnr(render, gt).item()))
                        lpipss.append(float(lpips_metric(render, gt).item()))
                        del render, gt

                mean_ssim = sum(ssims) / len(ssims)
                mean_psnr = sum(psnrs) / len(psnrs)
                mean_lpips = sum(lpipss) / len(lpipss)
                print(f"  SSIM : {mean_ssim:>12.7f}")
                print(f"  PSNR : {mean_psnr:>12.7f}")
                print(f"  LPIPS: {mean_lpips:>12.7f}\n")

                scene_results[method] = {
                    "SSIM": mean_ssim,
                    "PSNR": mean_psnr,
                    "LPIPS": mean_lpips,
                }
                scene_per_view[method] = {
                    "SSIM": dict(zip(image_names, ssims)),
                    "PSNR": dict(zip(image_names, psnrs)),
                    "LPIPS": dict(zip(image_names, lpipss)),
                }

            with results_path.open("w", encoding="utf-8") as file:
                json.dump(scene_results, file, indent=True)
            with per_view_path.open("w", encoding="utf-8") as file:
                json.dump(scene_per_view, file, indent=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as error:
            print(
                f"Unable to compute metrics for model {scene_dir}: "
                f"{type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate rendered model images")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Only evaluate these test subdirectories (for example ours_30000)",
    )
    args = parser.parse_args()
    evaluate(args.model_paths, methods=args.methods)

import argparse
import os

import numpy as np
import torch

from nnInteractive.inference.inference_session_undo_redo import nnInteractiveInferenceSessionUndoRedo


def make_gaussian_image(size: int = 256, sigma: float = 0.25) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    gaussian = np.exp(-((xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))).astype(np.float32)
    return gaussian[None]


def make_outward_points(center: int = 128, step: int = 12, n_steps: int = 6):
    points = [(center, center, center)]
    for i in range(1, n_steps + 1):
        d = i * step
        points.extend(
            [
                (center + d, center, center),
                (center - d, center, center),
                (center, center + d, center),
                (center, center - d, center),
                (center, center, center + d),
                (center, center, center - d),
            ]
        )
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get("NNINTERACTIVE_MODEL_PATH", ""),
        help="Path to trained nnInteractive model folder (contains plans.json, dataset.json, fold_*/checkpoint_*.pth).",
    )
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("Please provide --model-path or set NNINTERACTIVE_MODEL_PATH.")

    device = torch.device(args.device)
    session = nnInteractiveInferenceSessionUndoRedo(device=device, use_torch_compile=False, verbose=False)
    session.initialize_from_trained_model_folder(args.model_path)

    image = make_gaussian_image(size=256, sigma=0.25)
    session.set_image(image)

    target = torch.zeros(image.shape[1:], dtype=torch.uint8, device=device)
    session.set_target_buffer(target)

    points = make_outward_points(center=128, step=12, n_steps=5)

    prediction_states = []
    for p in points:
        session.add_point_interaction(p, include_interaction=True, run_prediction=True)
        prediction_states.append(target.detach().clone())
        print(f"Added point {p}, foreground voxels: {int(target.sum().item())}")

    undo_steps = 0
    while session.undo_prediction():
        undo_steps += 1
        expected = prediction_states[-(undo_steps + 1)]
        if not torch.equal(target, expected):
            raise RuntimeError(f"Undo mismatch at step {undo_steps}.")
        print(f"Undo {undo_steps} OK, foreground voxels: {int(target.sum().item())}")

    redo_steps = 0
    while session.redo_prediction():
        expected = prediction_states[redo_steps + 1]
        if not torch.equal(target, expected):
            raise RuntimeError(f"Redo mismatch at step {redo_steps + 1}.")
        redo_steps += 1
        print(f"Redo {redo_steps} OK, foreground voxels: {int(target.sum().item())}")

    print("Undo/redo test completed successfully.")


if __name__ == "__main__":
    main()

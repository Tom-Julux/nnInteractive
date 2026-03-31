from os.path import join
from typing import List, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import save_json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn

from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d


class _AutoPromptInputWrapper(nn.Module):
    """
    Wraps a nnU-Net network and auto-appends empty prompt channels when they are missing.
    """

    def __init__(
        self, network: nn.Module, image_channels: int, prompt_channels: int = 7
    ) -> None:
        super().__init__()
        self.network = network
        self.image_channels = image_channels
        self.prompt_channels = prompt_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.image_channels:
            prompt = torch.zeros(
                (x.shape[0], self.prompt_channels, *x.shape[2:]),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((x, prompt), dim=1)
        elif x.shape[1] != self.image_channels + self.prompt_channels:
            raise RuntimeError(
                f"Unexpected input channels {x.shape[1]}. "
                f"Expected {self.image_channels} or "
                f"{self.image_channels + self.prompt_channels}."
            )
        return self.network(x)

    def state_dict(self, *args, **kwargs):
        # Keep checkpoint weights compatible with nnInteractiveTrainer_stub-based inference checkpoints.
        return self.network.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.network.load_state_dict(state_dict, strict=strict, assign=assign)


class nnInteractiveTrainer(nnUNetTrainer):
    prompt_channels = 7
    point_channel_positive = 3
    point_channel_negative = 4
    scribble_channel_positive = 5
    scribble_channel_negative = 6

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + nnInteractiveTrainer.prompt_channels,
            2,  # nnUNet handles one-class segmentation via CE with 2 logits.
            enable_deep_supervision,
        )
        return _AutoPromptInputWrapper(
            base_network,
            image_channels=num_input_channels,
            prompt_channels=nnInteractiveTrainer.prompt_channels,
        )

    def on_train_start(self):
        super().on_train_start()
        self._write_inference_session_settings()

    def _write_inference_session_settings(self):
        settings = self.dataset_json.get("nninteractive_inference_settings", {})
        preferred_scribble_thickness = settings.get(
            "preferred_scribble_thickness", [2, 2, 2]
        )
        if not isinstance(preferred_scribble_thickness, (list, tuple)):
            preferred_scribble_thickness = [preferred_scribble_thickness] * 3

        inference_settings = {
            "point_radius": int(settings.get("point_radius", 4)),
            "preferred_scribble_thickness": [
                int(i) for i in preferred_scribble_thickness
            ],
            "interaction_decay": float(settings.get("interaction_decay", 0.98)),
            "pad_mode_image": str(settings.get("pad_mode_image", "constant")),
        }
        save_json(
            inference_settings,
            join(self.output_folder_base, "inference_session_class.json"),
            sort_keys=False,
        )

    def _get_highres_target(self, target):
        target_tensor = target[0] if isinstance(target, list) else target
        if (
            not self.label_manager.has_regions
            and target_tensor.ndim >= 3
            and target_tensor.shape[1] == 1
        ):
            target_tensor = target_tensor[:, 0]
        return target_tensor

    def _get_foreground_and_ignore_masks(self, target):
        target_tensor = self._get_highres_target(target)
        if self.label_manager.has_regions:
            if self.label_manager.has_ignore_label:
                foreground_mask = target_tensor[:, :-1].to(torch.bool).any(1)
                ignore_mask = target_tensor[:, -1:].to(torch.bool).squeeze(1)
            else:
                foreground_mask = target_tensor.to(torch.bool).any(1)
                ignore_mask = torch.zeros_like(foreground_mask, dtype=torch.bool)
        else:
            ignore_mask = (
                target_tensor == self.label_manager.ignore_label
                if self.label_manager.has_ignore_label
                else torch.zeros_like(target_tensor, dtype=torch.bool)
            )
            foreground_mask = (target_tensor > 0) & (~ignore_mask)
        return foreground_mask, ignore_mask

    @staticmethod
    def _sample_indices(candidate_idx: torch.Tensor, max_points: int) -> torch.Tensor:
        if candidate_idx.numel() == 0 or max_points < 1:
            return candidate_idx[:0]
        n = min(max_points, len(candidate_idx))
        selected = torch.randperm(len(candidate_idx), device=candidate_idx.device)[:n]
        return candidate_idx[selected]

    @staticmethod
    def _draw_binary_points(
        spatial_shape: Tuple[int, ...],
        selected_indices: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        out = torch.zeros(spatial_shape, dtype=dtype, device=device)
        if selected_indices.numel() > 0:
            out[tuple(selected_indices.T)] = 1
        return out

    def _dilate_sparse_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1:
            return mask
        if mask.ndim == 3:
            return iterative_3x3_same_padding_pool3d(mask[None, None], kernel_size)[
                0, 0
            ]
        return mask

    def _build_sparse_prompts(self, data: torch.Tensor, target) -> torch.Tensor:
        prompt = torch.zeros(
            (data.shape[0], self.prompt_channels, *data.shape[2:]),
            dtype=data.dtype,
            device=data.device,
        )
        foreground_mask, ignore_mask = self._get_foreground_and_ignore_masks(target)
        scribble_cfg = self.dataset_json.get("nninteractive_training_settings", {})
        configured_scribble_kernel_size = int(
            scribble_cfg.get("scribble_kernel_size", 5)
        )
        if configured_scribble_kernel_size < 0:
            raise ValueError(
                "nninteractive_training_settings.scribble_kernel_size must be >= 0"
            )
        scribble_kernel_size = max(1, configured_scribble_kernel_size)
        if scribble_kernel_size % 2 == 0:
            scribble_kernel_size += 1
        max_scribble_points = int(scribble_cfg.get("max_scribble_points", 32))

        for batch_idx in range(data.shape[0]):
            fg_idx = torch.argwhere(foreground_mask[batch_idx])
            if fg_idx.numel() > 0:
                rand_fg_idx = torch.randint(
                    0, len(fg_idx), (1,), device=fg_idx.device
                ).item()
                pick_fg = fg_idx[rand_fg_idx]
                prompt[(batch_idx, self.point_channel_positive, *pick_fg.tolist())] = 1

            bg_mask = (~foreground_mask[batch_idx]) & (~ignore_mask[batch_idx])
            bg_idx = torch.argwhere(bg_mask)
            if bg_idx.numel() > 0:
                rand_bg_idx = torch.randint(
                    0, len(bg_idx), (1,), device=bg_idx.device
                ).item()
                pick_bg = bg_idx[rand_bg_idx]
                prompt[(batch_idx, self.point_channel_negative, *pick_bg.tolist())] = 1

            fg_scribble_idx = self._sample_indices(fg_idx, max_scribble_points)
            if fg_scribble_idx.numel() > 0:
                fg_scribble = self._draw_binary_points(
                    tuple(foreground_mask[batch_idx].shape),
                    fg_scribble_idx,
                    prompt.dtype,
                    prompt.device,
                )
                fg_scribble = self._dilate_sparse_mask(
                    fg_scribble, kernel_size=scribble_kernel_size
                )
                prompt[batch_idx, self.scribble_channel_positive] = fg_scribble

            bg_scribble_idx = self._sample_indices(bg_idx, max_scribble_points)
            if bg_scribble_idx.numel() > 0:
                bg_scribble = self._draw_binary_points(
                    tuple(foreground_mask[batch_idx].shape),
                    bg_scribble_idx,
                    prompt.dtype,
                    prompt.device,
                )
                bg_scribble = self._dilate_sparse_mask(
                    bg_scribble, kernel_size=scribble_kernel_size
                )
                prompt[batch_idx, self.scribble_channel_negative] = bg_scribble
        return prompt

    def _prepare_training_data(
        self, data: torch.Tensor, target, add_sparse_prompts: bool
    ) -> torch.Tensor:
        expected_channels = self.num_input_channels
        prompt_expected_channels = expected_channels + self.prompt_channels
        if data.shape[1] == prompt_expected_channels:
            return data
        if data.shape[1] != expected_channels:
            raise RuntimeError(
                f"Unexpected data channels {data.shape[1]}. "
                f"Expected {expected_channels} or {prompt_expected_channels}."
            )
        prompt = (
            self._build_sparse_prompts(data, target)
            if add_sparse_prompts
            else torch.zeros(
                (data.shape[0], self.prompt_channels, *data.shape[2:]),
                dtype=data.dtype,
                device=data.device,
            )
        )
        return torch.cat((data, prompt), dim=1)

    def train_step(self, batch: dict) -> dict:
        batch = dict(batch)
        batch["data"] = self._prepare_training_data(
            batch["data"], batch["target"], add_sparse_prompts=True
        )
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        batch = dict(batch)
        batch["data"] = self._prepare_training_data(
            batch["data"], batch["target"], add_sparse_prompts=False
        )
        return super().validation_step(batch)


class nnInteractiveTrainer_stub:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + 7,
            2,  # nnUNet handles one-class segmentation via CE with 2 logits.
            enable_deep_supervision,
        )

"""IC-LoRA pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch
    from ltx_core.loader import LoraPathStrengthAndSDOps


class IcLoraPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        *,
        extra_loras: list["LoraPathStrengthAndSDOps"] | None = None,
    ) -> "IcLoraPipeline":
        ...

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        output_path: str,
    ) -> None:
        ...

    def compile_transformer(self) -> None: ...

"""Utilities for validating and materializing local LTX LoRA stacks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from api_types import LoraInput

LoraSignature = tuple[tuple[str, float, str], ...]

if TYPE_CHECKING:
    from ltx_core.loader import LoraPathStrengthAndSDOps


def build_lora_signature(loras: list[LoraInput]) -> LoraSignature:
    return tuple(
        (str(Path(lora.path).expanduser()), round(float(lora.strength), 6), lora.sd_ops_preset)
        for lora in loras
    )


def resolve_lora_entries(loras: list[LoraInput]) -> list["LoraPathStrengthAndSDOps"]:
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

    entries: list[LoraPathStrengthAndSDOps] = []
    for lora in loras:
        lora_path = Path(lora.path).expanduser()
        if not lora_path.exists():
            raise ValueError(f"LoRA file not found: {lora_path}")
        if not lora_path.is_file():
            raise ValueError(f"LoRA path is not a file: {lora_path}")
        if lora_path.suffix.lower() != ".safetensors":
            raise ValueError(f"LoRA must be a .safetensors file: {lora_path}")

        if lora.sd_ops_preset != "ltx_comfy":
            raise ValueError(f"Unsupported LoRA sd_ops_preset: {lora.sd_ops_preset}")
        sd_ops = LTXV_LORA_COMFY_RENAMING_MAP

        entries.append(
            LoraPathStrengthAndSDOps(
                path=str(lora_path),
                strength=float(lora.strength),
                sd_ops=sd_ops,
            )
        )

    return entries

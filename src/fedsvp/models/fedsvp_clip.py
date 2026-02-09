from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PromptStageConfig:
    shallow_start: int = 0
    shallow_end: int = 3
    mid_start: int = 4
    mid_end: int = 8
    deep_start: int = 9
    deep_end: int = 11


class MultiScaleVisualPrompt(nn.Module):
    """Fed-MSVP prompt parameters with adaptive per-layer gates.

    Modes
    -----
    - ``msvp``: three stage prompts (shallow/mid/deep).
    - ``input``: only one input-level prompt for ablations.
    """

    def __init__(
        self,
        width: int,
        num_layers: int,
        mode: str = "msvp",
        stage_cfg: PromptStageConfig | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.num_layers = int(num_layers)
        self.mode = str(mode).lower()
        self.stage_cfg = stage_cfg or PromptStageConfig()

        # Adaptive gating. Alpha starts from 0, so training starts from frozen CLIP behavior.
        self.alpha = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))

        if self.mode == "msvp":
            self.shallow_layers = list(range(self.stage_cfg.shallow_start, self.stage_cfg.shallow_end + 1))
            self.mid_layers = list(range(self.stage_cfg.mid_start, self.stage_cfg.mid_end + 1))
            self.deep_layers = list(range(self.stage_cfg.deep_start, self.stage_cfg.deep_end + 1))

            self.p_shallow = nn.Parameter(torch.randn(len(self.shallow_layers), 1, self.width) * init_std)
            self.p_mid = nn.Parameter(torch.randn(len(self.mid_layers), 1, self.width) * init_std)
            self.p_deep = nn.Parameter(torch.randn(len(self.deep_layers), 1, self.width) * init_std)

            self._layer_to_stage = {}
            for i, layer in enumerate(self.shallow_layers):
                self._layer_to_stage[layer] = ("shallow", i)
            for i, layer in enumerate(self.mid_layers):
                self._layer_to_stage[layer] = ("mid", i)
            for i, layer in enumerate(self.deep_layers):
                self._layer_to_stage[layer] = ("deep", i)
        elif self.mode in ("input", "standard", "single"):
            self.p_input = nn.Parameter(torch.randn(1, 1, self.width) * init_std)
        else:
            raise ValueError(f"Unsupported prompt mode: {self.mode}")

    def layer_prompt(self, layer_idx: int) -> torch.Tensor | None:
        layer_idx = int(layer_idx)
        if self.mode == "msvp":
            if layer_idx not in self._layer_to_stage:
                return None
            stage, idx = self._layer_to_stage[layer_idx]
            if stage == "shallow":
                return self.p_shallow[idx : idx + 1]
            if stage == "mid":
                return self.p_mid[idx : idx + 1]
            return self.p_deep[idx : idx + 1]

        # input-level baseline: only inject at layer 0
        if layer_idx == 0:
            return self.p_input
        return None

    def prompt_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = [self.alpha]
        if self.mode == "msvp":
            params.extend([self.p_shallow, self.p_mid, self.p_deep])
        else:
            params.append(self.p_input)
        return params


class PromptedOpenCLIPVision(nn.Module):
    """OpenCLIP ViT vision encoder with Fed-MSVP prompt injection."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        prompt_mode: str = "msvp",
    ) -> None:
        super().__init__()

        try:
            import open_clip
        except Exception as e:  # pragma: no cover
            raise RuntimeError("open_clip_torch is required. Install via `pip install open_clip_torch`.") from e

        self._open_clip = open_clip
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)

        pre = self.pretrained.strip().lower()
        if pre in ("", "none", "null", "false"):
            self.clip = open_clip.create_model(self.model_name, pretrained="")
        else:
            self.clip, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)

        self.visual = self.clip.visual
        if not hasattr(self.visual, "transformer") or not hasattr(self.visual.transformer, "resblocks"):
            raise RuntimeError("FedSVP currently supports ViT-style OpenCLIP visual encoders only.")

        self.num_layers = len(self.visual.transformer.resblocks)

        if hasattr(self.visual, "conv1") and hasattr(self.visual.conv1, "weight"):
            width = int(self.visual.conv1.weight.shape[0])
        else:
            width = int(getattr(self.visual, "width", 768))
        self.width = width

        self.prompt = MultiScaleVisualPrompt(width=self.width, num_layers=self.num_layers, mode=prompt_mode)
        self.tokenizer = self._build_tokenizer(open_clip, self.model_name)

    @staticmethod
    def _build_tokenizer(open_clip_module, model_name: str):
        """Build a text tokenizer across different open_clip API versions."""
        if hasattr(open_clip_module, "get_tokenizer"):
            return open_clip_module.get_tokenizer(model_name)

        if hasattr(open_clip_module, "tokenize"):
            tok = open_clip_module.tokenize

            def _tok(texts):
                return tok(list(texts))

            return _tok

        tok_mod = getattr(open_clip_module, "tokenizer", None)
        if tok_mod is not None and hasattr(tok_mod, "tokenize"):
            tok = tok_mod.tokenize

            def _tok(texts):
                return tok(list(texts))

            return _tok

        mod_file = str(getattr(open_clip_module, "__file__", "unknown"))
        raise RuntimeError(
            "Cannot find a tokenizer API in open_clip module. "
            f"Expected `get_tokenizer` or `tokenize`, got module at: {mod_file}"
        )

    def freeze_backbone(self) -> None:
        for p in self.clip.parameters():
            p.requires_grad_(False)
        for p in self.prompt.parameters():
            p.requires_grad_(True)

    def encode_text(self, texts: Sequence[str], device: torch.device, normalize: bool = True) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(device)
        with torch.no_grad():
            txt = self.clip.encode_text(tokens)
            txt = txt.float()
            if normalize:
                txt = F.normalize(txt, dim=-1)
        return txt

    def get_logit_scale(self) -> float:
        if hasattr(self.clip, "logit_scale"):
            with torch.no_grad():
                return float(self.clip.logit_scale.exp().item())
        return 100.0

    def encode_image(self, x: torch.Tensor, with_prompt: bool = True, normalize: bool = True) -> torch.Tensor:
        feat = self._forward_visual(x, with_prompt=with_prompt)
        feat = feat.float()
        if normalize:
            feat = F.normalize(feat, dim=-1)
        return feat

    def _forward_visual(self, x: torch.Tensor, with_prompt: bool = True) -> torch.Tensor:
        v = self.visual

        # patchify
        x = v.conv1(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, x.shape[1], -1).permute(0, 2, 1)

        # prepend cls token
        cls = v.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls + torch.zeros(batch_size, 1, cls.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)

        # positional embeddings and pre-LN
        if hasattr(v, "positional_embedding"):
            x = x + v.positional_embedding.to(dtype=x.dtype, device=x.device)
        if hasattr(v, "patch_dropout"):
            x = v.patch_dropout(x)
        if hasattr(v, "ln_pre"):
            x = v.ln_pre(x)

        # transformer blocks expect LND
        x = x.permute(1, 0, 2)
        for layer_idx, block in enumerate(v.transformer.resblocks):
            x = block(x)
            if with_prompt:
                p = self.prompt.layer_prompt(layer_idx)
                if p is not None:
                    alpha = self.prompt.alpha[layer_idx].to(dtype=x.dtype)
                    x = x + alpha.view(1, 1, 1) * p.to(dtype=x.dtype)

        x = x.permute(1, 0, 2)

        if hasattr(v, "ln_post"):
            x = v.ln_post(x)

        pool_type = getattr(v, "pool_type", "tok")
        if pool_type == "avg":
            feat = x[:, 1:, :].mean(dim=1)
        else:
            feat = x[:, 0, :]

        if getattr(v, "proj", None) is not None:
            feat = feat @ v.proj
        return feat

    def prompt_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.prompt.state_dict().items()}

    def load_prompt_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.prompt.load_state_dict(state, strict=True)

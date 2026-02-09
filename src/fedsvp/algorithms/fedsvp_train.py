from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedsvp.utils.metrics import AverageMeter, accuracy_top1


def _progress_iter(loader: DataLoader, progress: Optional[Dict[str, Any]]):
    if not progress or not progress.get("enabled", False):
        return None, loader
    try:
        from tqdm import tqdm
    except Exception:
        return None, loader

    pbar = tqdm(
        loader,
        desc=progress.get("desc"),
        leave=bool(progress.get("leave", False)),
        dynamic_ncols=bool(progress.get("dynamic_ncols", True)),
    )
    return pbar, pbar


def clip_classification_logits(
    image_features: torch.Tensor,
    class_text_features: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    return float(logit_scale) * image_features @ class_text_features.t()


def train_fedsvp_epoch(
    model,
    loader: DataLoader,
    class_text_features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_consistency: float,
    logit_scale: float,
    progress: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    model.prompt.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar, loader_iter = _progress_iter(loader, progress)
    for x, y in loader_iter:
        x = x.to(device)
        y = y.to(device)

        feat_prompt = model.encode_image(x, with_prompt=True, normalize=True)
        logits = clip_classification_logits(feat_prompt, class_text_features, logit_scale=logit_scale)
        ce = F.cross_entropy(logits, y)

        with torch.no_grad():
            feat_frozen = model.encode_image(x, with_prompt=False, normalize=True)

        consistency = 1.0 - F.cosine_similarity(feat_prompt, feat_frozen, dim=-1).mean()
        loss = ce + float(lambda_consistency) * consistency

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(float(loss.item()), n=x.size(0))
        acc_meter.update(float(accuracy_top1(logits.detach(), y)), n=x.size(0))

    if pbar is not None:
        pbar.close()

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def eval_fedsvp(
    model,
    loader: DataLoader,
    class_text_features: torch.Tensor,
    device: torch.device,
    logit_scale: float,
) -> Tuple[float, float]:
    model.prompt.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        feat_prompt = model.encode_image(x, with_prompt=True, normalize=True)
        logits = clip_classification_logits(feat_prompt, class_text_features, logit_scale=logit_scale)
        loss = F.cross_entropy(logits, y)

        loss_meter.update(float(loss.item()), n=x.size(0))
        acc_meter.update(float(accuracy_top1(logits, y)), n=x.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def semantic_score_on_proxy(
    model,
    prompt_state: Dict[str, torch.Tensor],
    proxy_loader: DataLoader,
    anchor_text_features: torch.Tensor,
    device: torch.device,
    max_batches: int = -1,
) -> float:
    model.load_prompt_state_dict(prompt_state)
    model.prompt.eval()

    score_meter = AverageMeter()
    for batch_idx, (x, _) in enumerate(proxy_loader):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        x = x.to(device)
        v = model.encode_image(x, with_prompt=True, normalize=True)
        sim = v @ anchor_text_features.t()
        score = sim.max(dim=1).values.mean()
        score_meter.update(float(score.item()), n=x.size(0))

    return float(score_meter.avg)

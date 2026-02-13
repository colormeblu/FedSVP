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


def _model_core(model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _model_encode_image(
    model,
    x: torch.Tensor,
    with_prompt: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        # Use keyword `x` to make wrapper scatter kwargs robustly.
        return model(
            x=x,
            with_prompt=with_prompt,
            normalize=normalize,
        )
    return model.encode_image(x, with_prompt=with_prompt, normalize=normalize)


def clip_classification_logits(
    image_features: torch.Tensor,
    class_text_features: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    return float(logit_scale) * image_features @ class_text_features.t()


def global_local_contrastive_loss(
    image_features: torch.Tensor,
    labels: torch.Tensor,
    global_prototypes: torch.Tensor,
    temperature: float,
    prototype_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE-style global-local contrastive loss over class prototypes."""
    if image_features.ndim != 2 or global_prototypes.ndim != 2:
        raise ValueError("image_features and global_prototypes must both be 2D tensors")

    num_classes = int(global_prototypes.size(0))
    if num_classes <= 0:
        return torch.tensor(0.0, device=image_features.device)

    if prototype_mask is None:
        mask = torch.ones(num_classes, dtype=torch.bool, device=image_features.device)
    else:
        mask = prototype_mask.to(device=image_features.device, dtype=torch.bool)
        if mask.ndim != 1 or int(mask.numel()) != num_classes:
            raise ValueError("prototype_mask must be 1D with length == num_classes")

    if not bool(mask.any()):
        return torch.tensor(0.0, device=image_features.device)

    tau = max(float(temperature), 1e-6)
    logits = (image_features @ global_prototypes.t()) / tau
    logits = logits.masked_fill(~mask.view(1, -1), -1e4)

    # Ignore out-of-range labels or labels whose global prototype is unavailable.
    targets = labels.clone().to(dtype=torch.long)
    valid = (targets >= 0) & (targets < num_classes)
    if bool(valid.any()):
        safe_targets = targets.clamp(min=0, max=num_classes - 1)
        has_proto = mask[safe_targets]
        valid = valid & has_proto
    targets[~valid] = -100

    if not bool(valid.any()):
        return torch.tensor(0.0, device=image_features.device)
    return F.cross_entropy(logits, targets, ignore_index=-100)


def train_fedsvp_epoch(
    model,
    loader: DataLoader,
    class_text_features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_consistency: float,
    logit_scale: float,
    global_prototypes: Optional[torch.Tensor] = None,
    global_proto_mask: Optional[torch.Tensor] = None,
    lambda_contrastive: float = 0.0,
    contrastive_tau: float = 0.07,
    progress: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    core = _model_core(model)
    core.prompt.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    proto = None
    proto_mask = None
    if global_prototypes is not None and float(lambda_contrastive) > 0:
        proto = global_prototypes.to(device=device, dtype=torch.float32)
        proto = F.normalize(proto, dim=-1)
        if global_proto_mask is None:
            proto_mask = torch.ones(proto.size(0), dtype=torch.bool, device=device)
        else:
            proto_mask = global_proto_mask.to(device=device, dtype=torch.bool)

    pbar, loader_iter = _progress_iter(loader, progress)
    for x, y in loader_iter:
        x = x.to(device)
        y = y.to(device)

        feat_prompt = _model_encode_image(model, x, with_prompt=True, normalize=True)
        logits = clip_classification_logits(feat_prompt, class_text_features, logit_scale=logit_scale)
        ce = F.cross_entropy(logits, y)

        with torch.no_grad():
            feat_frozen = _model_encode_image(model, x, with_prompt=False, normalize=True)

        consistency = 1.0 - F.cosine_similarity(feat_prompt, feat_frozen, dim=-1).mean()
        loss = ce + float(lambda_consistency) * consistency

        if proto is not None:
            con = global_local_contrastive_loss(
                image_features=feat_prompt,
                labels=y,
                global_prototypes=proto,
                temperature=contrastive_tau,
                prototype_mask=proto_mask,
            )
            loss = loss + float(lambda_contrastive) * con

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(float(loss.item()), n=x.size(0))
        acc_meter.update(float(accuracy_top1(logits.detach(), y)), n=x.size(0))

    if pbar is not None:
        pbar.close()

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def collect_local_class_prototypes(
    model,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    feature_dim: Optional[int] = None,
    max_batches: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class local prototypes and class counts on one client."""
    core = _model_core(model)
    core.prompt.eval()

    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    feat_sum = None
    if feature_dim is not None:
        feat_sum = torch.zeros(num_classes, int(feature_dim), dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    for batch_idx, (x, y) in enumerate(loader):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break

        x = x.to(device)
        y = y.to(device)
        feat = _model_encode_image(model, x, with_prompt=True, normalize=True).float()

        if feat_sum is None:
            feat_sum = torch.zeros(num_classes, feat.size(-1), dtype=torch.float32, device=device)

        for cls in torch.unique(y):
            cls_idx = int(cls.item())
            if cls_idx < 0 or cls_idx >= num_classes:
                continue
            m = (y == cls_idx)
            if bool(m.any()):
                feat_sum[cls_idx] += feat[m].sum(dim=0)
                counts[cls_idx] += int(m.sum().item())

    if feat_sum is None:
        if feature_dim is None:
            raise RuntimeError("Cannot infer feature dimension from an empty loader")
        feat_sum = torch.zeros(num_classes, int(feature_dim), dtype=torch.float32, device=device)

    protos = torch.zeros_like(feat_sum)
    valid = counts > 0
    if bool(valid.any()):
        protos[valid] = feat_sum[valid] / counts[valid].unsqueeze(1).to(dtype=feat_sum.dtype)
        protos[valid] = F.normalize(protos[valid], dim=-1)

    return protos.cpu(), counts.cpu()


@torch.no_grad()
def eval_fedsvp(
    model,
    loader: DataLoader,
    class_text_features: torch.Tensor,
    device: torch.device,
    logit_scale: float,
) -> Tuple[float, float]:
    core = _model_core(model)
    core.prompt.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        feat_prompt = _model_encode_image(model, x, with_prompt=True, normalize=True)
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
    core = _model_core(model)
    core.load_prompt_state_dict(prompt_state)
    core.prompt.eval()

    score_meter = AverageMeter()
    for batch_idx, (x, _) in enumerate(proxy_loader):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        x = x.to(device)
        v = _model_encode_image(model, x, with_prompt=True, normalize=True)
        sim = v @ anchor_text_features.t()
        score = sim.max(dim=1).values.mean()
        score_meter.update(float(score.item()), n=x.size(0))

    return float(score_meter.avg)

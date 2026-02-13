from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW, SGD

from fedsvp.registry import register_algorithm
from fedsvp.data.base import make_loader
from fedsvp.data.public import build_public_loader_from_clients, build_public_loader_from_dir
from fedsvp.models.clip_backbone import build_classifier_from_cfg
from fedsvp.models.fedsvp_clip import PromptedOpenCLIPVision
from fedsvp.algorithms.agg import (
    aggregate_class_prototypes,
    fedavg_state_dict,
    semantic_softmax_weights,
    weighted_average_state_dict,
)
from fedsvp.algorithms.local_train import train_classifier_epoch, eval_classifier
from fedsvp.algorithms.fedsvp_train import (
    collect_local_class_prototypes,
    eval_fedsvp,
    semantic_score_on_proxy,
    train_fedsvp_epoch,
)
from fedsvp.utils.io import save_json


def _parse_gpu_ids(cfg: Dict[str, Any]) -> List[int]:
    raw = cfg.get("gpu_ids", None)
    if raw is None:
        return []

    vals: List[int]
    if isinstance(raw, int):
        vals = [int(raw)]
    elif isinstance(raw, str):
        s = raw.strip()
        if len(s) == 0:
            return []
        if s.startswith("["):
            parsed = json.loads(s)
            if not isinstance(parsed, list):
                raise ValueError(f"gpu_ids must be a list/int/string, got {type(parsed).__name__}")
            vals = [int(x) for x in parsed]
        else:
            vals = [int(x.strip()) for x in s.split(",") if len(x.strip()) > 0]
    elif isinstance(raw, (list, tuple)):
        vals = [int(x) for x in raw]
    else:
        raise ValueError(f"gpu_ids must be a list/int/string, got {type(raw).__name__}")

    out: List[int] = []
    seen = set()
    for gid in vals:
        if gid < 0:
            raise ValueError(f"gpu_ids contains invalid id: {gid}")
        if gid not in seen:
            out.append(gid)
            seen.add(gid)
    return out


def _ddp_ctx(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = cfg.get("_runtime") if isinstance(cfg, dict) else None
    if not isinstance(runtime, dict):
        return {}
    ddp = runtime.get("ddp")
    if not isinstance(ddp, dict):
        return {}
    return ddp


def _ddp_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_ddp_ctx(cfg).get("enabled", False))


def _ddp_rank(cfg: Dict[str, Any]) -> int:
    return int(_ddp_ctx(cfg).get("rank", 0))


def _ddp_world_size(cfg: Dict[str, Any]) -> int:
    return int(_ddp_ctx(cfg).get("world_size", 1))


def _is_main_process(cfg: Dict[str, Any]) -> bool:
    return _ddp_rank(cfg) == 0


def _resolve_multi_gpu_ids(cfg: Dict[str, Any]) -> List[int]:
    if not torch.cuda.is_available():
        return []
    gpu_ids = _parse_gpu_ids(cfg)
    want_multi_gpu = bool(cfg.get("multi_gpu", False))
    if len(gpu_ids) == 0 and want_multi_gpu:
        gpu_ids = list(range(int(torch.cuda.device_count())))
    if len(gpu_ids) <= 1:
        return []
    n_gpu = int(torch.cuda.device_count())
    bad = [gid for gid in gpu_ids if gid >= n_gpu]
    if len(bad) > 0:
        raise ValueError(f"gpu_ids contains unavailable CUDA ids {bad}; detected cuda device count={n_gpu}.")
    return gpu_ids


def _wrap_ddp(model: torch.nn.Module, cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    if not _ddp_enabled(cfg):
        return model
    if device.type != "cuda" or device.index is None:
        raise RuntimeError("DDP requires a CUDA device with explicit index.")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = model.to(device)
    return DDP(
        model,
        device_ids=[int(device.index)],
        output_device=int(device.index),
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def _set_loader_epoch(loader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))


def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    if _ddp_enabled(cfg):
        ddp = _ddp_ctx(cfg)
        did = ddp.get("device_id", None)
        if did is None:
            raise RuntimeError("DDP is enabled but runtime device_id is missing.")
        return torch.device(f"cuda:{int(did)}")

    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev_raw = cfg.get("device", default_dev)
    dev = torch.device(dev_raw)
    if dev.type == "cuda" and dev.index is None:
        gpu_ids = _parse_gpu_ids(cfg)
        if len(gpu_ids) > 0:
            dev = torch.device(f"cuda:{gpu_ids[0]}")
    return dev


def _build_clients(
    cfg: Dict[str, Any],
    domains: Dict[str, Any],
    target_domain: str,
    batch_size: int,
    num_workers: int,
):
    ddp_on = _ddp_enabled(cfg)
    ddp_rank = _ddp_rank(cfg)
    ddp_world_size = _ddp_world_size(cfg)
    seed = int(cfg.get("seed", 0))

    clients = []
    for name, dd in domains.items():
        if name == target_domain:
            continue
        clients.append(
            {
                "name": name,
                "train_loader": make_loader(
                    dd.train,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    distributed=ddp_on,
                    rank=ddp_rank,
                    world_size=ddp_world_size,
                    seed=seed,
                ),
                "val_loader": make_loader(dd.val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
                "train_dataset": dd.train,
                "n_train": len(dd.train),
            }
        )
    test_loader = make_loader(domains[target_domain].test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return clients, test_loader


def _save_history(outdir: Path, history: Dict[str, Any]):
    save_json(history, outdir / "history.json")


def _update_live_results(cfg: Dict[str, Any], history: Dict[str, Any], final_acc: float, seconds: float) -> None:
    if not _is_main_process(cfg):
        return
    runtime = cfg.get("_runtime") if isinstance(cfg, dict) else None
    if not runtime:
        return
    outdir = runtime.get("outdir")
    if not outdir:
        return
    outdir = Path(outdir)
    _save_history(outdir, history)

    summary = runtime.get("summary")
    summary = dict(summary) if isinstance(summary, dict) else {}
    summary["final_acc"] = float(final_acc)
    summary["seconds"] = float(seconds)
    save_json({"summary": summary, "raw": {"history": history, "final_acc": float(final_acc)}}, outdir / "result.json")


def _norm_weights_from_counts(counts: List[int]) -> List[float]:
    total = float(sum(int(c) for c in counts))
    if total <= 0:
        return [1.0 / max(1, len(counts)) for _ in counts]
    return [float(c) / total for c in counts]


def _format_class_name(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ")


def _template_text(template: str, class_name: str) -> str:
    cname = _format_class_name(class_name)
    if "{class}" in template:
        return template.format(**{"class": cname})
    if "{}" in template:
        return template.format(cname)
    return f"{template} {cname}"


def _build_text_set(class_names: List[str], templates: List[str]) -> List[str]:
    texts: List[str] = []
    for tpl in templates:
        for cname in class_names:
            texts.append(_template_text(str(tpl), cname))
    return texts


@register_algorithm("fedavg")
class FedAvgRunner:
    """Simple supervised FedAvg baseline (classifier finetuning)."""

    def run(self, cfg: Dict[str, Any], domains: Dict[str, Any]) -> Dict[str, Any]:
        algo = cfg["algorithm"]
        target = cfg["dataset"]["target_domain"]

        rounds = int(algo.get("rounds", 100))
        local_epochs = int(algo.get("local_epochs", 1))
        batch_size = int(algo.get("batch_size", 32))
        lr = float(algo.get("lr", 0.001))
        momentum = float(algo.get("momentum", 0.9))
        wd = float(algo.get("wd", 5e-4))
        show_progress = bool(algo.get("tqdm", True)) and _is_main_process(cfg)

        num_workers = int(cfg.get("num_workers", 4))
        device = _device_from_cfg(cfg)
        ddp_on = _ddp_enabled(cfg)
        parallel_gpu_ids = _resolve_multi_gpu_ids(cfg)
        if ddp_on and _is_main_process(cfg):
            print(f"[FedAvg] DDP enabled rank={_ddp_rank(cfg)}/{_ddp_world_size(cfg)} device={device}")
        if (not ddp_on) and len(parallel_gpu_ids) > 1 and _is_main_process(cfg):
            print("[FedAvg] multi_gpu=true but DDP is not initialized; using single-process training.")
        num_classes = next(iter(domains.values())).num_classes

        clients, test_loader = _build_clients(cfg, domains, target, batch_size, num_workers)
        if len(clients) == 0:
            raise RuntimeError("No source clients found. Check dataset.target_domain and protocol settings.")
        global_model, _ = build_classifier_from_cfg(cfg, num_classes=num_classes, device=device, pretrained=True)
        global_model = _wrap_ddp(global_model, cfg, device)

        history = {"round": [], "test_acc": [], "test_loss": []}
        t_start = time.time()

        for r in range(1, rounds + 1):
            updates = []
            for ci, c in enumerate(clients):
                local, _ = build_classifier_from_cfg(cfg, num_classes=num_classes, device=device)
                local = _wrap_ddp(local, cfg, device)
                local.load_state_dict(global_model.state_dict(), strict=True)
                opt = SGD(local.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

                for e in range(local_epochs):
                    _set_loader_epoch(
                        c["train_loader"],
                        (r - 1) * max(1, len(clients) * local_epochs) + ci * local_epochs + e,
                    )
                    progress = None
                    if show_progress:
                        progress = {
                            "enabled": True,
                            "desc": f"round {r} client {c['name']} epoch {e + 1}/{local_epochs}",
                            "leave": False,
                            "dynamic_ncols": True,
                        }
                    train_classifier_epoch(local, c["train_loader"], opt, device, progress=progress)

                updates.append((local.state_dict(), int(c["n_train"])))

            avg_sd = fedavg_state_dict(updates)
            global_model.load_state_dict(avg_sd, strict=True)

            test_loss, test_acc = eval_classifier(global_model, test_loader, device)
            history["round"].append(r)
            history["test_acc"].append(float(test_acc))
            history["test_loss"].append(float(test_loss))

            _update_live_results(cfg, history, final_acc=float(test_acc), seconds=time.time() - t_start)
            if _is_main_process(cfg) and (r == 1 or r % int(algo.get("log_every", 1)) == 0):
                print(f"[Round {r:03d}] test_acc={test_acc * 100:.2f} test_loss={test_loss:.4f}")

        return {"history": history, "final_acc": float(history["test_acc"][-1])}


@register_algorithm("fedsvp")
class FedSVPRunner:
    """Fed-SVP runner: client Fed-MSVP + server Fed-SDWA."""

    def run(self, cfg: Dict[str, Any], domains: Dict[str, Any]) -> Dict[str, Any]:
        algo = cfg["algorithm"]
        ds_cfg = cfg["dataset"]
        model_cfg = cfg.get("model", {}) or {}
        target = ds_cfg["target_domain"]

        rounds = int(algo.get("rounds", 100))
        local_epochs = int(algo.get("local_epochs", 1))
        batch_size = int(algo.get("batch_size", 32))
        lr = float(algo.get("lr", 1e-4))
        wd = float(algo.get("wd", 1e-4))
        tau = float(algo.get("tau", 0.2))
        lambda_consistency = float(algo.get("lambda_consistency", 0.5))
        use_gca = bool(algo.get("use_gca", True))
        lambda_contrastive = float(algo.get("lambda_contrastive", 0.2))
        contrastive_tau = float(algo.get("contrastive_tau", 0.07))
        prototype_max_batches = int(algo.get("prototype_max_batches", -1))
        show_progress = bool(algo.get("tqdm", True)) and _is_main_process(cfg)
        prompt_mode = str(algo.get("prompt_mode", "msvp")).lower()
        server_agg = str(algo.get("server_agg", "sdwa")).lower()
        max_score_batches = int(algo.get("max_score_batches", -1))
        if not use_gca:
            lambda_contrastive = 0.0
        gca_active = bool(use_gca and lambda_contrastive > 0.0)

        num_workers = int(cfg.get("num_workers", 4))
        device = _device_from_cfg(cfg)
        ddp_on = _ddp_enabled(cfg)
        parallel_gpu_ids = _resolve_multi_gpu_ids(cfg)
        if ddp_on and _is_main_process(cfg):
            print(f"[FedSVP] DDP enabled rank={_ddp_rank(cfg)}/{_ddp_world_size(cfg)} device={device}")
        if (not ddp_on) and len(parallel_gpu_ids) > 1 and _is_main_process(cfg):
            print("[FedSVP] multi_gpu=true but DDP is not initialized; using single-process training.")

        clients, test_loader = _build_clients(cfg, domains, target, batch_size, num_workers)
        if len(clients) == 0:
            raise RuntimeError("No source clients found. Check dataset.target_domain and protocol settings.")

        open_clip_cfg = (model_cfg.get("open_clip", {}) or {})
        model_name = str(open_clip_cfg.get("name", "ViT-B-16"))
        pretrained = str(open_clip_cfg.get("pretrained", "openai"))

        model_core = PromptedOpenCLIPVision(
            model_name=model_name,
            pretrained=pretrained,
            prompt_mode=prompt_mode,
        ).to(device)
        model_core.freeze_backbone()
        model_train = _wrap_ddp(model_core, cfg, device)

        class_names = list(domains[target].class_names)
        num_classes = int(len(class_names))
        class_template = str(algo.get("class_template", "a photo of a {}"))
        class_texts = [_template_text(class_template, c) for c in class_names]
        class_text_features = model_core.encode_text(class_texts, device=device, normalize=True)
        feature_dim = int(class_text_features.size(1))

        anchor_templates = algo.get(
            "anchor_templates",
            ["a photo of a {}", "a sketch of a {}", "a cartoon of a {}", "an art painting of a {}"],
        )
        if not isinstance(anchor_templates, list) or len(anchor_templates) == 0:
            anchor_templates = ["a photo of a {}"]
        anchor_texts = _build_text_set(class_names, [str(t) for t in anchor_templates])
        anchor_text_features = model_core.encode_text(anchor_texts, device=device, normalize=True)

        # For SDWA quality scoring, use a small proxy dataset.
        public_cfg = algo.get("public", {}) or {}
        proxy_loader = None
        num_public = int(public_cfg.get("num_public", 256))
        if num_public > 0:
            if public_cfg.get("root"):
                proxy_loader = build_public_loader_from_dir(
                    public_root=str(public_cfg["root"]),
                    batch_size=int(public_cfg.get("batch_size", batch_size)),
                    num_workers=num_workers,
                    num_public=num_public,
                    image_size=int(ds_cfg.get("image_size", 224)),
                    transforms=str(ds_cfg.get("transforms", "clip")),
                )
            else:
                proxy_loader = build_public_loader_from_clients(
                    client_train_datasets=[c["train_dataset"] for c in clients],
                    batch_size=int(public_cfg.get("batch_size", batch_size)),
                    num_workers=num_workers,
                    num_public=num_public,
                    seed=int(cfg.get("seed", 0)),
                    image_size=int(ds_cfg.get("image_size", 224)),
                    transforms=str(ds_cfg.get("transforms", "clip")),
                )

        logit_scale = float(algo.get("logit_scale", model_core.get_logit_scale()))
        global_prompt_state = model_core.prompt_state_dict_cpu()
        global_prototypes: Optional[torch.Tensor] = None
        global_proto_mask: Optional[torch.Tensor] = None

        history = {
            "round": [],
            "test_acc": [],
            "test_loss": [],
            "semantic_scores": [],
            "agg_weights": [],
            "proto_coverage": [],
            "global_proto_valid_classes": [],
            "client_names": [c["name"] for c in clients],
        }
        t_start = time.time()

        for r in range(1, rounds + 1):
            client_states: List[Dict[str, torch.Tensor]] = []
            client_sizes: List[int] = []
            semantic_scores: List[float] = []
            client_local_protos: List[torch.Tensor] = []
            client_local_counts: List[torch.Tensor] = []

            for ci, c in enumerate(clients):
                model_core.load_prompt_state_dict(global_prompt_state)
                opt = AdamW(model_core.prompt.parameters(), lr=lr, weight_decay=wd)

                for e in range(local_epochs):
                    _set_loader_epoch(
                        c["train_loader"],
                        (r - 1) * max(1, len(clients) * local_epochs) + ci * local_epochs + e,
                    )
                    progress = None
                    if show_progress:
                        progress = {
                            "enabled": True,
                            "desc": f"round {r} client {c['name']} epoch {e + 1}/{local_epochs}",
                            "leave": False,
                            "dynamic_ncols": True,
                        }
                    train_fedsvp_epoch(
                        model=model_train,
                        loader=c["train_loader"],
                        class_text_features=class_text_features,
                        optimizer=opt,
                        device=device,
                        lambda_consistency=lambda_consistency,
                        logit_scale=logit_scale,
                        global_prototypes=global_prototypes,
                        global_proto_mask=global_proto_mask,
                        lambda_contrastive=lambda_contrastive,
                        contrastive_tau=contrastive_tau,
                        progress=progress,
                    )

                if gca_active:
                    local_proto, local_counts = collect_local_class_prototypes(
                        model=model_core,
                        loader=c["train_loader"],
                        num_classes=num_classes,
                        device=device,
                        feature_dim=feature_dim,
                        max_batches=prototype_max_batches,
                    )
                    client_local_protos.append(local_proto)
                    client_local_counts.append(local_counts)

                local_state = model_core.prompt_state_dict_cpu()
                client_states.append(local_state)
                client_sizes.append(int(c["n_train"]))

                if proxy_loader is not None and server_agg in ("sdwa", "semantic", "semantic_softmax"):
                    score = semantic_score_on_proxy(
                        model=model_core,
                        prompt_state=local_state,
                        proxy_loader=proxy_loader,
                        anchor_text_features=anchor_text_features,
                        device=device,
                        max_batches=max_score_batches,
                    )
                else:
                    score = float(c["n_train"])
                semantic_scores.append(float(score))

            # Fed-SDWA (or ablation fallback to FedAvg on prompts)
            if server_agg in ("fedavg", "avg", "mean") or proxy_loader is None:
                weights = _norm_weights_from_counts(client_sizes)
            else:
                weights = semantic_softmax_weights(semantic_scores, tau=tau)

            global_prompt_state = weighted_average_state_dict(client_states, weights)
            model_core.load_prompt_state_dict(global_prompt_state)

            if gca_active and len(client_local_protos) > 0:
                global_prototypes, global_proto_mask = aggregate_class_prototypes(
                    client_prototypes=client_local_protos,
                    client_counts=client_local_counts,
                    normalize=True,
                )
                valid_classes = int(global_proto_mask.sum().item())
                proto_coverage = float(valid_classes / max(1, num_classes))
            else:
                global_prototypes = None
                global_proto_mask = None
                valid_classes = 0
                proto_coverage = 0.0

            test_loss, test_acc = eval_fedsvp(
                model=model_train,
                loader=test_loader,
                class_text_features=class_text_features,
                device=device,
                logit_scale=logit_scale,
            )

            history["round"].append(r)
            history["test_acc"].append(float(test_acc))
            history["test_loss"].append(float(test_loss))
            history["semantic_scores"].append([float(s) for s in semantic_scores])
            history["agg_weights"].append([float(w) for w in weights])
            history["proto_coverage"].append(float(proto_coverage))
            history["global_proto_valid_classes"].append(int(valid_classes))

            _update_live_results(cfg, history, final_acc=float(test_acc), seconds=time.time() - t_start)

            if _is_main_process(cfg) and (r == 1 or r % int(algo.get("log_every", 1)) == 0):
                if len(semantic_scores) > 0:
                    s_min = min(semantic_scores)
                    s_max = max(semantic_scores)
                else:
                    s_min = 0.0
                    s_max = 0.0
                print(
                    f"[Round {r:03d}] test_acc={test_acc * 100:.2f} "
                    f"test_loss={test_loss:.4f} score_range=[{s_min:.4f},{s_max:.4f}] "
                    f"proto_cov={proto_coverage:.2f}"
                )

        return {"history": history, "final_acc": float(history["test_acc"][-1])}

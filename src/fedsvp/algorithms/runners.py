from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW, SGD

from fedsvp.registry import register_algorithm
from fedsvp.data.base import make_loader
from fedsvp.data.public import build_public_loader_from_clients, build_public_loader_from_dir
from fedsvp.models.clip_backbone import build_classifier_from_cfg
from fedsvp.models.fedsvp_clip import PromptedOpenCLIPVision
from fedsvp.algorithms.agg import fedavg_state_dict, semantic_softmax_weights, weighted_average_state_dict
from fedsvp.algorithms.local_train import train_classifier_epoch, eval_classifier
from fedsvp.algorithms.fedsvp_train import train_fedsvp_epoch, eval_fedsvp, semantic_score_on_proxy
from fedsvp.utils.io import save_json


def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _build_clients(domains: Dict[str, Any], target_domain: str, batch_size: int, num_workers: int):
    clients = []
    for name, dd in domains.items():
        if name == target_domain:
            continue
        clients.append(
            {
                "name": name,
                "train_loader": make_loader(dd.train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
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
        show_progress = bool(algo.get("tqdm", True))

        num_workers = int(cfg.get("num_workers", 4))
        device = _device_from_cfg(cfg)
        num_classes = next(iter(domains.values())).num_classes

        clients, test_loader = _build_clients(domains, target, batch_size, num_workers)
        if len(clients) == 0:
            raise RuntimeError("No source clients found. Check dataset.target_domain and protocol settings.")
        global_model, _ = build_classifier_from_cfg(cfg, num_classes=num_classes, device=device, pretrained=True)

        history = {"round": [], "test_acc": [], "test_loss": []}
        t_start = time.time()

        for r in range(1, rounds + 1):
            updates = []
            for c in clients:
                local, _ = build_classifier_from_cfg(cfg, num_classes=num_classes, device=device)
                local.load_state_dict(global_model.state_dict(), strict=True)
                opt = SGD(local.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

                for e in range(local_epochs):
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
            if r == 1 or r % int(algo.get("log_every", 1)) == 0:
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
        show_progress = bool(algo.get("tqdm", True))
        prompt_mode = str(algo.get("prompt_mode", "msvp")).lower()
        server_agg = str(algo.get("server_agg", "sdwa")).lower()
        max_score_batches = int(algo.get("max_score_batches", -1))

        num_workers = int(cfg.get("num_workers", 4))
        device = _device_from_cfg(cfg)

        clients, test_loader = _build_clients(domains, target, batch_size, num_workers)
        if len(clients) == 0:
            raise RuntimeError("No source clients found. Check dataset.target_domain and protocol settings.")

        open_clip_cfg = (model_cfg.get("open_clip", {}) or {})
        model_name = str(open_clip_cfg.get("name", "ViT-B-16"))
        pretrained = str(open_clip_cfg.get("pretrained", "openai"))

        model = PromptedOpenCLIPVision(
            model_name=model_name,
            pretrained=pretrained,
            prompt_mode=prompt_mode,
        ).to(device)
        model.freeze_backbone()

        class_names = list(domains[target].class_names)
        class_template = str(algo.get("class_template", "a photo of a {}"))
        class_texts = [_template_text(class_template, c) for c in class_names]
        class_text_features = model.encode_text(class_texts, device=device, normalize=True)

        anchor_templates = algo.get(
            "anchor_templates",
            ["a photo of a {}", "a sketch of a {}", "a cartoon of a {}", "an art painting of a {}"],
        )
        if not isinstance(anchor_templates, list) or len(anchor_templates) == 0:
            anchor_templates = ["a photo of a {}"]
        anchor_texts = _build_text_set(class_names, [str(t) for t in anchor_templates])
        anchor_text_features = model.encode_text(anchor_texts, device=device, normalize=True)

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

        logit_scale = float(algo.get("logit_scale", model.get_logit_scale()))
        global_prompt_state = model.prompt_state_dict_cpu()

        history = {
            "round": [],
            "test_acc": [],
            "test_loss": [],
            "semantic_scores": [],
            "agg_weights": [],
            "client_names": [c["name"] for c in clients],
        }
        t_start = time.time()

        for r in range(1, rounds + 1):
            client_states: List[Dict[str, torch.Tensor]] = []
            client_sizes: List[int] = []
            semantic_scores: List[float] = []

            for c in clients:
                model.load_prompt_state_dict(global_prompt_state)
                opt = AdamW(model.prompt.parameters(), lr=lr, weight_decay=wd)

                for e in range(local_epochs):
                    progress = None
                    if show_progress:
                        progress = {
                            "enabled": True,
                            "desc": f"round {r} client {c['name']} epoch {e + 1}/{local_epochs}",
                            "leave": False,
                            "dynamic_ncols": True,
                        }
                    train_fedsvp_epoch(
                        model=model,
                        loader=c["train_loader"],
                        class_text_features=class_text_features,
                        optimizer=opt,
                        device=device,
                        lambda_consistency=lambda_consistency,
                        logit_scale=logit_scale,
                        progress=progress,
                    )

                local_state = model.prompt_state_dict_cpu()
                client_states.append(local_state)
                client_sizes.append(int(c["n_train"]))

                if proxy_loader is not None and server_agg in ("sdwa", "semantic", "semantic_softmax"):
                    score = semantic_score_on_proxy(
                        model=model,
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
            model.load_prompt_state_dict(global_prompt_state)

            test_loss, test_acc = eval_fedsvp(
                model=model,
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

            _update_live_results(cfg, history, final_acc=float(test_acc), seconds=time.time() - t_start)

            if r == 1 or r % int(algo.get("log_every", 1)) == 0:
                if len(semantic_scores) > 0:
                    s_min = min(semantic_scores)
                    s_max = max(semantic_scores)
                else:
                    s_min = 0.0
                    s_max = 0.0
                print(
                    f"[Round {r:03d}] test_acc={test_acc * 100:.2f} "
                    f"test_loss={test_loss:.4f} score_range=[{s_min:.4f},{s_max:.4f}]"
                )

        return {"history": history, "final_acc": float(history["test_acc"][-1])}

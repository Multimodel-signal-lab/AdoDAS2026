#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .data.dataset import FeatureConfig, ITEM_COLS, A1_COLS
from .data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from .models.mtcn_backbone import BackboneConfig, MTCNBackbone
from .models.heads import A1Head, A2OrdinalHead, a1_loss, a2_ordinal_loss
from .models.grouped_model import GroupedModel, CORALHead
from .utils.seed import seed_everything
from .utils.metrics import binary_f1, macro_auroc, per_class_f1, mean_qwk, mean_mae, per_item_qwk
from .utils.ckpt import save_checkpoint, load_checkpoint
from .utils.run_naming import build_run_name, setup_run_dirs
from .utils.run_metadata import RunMetadata

log = logging.getLogger("train_grouped")


class _RealtimeFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        if self.stream is None:
            return
        try:
            os.fsync(self.stream.fileno())
        except OSError:
            pass

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["a1", "a2"])
    p.add_argument("--config", type=str, default="configs/default.yaml")

    p.add_argument("--feature_root", type=str, default=None)
    p.add_argument("--manifest_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--audio_features", nargs="+", default=None)
    p.add_argument("--video_features", nargs="+", default=None)
    p.add_argument("--audio_ssl_model_tag", type=str, default=None)
    p.add_argument("--video_ssl_model_tag", type=str, default=None)

    p.add_argument("--mask_policy", type=str, default=None, choices=['or', 'and_core', 'require_k'])

    p.add_argument("--d_adapter", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--tcn_layers", type=int, default=None)
    p.add_argument("--tcn_kernel_size", type=int, default=None)
    p.add_argument("--asp_alpha", type=float, default=None)
    p.add_argument("--asp_beta", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--d_shared", type=int, default=None)

    p.add_argument("--aggregator", type=str, default=None, choices=["mean", "mlp", "attention"])
    p.add_argument("--session_loss_weight", type=float, default=None)
    p.add_argument("--session_type_loss_weight", type=float, default=None)
    p.add_argument("--use_coral", type=int, default=None, help="1=use CORAL head for A2")

    p.add_argument("--submission_level", type=str, default=None,
                    choices=["session", "participant"], help="Use participant-level preds for submission")
    p.add_argument("--decode_method", type=str, default=None,
                    choices=["auto", "argmax", "expectation", "monotonic"],
                    help="A2 decode: auto-select on val, or use argmax / expectation / monotonic")
    p.add_argument("--label_smoothing", type=float, default=None, help="Label smoothing factor")
    p.add_argument("--feature_noise_std", type=float, default=None, help="Gaussian noise std on features during training")
    p.add_argument("--session_drop_prob", type=float, default=None, help="Prob of dropping a session during training")
    p.add_argument("--early_stop_metric", type=str, default=None,
                    choices=["primary", "val_loss"], help="Metric for early stopping")

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--amp", type=int, default=None)
    p.add_argument("--preload", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--run_inference_after_train", type=int, default=None)

    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    cfg = cfg or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping in the config YAML")
    cfg.update(feature_selection)
    for k, v in vars(args).items():
        if k == "config":
            continue
        if v is not None:
            cfg[k] = v
    return cfg



def setup_logging(log_dir: Path, task: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_grouped_{task}_{ts}.log"
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = _RealtimeFileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass

    root.addHandler(ch)
    root.addHandler(fh)
    log.info(f"Logging to {log_file}")


class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  
        self.best_score: float | None = None
        self.counter = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _build_scheduler(optimizer, warmup_epochs, total_epochs):
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)


def _flatten_valid_session_mask(session_valid: torch.Tensor) -> torch.Tensor:
    return session_valid.reshape(-1).bool()


def _normalize_decode_method(decode_method: str | None) -> str:
    if decode_method is None:
        return "argmax"

    method = str(decode_method).strip().lower()
    valid_methods = {"auto", "argmax", "expectation", "monotonic"}
    if method not in valid_methods:
        raise ValueError(
            f"Unsupported decode_method: {decode_method!r}. "
            f"Expected one of {sorted(valid_methods)}"
        )
    return method


def _decode_a2_logits(task_head: nn.Module, logits: torch.Tensor, decode_method: str = "expectation") -> torch.Tensor:
    method = _normalize_decode_method(decode_method)
    if method == "auto":
        raise ValueError("decode_method='auto' is selection-only; pass a concrete decode method")

    if method == "expectation":
        decode_name = "predict_expectation"
    elif method == "monotonic":
        decode_name = "predict_int_monotonic"
    else:
        decode_name = "predict_int"

    decode_fn = getattr(task_head, decode_name, None)
    if decode_fn is None:
        decode_fn = getattr(A2OrdinalHead, decode_name)
    return decode_fn(logits.float())


def _evaluate_a2_decode_candidates(
    task_head: nn.Module,
    logits: torch.Tensor,
    labels: np.ndarray,
    decode_methods: list[str],
    offsets: np.ndarray | None = None,
) -> dict[str, dict[str, float | np.ndarray | str]]:
    logits_f = logits.float()
    if offsets is not None:
        logits_f = logits_f + torch.as_tensor(offsets, device=logits_f.device, dtype=torch.float32)

    results: dict[str, dict[str, float | np.ndarray | str]] = {}
    for method in decode_methods:
        preds = _decode_a2_logits(task_head, logits_f, decode_method=method).cpu().numpy()
        qwk = mean_qwk(preds, labels)
        mae = mean_mae(preds, labels)
        results[method] = {
            "preds": preds,
            "qwk": qwk,
            "mae": mae,
            "decode_method": method,
        }
    return results


def _select_best_a2_result(results: dict[str, dict[str, float | np.ndarray | str]]) -> tuple[str, dict[str, float | np.ndarray | str]]:
    best_name = max(
        results,
        key=lambda name: (
            float(results[name]["qwk"]),
            -float(results[name]["mae"]),
        ),
    )
    return best_name, results[best_name]


def _compute_pos_weight_a1(manifest_path: Path) -> list[float]:
    df = pd.read_csv(manifest_path)
    weights = []
    for col in ["y_D", "y_A", "y_S"]:
        n_pos = df[col].sum()
        n_neg = len(df) - n_pos
        w = float(np.sqrt(n_neg / max(n_pos, 1)))
        w = max(1.0, min(w, 4.0))
        weights.append(w)
    return weights


def _compute_bias_init_a1(manifest_path: Path) -> list[float]:
    df = pd.read_csv(manifest_path)
    biases = []
    for col in ["y_D", "y_A", "y_S"]:
        rate = df[col].mean()
        rate = max(min(rate, 0.99), 0.01)
        biases.append(math.log(rate / (1 - rate)))
    return biases


def compute_a2_pos_weight(manifest_path: Path, n_items=21, n_thresholds=3):
    df = pd.read_csv(manifest_path)
    item_cols = [f"d{i:02d}" for i in range(1, n_items + 1)]
    pw = np.ones((n_items, n_thresholds), dtype=np.float32)
    for j, col in enumerate(item_cols):
        vals = df[col].values.astype(int)
        for k in range(n_thresholds):
            p = max(np.mean(vals >= (k + 1)), 1e-6)
            pw[j, k] = np.clip(np.sqrt((1 - p) / p), 1.0, 10.0)
    return torch.from_numpy(pw).unsqueeze(0)

def train_one_epoch_grouped(
    grouped_model: GroupedModel,
    task_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    scaler=None,
    use_amp: bool = False,
    pos_weight=None,
    grad_clip: float = 1.0,
    session_loss_weight: float = 0.5,
    session_type_loss_weight: float = 0.15,
    best_metric: float = -1.0,
    label_smoothing: float = 0.0,
    feature_noise_std: float = 0.0,
) -> float:
    grouped_model.train()
    task_head.train()
    total_loss = 0.0
    n_batches = 0

    desc = f"Train {epoch}/{epochs}"
    if best_metric >= 0:
        desc += f" [best={best_metric:.4f}]"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    for batch in pbar:
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        session_types = batch["session_types"].to(device)
        B = batch["n_participants"]

        if feature_noise_std > 0.0:
            noise_mask = (~flat_batch["pad_mask"]).unsqueeze(-1).float()
            for key in ("audio_groups", "video_groups"):
                for name in flat_batch[key]:
                    flat_batch[key][name] = flat_batch[key][name] + torch.randn_like(
                        flat_batch[key][name]
                    ) * feature_noise_std * noise_mask

        if task == "a1":
            targets = batch["participant_y_a1"].to(device)
        else:
            targets = batch["participant_y_a2"].to(device).long()

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            valid_session_mask = _flatten_valid_session_mask(session_valid)
            has_valid_sessions = bool(valid_session_mask.any().item())

            p_logits = task_head(out["participant_repr"])
            if task == "a1":
                main_loss = a1_loss(p_logits, targets, pos_weight=pos_weight, label_smoothing=label_smoothing)
            else:
                main_loss = a2_ordinal_loss(p_logits, targets, pos_weight=pos_weight, label_smoothing=label_smoothing)

            if has_valid_sessions:
                s_logits = task_head(out["session_reprs"])[valid_session_mask]
                if task == "a1":
                    s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)[valid_session_mask]
                    sess_loss = a1_loss(s_logits, s_targets, pos_weight=pos_weight, label_smoothing=label_smoothing)
                else:
                    s_targets = targets.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21)[valid_session_mask]
                    sess_loss = a2_ordinal_loss(
                        s_logits, s_targets, pos_weight=pos_weight, label_smoothing=label_smoothing
                    )

                type_loss = F.cross_entropy(
                    out["session_type_logits"][valid_session_mask],
                    session_types[valid_session_mask],
                )
            else:
                sess_loss = p_logits.new_zeros(())
                type_loss = p_logits.new_zeros(())

            loss = main_loss + session_loss_weight * sess_loss + session_type_loss_weight * type_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(grouped_model.parameters()) + list(task_head.parameters()),
                max_norm=grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(grouped_model.parameters()) + list(task_head.parameters()),
                max_norm=grad_clip,
            )
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix_str(f"{loss.item():.4f}")

    pbar.close()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_grouped(
    grouped_model: GroupedModel,
    task_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    epoch: int,
    epochs: int,
    use_amp: bool = False,
    pos_weight=None,
    decode_method: str = "expectation",
):
    """Validate grouped model. Returns metrics dict."""
    grouped_model.eval()
    task_head.eval()
    decode_method = _normalize_decode_method(decode_method)
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    all_logits = []
    all_sess_preds = []

    for batch in tqdm(loader, desc=f"Val {epoch}/{epochs}", leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]

        if task == "a1":
            targets = batch["participant_y_a1"].to(device)
        else:
            targets = batch["participant_y_a2"].to(device).long()

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            p_logits = task_head(out["participant_repr"])
            if task == "a1":
                loss = a1_loss(p_logits, targets, pos_weight=pos_weight)
            else:
                loss = a2_ordinal_loss(p_logits, targets, pos_weight=pos_weight)

            s_logits = task_head(out["session_reprs"])

        if task == "a1":
            logits_np = p_logits.float().cpu().numpy()
            probs = torch.sigmoid(p_logits.float()).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(targets.cpu().numpy())
            all_logits.append(logits_np)

            s_probs = torch.sigmoid(s_logits.float()).cpu().numpy()
            all_sess_preds.append(s_probs)
        else:
            if decode_method == "auto":
                all_logits.append(p_logits.float().cpu())
            else:
                preds = _decode_a2_logits(task_head, p_logits, decode_method=decode_method)
                all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    if task == "a1":
        probs_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)
        logits_np = np.concatenate(all_logits)
        mf1 = binary_f1(probs_np, labels_np, threshold=0.5)
        auroc = macro_auroc(probs_np, labels_np)
        pcf1 = per_class_f1(probs_np, labels_np, threshold=0.5)
        cal_biases, cal_pcf1 = calibrate_a1_bias(logits_np, labels_np)
        cal_logits_np = logits_np + cal_biases.reshape(1, -1)
        cal_probs_np = 1.0 / (1.0 + np.exp(-cal_logits_np))
        cal_mf1 = binary_f1(cal_probs_np, labels_np, threshold=0.5)
        selection_source = "calibrated" if cal_mf1 > mf1 else "raw"

        task_names = ["D", "A", "S"]
        for t, name in enumerate(task_names):
            gt = labels_np[:, t]
            pr = (probs_np[:, t] > 0.5).astype(int)
            gt_rate = gt.mean()
            pred_rate = pr.mean()
            p_mean = probs_np[:, t].mean()
            tp = ((pr == 1) & (gt == 1)).sum()
            prec = tp / max(pr.sum(), 1)
            rec = tp / max(gt.sum(), 1)
            log.info(
                f"    {name}: gt_pos={gt_rate:.3f} pred_pos={pred_rate:.3f} "
                f"p_mean={p_mean:.3f} P={prec:.3f} R={rec:.3f} F1={pcf1[t]:.3f}"
            )

        if all_sess_preds:
            sess_probs = np.concatenate(all_sess_preds)
            n_sess = sess_probs.shape[0]
            if n_sess % 4 == 0:
                n_part = n_sess // 4
                sess_grid = sess_probs.reshape(n_part, 4, 3)
                sess_var = np.mean(np.var(sess_grid, axis=1))
                log.info(f"    Session-level variance (collapse metric): {sess_var:.6f}")

        log.info(
            f"    calibrated F1={cal_mf1:.4f} via biases "
            f"D={cal_biases[0]:+.2f} A={cal_biases[1]:+.2f} S={cal_biases[2]:+.2f} "
            f"(selected={selection_source})"
        )

        return {
            "loss": avg_loss, "mean_f1": mf1, "auroc": auroc,
            "pcf1": pcf1,
            "mean_f1_calibrated": cal_mf1,
            "pcf1_calibrated": cal_pcf1,
            "calibration_biases": cal_biases.tolist(),
            "primary_metric": max(mf1, cal_mf1),
            "selection_source": selection_source,
        }
    else:
        labels_np = np.concatenate(all_labels)
        auto_selected_decode = None
        if decode_method == "auto":
            logits_t = torch.cat(all_logits, dim=0)
            raw_results = _evaluate_a2_decode_candidates(
                task_head,
                logits_t,
                labels_np,
                decode_methods=["argmax", "monotonic", "expectation"],
            )
            auto_selected_decode, best_result = _select_best_a2_result(raw_results)
            preds_np = best_result["preds"]
            log.info(
                f"    auto decode selected: {auto_selected_decode} "
                f"(QWK={float(best_result['qwk']):.4f}, MAE={float(best_result['mae']):.4f})"
            )
        else:
            preds_np = np.concatenate(all_preds)
        mqwk = mean_qwk(preds_np, labels_np)
        mmae = mean_mae(preds_np, labels_np)

        total = preds_np.size
        dist = [np.sum(preds_np == v) / total * 100 for v in range(4)]
        gt_dist = [np.sum(labels_np == v) / total * 100 for v in range(4)]
        log.info(f"    pred dist: 0={dist[0]:.1f}% 1={dist[1]:.1f}% 2={dist[2]:.1f}% 3={dist[3]:.1f}%")
        log.info(f"    GT   dist: 0={gt_dist[0]:.1f}% 1={gt_dist[1]:.1f}% 2={gt_dist[2]:.1f}% 3={gt_dist[3]:.1f}%")

        item_qwk = per_item_qwk(preds_np, labels_np)
        ranked = sorted(range(21), key=lambda i: item_qwk[i], reverse=True)
        top3 = " ".join(f"d{r+1:02d}={item_qwk[r]:.3f}" for r in ranked[:3])
        bot3 = " ".join(f"d{r+1:02d}={item_qwk[r]:.3f}" for r in ranked[-3:])
        log.info(f"    top3: {top3}  |  bot3: {bot3}")

        return {
            "loss": avg_loss, "mean_qwk": mqwk, "mean_mae": mmae,
            "primary_metric": mqwk, "selected_decode_method": auto_selected_decode,
        }



@torch.no_grad()
def generate_submission_grouped(
    grouped_model: GroupedModel,
    task_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    use_amp: bool = False,
    desc: str = "Submit",
    submission_level: str = "participant",
    a1_biases: np.ndarray | None = None,
    decode_method: str = "expectation",
    a2_threshold_offsets: np.ndarray | None = None,
):
    grouped_model.eval()
    task_head.eval()
    decode_method = _normalize_decode_method(decode_method)
    if submission_level not in {"participant", "session"}:
        raise ValueError("submission_level must be 'participant' or 'session'")

    all_pids = []
    all_sessions = []
    all_preds = []
    a1_biases_t = None if a1_biases is None else torch.as_tensor(a1_biases, device=device, dtype=torch.float32)
    a2_offsets_t = (
        None if a2_threshold_offsets is None
        else torch.as_tensor(a2_threshold_offsets, device=device, dtype=torch.float32)
    )

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)

            if submission_level == "participant":
                logits = task_head(out["participant_repr"])
            else:
                logits = task_head(out["session_reprs"])

        if task == "a1":
            logits_f = logits.float()
            if a1_biases_t is not None:
                logits_f = logits_f + a1_biases_t
            preds = torch.sigmoid(logits_f).cpu().numpy()
        else:
            logits_f = logits.float()
            if a2_offsets_t is not None:
                logits_f = logits_f + a2_offsets_t
            preds = _decode_a2_logits(task_head, logits_f, decode_method=decode_method).cpu().numpy()

        if submission_level == "participant":
            participant_ids = [str(pid) for pid in batch["anon_pids"]]
            all_pids.extend(participant_ids)
            all_sessions.extend(["participant"] * len(participant_ids))
        else:
            all_pids.extend(batch["flat_pids"])
            all_sessions.extend(batch["flat_sessions"])
        all_preds.append(preds)

    return all_pids, all_sessions, np.concatenate(all_preds)



@torch.no_grad()
def collect_val_logits_grouped_a1(grouped_model, task_head, loader, device, use_amp,
                                   submission_level="participant"):
    grouped_model.eval()
    task_head.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            if submission_level == "participant":
                logits = task_head(out["participant_repr"]).float().cpu().numpy()
                labels = batch["participant_y_a1"].numpy()
            else:
                valid_session_mask = _flatten_valid_session_mask(session_valid).cpu().numpy()
                logits = task_head(out["session_reprs"]).float().cpu().numpy()[valid_session_mask]
                labels = batch["participant_y_a1"].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3).numpy()
                labels = labels[valid_session_mask]
        all_logits.append(logits)
        all_labels.append(labels)
    return np.concatenate(all_logits), np.concatenate(all_labels)


@torch.no_grad()
def collect_val_logits_grouped_a2(grouped_model, task_head, loader, device, use_amp,
                                   submission_level="participant"):
    """Collect A2 logits and labels from validation set for calibration."""
    grouped_model.eval()
    task_head.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        B = batch["n_participants"]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, B, session_valid)
            if submission_level == "participant":
                logits = task_head(out["participant_repr"]).float().cpu().numpy()
                labels = batch["participant_y_a2"].numpy()
            else:
                valid_session_mask = _flatten_valid_session_mask(session_valid).cpu().numpy()
                logits = task_head(out["session_reprs"]).float().cpu().numpy()[valid_session_mask]
                labels = batch["participant_y_a2"].unsqueeze(1).expand(-1, 4, -1).reshape(-1, 21).numpy()
                labels = labels[valid_session_mask]
        all_logits.append(logits)
        all_labels.append(labels)
    return np.concatenate(all_logits), np.concatenate(all_labels)


def calibrate_a2_thresholds(logits, labels, n_items=21, n_thresholds=3,
                             grid_min=-2.0, grid_max=2.0, grid_step=0.1,
                             decode_method: str = "expectation"):
    import warnings
    from sklearn.metrics import cohen_kappa_score
    decode_method = _normalize_decode_method(decode_method)
    decode_head = A2OrdinalHead(1)
    grid = np.arange(grid_min, grid_max + grid_step, grid_step)
    offsets = np.zeros((n_items, n_thresholds), dtype=np.float64)
    item_qwks = []

    for j in range(n_items):
        best_qwk = -1.0
        best_offset = np.zeros(n_thresholds)

        # Single shared offset per item (simpler, less overfitting)
        for b in grid:
            shifted = logits[:, j, :] + b  # (N, 3)
            shifted_t = torch.from_numpy(shifted).float().unsqueeze(0)
            preds = _decode_a2_logits(task_head=decode_head, logits=shifted_t, decode_method=decode_method)
            preds = preds.squeeze(0).cpu().numpy().astype(int)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    qwk = cohen_kappa_score(labels[:, j].astype(int), preds, weights="quadratic")
                if not np.isfinite(qwk):
                    qwk = 0.0
            except Exception:
                qwk = 0.0
            if qwk > best_qwk:
                best_qwk = qwk
                best_offset = np.full(n_thresholds, b)

        offsets[j] = best_offset
        item_qwks.append(best_qwk)

    return offsets, item_qwks


def calibrate_a1_bias(logits, labels, grid_min=-3.0, grid_max=3.0, grid_step=0.1):
    from sklearn.metrics import f1_score as skf1
    grid = np.arange(grid_min, grid_max + grid_step, grid_step)
    biases = np.zeros(3, dtype=np.float64)
    best_f1s = []
    for t in range(3):
        best_f1 = -1.0
        best_b = 0.0
        for b in grid:
            probs = 1.0 / (1.0 + np.exp(-(logits[:, t] + b)))
            preds = (probs > 0.5).astype(int)
            f1 = skf1(labels[:, t], preds, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_b = b
        biases[t] = best_b
        best_f1s.append(best_f1)
    return biases, best_f1s



def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    task = cfg["task"]

    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = Path(cfg.get("output_dir", "/media/k3nwong/Data1/test/train/output"))
    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_run_name(cfg, task, timestamp, training_mode="grouped_participant")
    run_dirs = setup_run_dirs(output_root, run_name)

    setup_logging(run_dirs["logs"], task)
    log.info(f"Device: {device}")
    log.info(f"Task: {task}")
    log.info(f"Run name: {run_name}")
    log.info(f"Config: {cfg}")

    meta = RunMetadata(run_dirs["root"], cfg, task, run_name)

    _defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", _defaults.feature_root),
        audio_features=cfg.get("audio_features", _defaults.audio_features),
        video_features=cfg.get("video_features", _defaults.video_features),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", _defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", _defaults.video_ssl_model_tag),
        mask_policy=cfg.get("mask_policy", _defaults.mask_policy),
        core_audio=cfg.get("core_audio", _defaults.core_audio),
        core_video=cfg.get("core_video", _defaults.core_video),
    )
    log.info(f"Mask policy: {feat_cfg.mask_policy}")

    train_ds = GroupedParticipantDataset(
        manifest_dir / "train.csv", feat_cfg, split="train",
        session_drop_prob=cfg.get("session_drop_prob", 0.1),
    )
    val_ds = GroupedParticipantDataset(manifest_dir / "val.csv", feat_cfg, split="val")

    batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 8)
    log.info(f"Train: {len(train_ds)} participants, Val: {len(val_ds)} participants")

    preload = bool(cfg.get("preload", True))
    if preload:
        log.info("Preloading data into RAM ...")
        t_pre = time.time()
        train_gb = train_ds.preload(desc="Preload train")
        val_gb = val_ds.preload(desc="Preload val")
        log.info(f"Preload done: {train_gb:.1f}G + {val_gb:.1f}G = {train_gb + val_gb:.1f}G, "
                 f"took {_fmt_duration(time.time() - t_pre)}")
        num_workers = 0

    log.info(f"batch_size={batch_size}, num_workers={num_workers}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=grouped_collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=grouped_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    dims = train_ds.feature_dims
    audio_group_dims = {n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims}
    audio_pooled_group_dims = {n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims}
    video_group_dims = {n: dims[n] for n in feat_cfg.video_features if n in dims}

    bb_cfg = BackboneConfig(
        audio_group_dims=audio_group_dims,
        audio_pooled_group_dims=audio_pooled_group_dims,
        video_group_dims=video_group_dims,
        d_adapter=cfg.get("d_adapter", 64),
        d_model=cfg.get("d_model", 256),
        tcn_layers=cfg.get("tcn_layers", 6),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
        asp_alpha=cfg.get("asp_alpha", 0.5),
        asp_beta=cfg.get("asp_beta", 0.5),
        dropout=cfg.get("dropout", 0.2),
        d_shared=cfg.get("d_shared", 256),
    )

    backbone = MTCNBackbone(bb_cfg)
    grouped_model = GroupedModel(
        backbone=backbone,
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    use_coral = bool(cfg.get("use_coral", False))
    if task == "a1":
        bias_init = _compute_bias_init_a1(manifest_dir / "train.csv")
        task_head = A1Head(bb_cfg.d_shared, bias_init=bias_init).to(device)
    else:
        if use_coral:
            task_head = CORALHead(bb_cfg.d_shared).to(device)
            log.info("Using CORAL head for A2")
        else:
            task_head = A2OrdinalHead(bb_cfg.d_shared).to(device)

    n_params = sum(p.numel() for p in grouped_model.parameters()) + sum(p.numel() for p in task_head.parameters())
    log.info(f"Model params: {n_params:,}")

    use_amp = bool(cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        log.info("AMP enabled (BF16)")

    grad_clip = cfg.get("grad_clip", 1.0)
    pos_weight_t = None
    if cfg.get("use_pos_weight", True):
        if task == "a1":
            pw = _compute_pos_weight_a1(manifest_dir / "train.csv")
            pos_weight_t = torch.tensor(pw, dtype=torch.float32, device=device)
            log.info(f"pos_weight [D/A/S]: {pw[0]:.2f} / {pw[1]:.2f} / {pw[2]:.2f}")
        else:
            pos_weight_t = compute_a2_pos_weight(manifest_dir / "train.csv").to(device)
            log.info(f"A2 pos_weight shape: {pos_weight_t.shape}")

    params = list(grouped_model.parameters()) + list(task_head.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-2)
    )
    epochs = cfg.get("epochs", 20)
    warmup_epochs = cfg.get("warmup_epochs", 3)
    scheduler = _build_scheduler(optimizer, warmup_epochs, epochs)
    log.info(f"Scheduler: warmup={warmup_epochs} -> cosine, total={epochs}")
    log.info(f"Grad clip: {grad_clip}")

    session_loss_weight = cfg.get("session_loss_weight", 0.5)
    session_type_loss_weight = cfg.get("session_type_loss_weight", 0.15)
    log.info(f"Session loss weight: {session_loss_weight}")
    log.info(f"Session type loss weight: {session_type_loss_weight}")

    patience = cfg.get("patience", 8)
    early_stop_metric = cfg.get("early_stop_metric", "val_loss")
    es_mode = "min" if early_stop_metric == "val_loss" else "max"
    early_stop = EarlyStopping(patience=patience, mode=es_mode)
    log.info(f"EarlyStopping: patience={patience}, metric={early_stop_metric}, mode={es_mode}")

    label_smoothing = cfg.get("label_smoothing", 0.05)
    feature_noise_std = cfg.get("feature_noise_std", 0.01)
    session_drop_prob = cfg.get("session_drop_prob", 0.1)
    log.info(f"Label smoothing: {label_smoothing}")
    log.info(f"Feature noise std: {feature_noise_std}")
    log.info(f"Session drop prob: {session_drop_prob}")

    best_metric = -1.0
    metric_name = "F1" if task == "a1" else "QWK"
    t_start = time.time()

    log.info("=" * 90)
    if task == "a1":
        log.info("  Epoch  |    LR     | Train Loss | Val Loss | F1 raw | F1 sel |  AUROC | F1[D/A/S]       | Time")
    else:
        log.info("  Epoch  |    LR     | Train Loss | Val Loss | mean QWK | mean MAE | Time")
    log.info("=" * 90)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch_grouped(
            grouped_model, task_head, train_loader, optimizer, device,
            task, epoch, epochs, scaler, use_amp,
            pos_weight=pos_weight_t, grad_clip=grad_clip,
            session_loss_weight=session_loss_weight,
            session_type_loss_weight=session_type_loss_weight,
            best_metric=best_metric,
            label_smoothing=label_smoothing,
            feature_noise_std=feature_noise_std,
        )

        val_metrics = validate_grouped(
            grouped_model, task_head, val_loader, device,
            task, epoch, epochs, use_amp, pos_weight=pos_weight_t,
            decode_method=cfg.get("decode_method", "expectation"),
        )
        scheduler.step()

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (epochs - epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        primary = val_metrics["primary_metric"]
        is_best = primary > best_metric
        marker = " *" if is_best else ""

        if task == "a1":
            pcf1 = val_metrics.get("pcf1", [0, 0, 0])
            selected_f1 = val_metrics["primary_metric"]
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_metrics['loss']:.4f}  | "
                f"{val_metrics['mean_f1']:.4f} | {selected_f1:.4f} | {val_metrics['auroc']:.4f} | "
                f"{pcf1[0]:.3f}/{pcf1[1]:.3f}/{pcf1[2]:.3f} | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )
        else:
            log.info(
                f"  {epoch:3d}/{epochs:3d} | {lr_now:.2e} |   {train_loss:.4f}   |  {val_metrics['loss']:.4f}  | "
                f" {val_metrics['mean_qwk']:.4f}  |  {val_metrics['mean_mae']:.4f}  | "
                f"{_fmt_duration(elapsed)} ETA {_fmt_duration(eta)} VRAM {vram_gb:.1f}G{marker}"
            )

        if is_best:
            best_metric = primary
            save_checkpoint(
                run_dirs["checkpoints"] / "best.pt",
                grouped_model, optimizer, epoch, best_metric,
                extra={"head_state_dict": task_head.state_dict()},
            )
            log.info(f"  >>> New best {metric_name}={best_metric:.4f} saved at epoch {epoch}.")
            meta.update_best(epoch, val_metrics)

        es_value = val_metrics["loss"] if early_stop_metric == "val_loss" else primary
        if early_stop.step(es_value):
            log.info(f"  EarlyStopping triggered at epoch {epoch} (patience={patience}, metric={early_stop_metric})")
            break

    log.info("=" * 90)
    total_time = time.time() - t_start
    log.info(f"Training complete. Best {metric_name}={best_metric:.4f}, time={_fmt_duration(total_time)}")

    log.info("Loading best checkpoint for submission generation ...")
    state = load_checkpoint(run_dirs["checkpoints"] / "best.pt", grouped_model, optimizer=None)
    task_head.load_state_dict(state["head_state_dict"])
    grouped_model.to(device)
    task_head.to(device)

    submission_level = cfg.get("submission_level", "participant")
    decode_method = _normalize_decode_method(cfg.get("decode_method", "expectation"))
    log.info(f"Submission level: {submission_level}")
    log.info(f"Decode method: {decode_method}")

    a1_biases = None
    a2_offsets = None
    selected_decode_method = decode_method

    if task == "a1":
        log.info("Calibrating per-task bias offsets on val ...")
        val_logits, val_labels = collect_val_logits_grouped_a1(
            grouped_model, task_head, val_loader, device, use_amp,
            submission_level=submission_level,
        )
        biases, cal_f1s = calibrate_a1_bias(val_logits, val_labels)
        for t, name in enumerate(["D", "A", "S"]):
            log.info(f"  {name}: bias={biases[t]:+.2f}  F1_cal={cal_f1s[t]:.4f}")
        cal_mean_f1 = float(np.mean(cal_f1s))
        best_raw_f1 = float(meta.meta.get("best_metrics", {}).get("mean_f1", best_metric))
        best_selected_f1 = float(meta.meta.get("best_metrics", {}).get("primary_metric", best_metric))
        log.info(
            f"  Mean calibrated F1: {cal_mean_f1:.4f} "
            f"(vs selected best: {best_selected_f1:.4f}, raw best: {best_raw_f1:.4f})"
        )
        a1_biases = biases
        final_a1_metric = max(best_raw_f1, cal_mean_f1)
        final_a1_strategy = "bias_calibrated" if cal_mean_f1 >= best_raw_f1 else "raw"
        meta.set_extra("final_selected_strategy", final_a1_strategy)
        meta.set_extra("final_selected_metrics", {
            "mean_f1": final_a1_metric,
            "mean_f1_raw": best_raw_f1,
            "mean_f1_calibrated": cal_mean_f1,
            "auroc": meta.meta.get("best_metrics", {}).get("auroc"),
        })

        cal_data = {"biases": biases.tolist(), "cal_f1": cal_f1s, "mean_cal_f1": cal_mean_f1}
        with open(run_dirs["calibration"] / "a1_bias_grouped.json", "w") as f:
            json.dump(cal_data, f, indent=2)
    else:
        log.info("Calibrating and selecting A2 decode strategy on val ...")
        val_logits, val_labels = collect_val_logits_grouped_a2(
            grouped_model, task_head, val_loader, device, use_amp,
            submission_level=submission_level,
        )
        val_labels_int = val_labels.astype(int)
        raw_results = _evaluate_a2_decode_candidates(
            task_head,
            torch.from_numpy(val_logits).float(),
            val_labels_int,
            decode_methods=["argmax", "monotonic", "expectation"],
        )
        calibrated_results = {}
        for method in ("argmax", "monotonic", "expectation"):
            offsets, item_qwks = calibrate_a2_thresholds(
                val_logits,
                val_labels_int,
                decode_method=method,
            )
            preds = _decode_a2_logits(
                task_head,
                torch.from_numpy(val_logits).float() + torch.as_tensor(offsets, dtype=torch.float32),
                decode_method=method,
            ).cpu().numpy()
            calibrated_results[f"calibrated_{method}"] = {
                "preds": preds,
                "qwk": mean_qwk(preds, val_labels_int),
                "mae": mean_mae(preds, val_labels_int),
                "decode_method": method,
                "offsets": offsets,
                "item_qwks": item_qwks,
            }

        strategy_results = {**raw_results, **calibrated_results}
        best_strategy, best_result = _select_best_a2_result(strategy_results)
        selected_decode_method = str(best_result["decode_method"])
        a2_offsets = best_result.get("offsets")

        log.info("  A2 decode comparison on val:")
        for name in ("argmax", "monotonic", "expectation", "calibrated_argmax", "calibrated_monotonic", "calibrated_expectation"):
            result = strategy_results[name]
            preds = result["preds"]
            total = preds.size
            dist = [np.sum(preds == v) / total * 100 for v in range(4)]
            log.info(
                f"    {name:<22} QWK={float(result['qwk']):.4f} MAE={float(result['mae']):.4f} "
                f"| 0={dist[0]:.1f}% 1={dist[1]:.1f}% 2={dist[2]:.1f}% 3={dist[3]:.1f}%"
            )

        log.info(
            f"  Selected A2 strategy: {best_strategy} "
            f"(decode={selected_decode_method}, QWK={float(best_result['qwk']):.4f}, MAE={float(best_result['mae']):.4f})"
        )

        meta.set_extra("final_selected_strategy", best_strategy)
        meta.set_extra("final_selected_metrics", {
            "mean_qwk": float(best_result["qwk"]),
            "mean_mae": float(best_result["mae"]),
            "decode_method": selected_decode_method,
        })

        cal_data = {
            "selected_strategy": best_strategy,
            "selected_decode_method": selected_decode_method,
            "selected_qwk": float(best_result["qwk"]),
            "selected_mae": float(best_result["mae"]),
            "strategies": {
                name: {
                    "decode_method": str(result["decode_method"]),
                    "qwk": float(result["qwk"]),
                    "mae": float(result["mae"]),
                    **({"offsets": result["offsets"].tolist()} if "offsets" in result else {}),
                    **({"item_qwks": result["item_qwks"]} if "item_qwks" in result else {}),
                }
                for name, result in strategy_results.items()
            },
        }
        with open(run_dirs["calibration"] / "a2_threshold_offsets_grouped.json", "w") as f:
            json.dump(cal_data, f, indent=2)

    if bool(cfg.get("run_inference_after_train", False)):
        run_dirs["submissions"].mkdir(parents=True, exist_ok=True)
        for split_name in ("val", "test_hidden"):
            manifest_path = manifest_dir / f"{split_name}.csv"
            if not manifest_path.exists():
                continue
            ds = GroupedParticipantDataset(manifest_path, feat_cfg, split=split_name)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=grouped_collate_fn,
            )

            pids, sessions, preds = generate_submission_grouped(
                grouped_model, task_head, loader, device, task, use_amp,
                desc=f"Submit {split_name}",
                submission_level=submission_level,
                a1_biases=a1_biases,
                decode_method=selected_decode_method,
                a2_threshold_offsets=a2_offsets,
            )

            manifest_df = pd.read_csv(manifest_path)
            file_ids = []
            filtered_preds = []
            if submission_level == "participant":
                pid_to_info = {}
                for _, row in manifest_df.iterrows():
                    pid = str(row["anon_pid"])
                    pid_to_info.setdefault(pid, (str(row["anon_school"]), str(row["anon_class"])))

                for pid, pred in zip(pids, preds):
                    pid_str = str(pid)
                    info = pid_to_info.get(pid_str)
                    if info is None:
                        continue
                    school, cls = info
                    file_ids.append(f"{school}_{cls}_{pid_str}")
                    filtered_preds.append(pred)
                expected_rows = int(manifest_df["anon_pid"].astype(str).nunique())
            else:
                pid_to_info = {}
                for _, row in manifest_df.iterrows():
                    pid_to_info[(str(row["anon_pid"]), str(row["session"]))] = (
                        str(row["anon_school"]), str(row["anon_class"])
                    )

                for pid, sess, pred in zip(pids, sessions, preds):
                    key = (str(pid), str(sess))
                    info = pid_to_info.get(key)
                    if info is None:
                        continue
                    school, cls = info
                    file_ids.append(f"{school}_{cls}_{key[0]}_{key[1]}")
                    filtered_preds.append(pred)
                expected_rows = len(manifest_df)

            if filtered_preds:
                preds = np.asarray(filtered_preds)
            elif task == "a1":
                preds = np.zeros((0, 3), dtype=np.float32)
            else:
                preds = np.zeros((0, 21), dtype=np.int64)
            if len(file_ids) != expected_rows:
                log.warning(
                    f"Submission row count mismatch for {split_name}: expected={expected_rows} generated={len(file_ids)}"
                )

            if task == "a1":
                sub = pd.DataFrame({
                    "file_id": file_ids,
                    "p_D": preds[:, 0],
                    "p_A": preds[:, 1],
                    "p_S": preds[:, 2],
                })
            else:
                item_cols = [f"d{i:02d}" for i in range(1, 22)]
                sub = pd.DataFrame({"file_id": file_ids})
                for j, col in enumerate(item_cols):
                    sub[col] = preds[:, j]

            out_path = run_dirs["submissions"] / f"submission_{task}_{split_name}.csv"
            sub.to_csv(out_path, index=False)
            log.info(f"Wrote {len(sub)} rows to {out_path}")
    else:
        log.info("Skipping submission generation after training; use infer.py for release inference.")

    meta.finish("completed")
    log.info(f"Run complete: {run_name}")
    log.info(f"Output dir: {run_dirs['root']}")


if __name__ == "__main__":
    main()

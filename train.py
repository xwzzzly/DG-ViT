from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dgvit.config import load_config, save_config
from dgvit.data import RandomRasterPatchDataset, load_manifest, split_manifest
from dgvit.factory import build_model
from dgvit.losses import reconstruction_loss
from dgvit.priors import load_prior_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DG-ViT or its no-prior ViT ablation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--land-cover", required=True)
    parser.add_argument("--prior-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["dg_vit", "vit"], default="dg_vit")
    parser.add_argument("--lambda-prior", type=float, default=None)
    parser.add_argument(
        "--active-prior-classes",
        default="",
        help="Comma-separated CLCD codes; blank activates all classes.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(
    frame,
    config,
    land_cover,
    samples,
    augment,
    seed,
) -> RandomRasterPatchDataset:
    data_config = config["data"]
    return RandomRasterPatchDataset(
        frame,
        config["source_names"],
        land_cover,
        samples_per_epoch=samples,
        patch_size=int(config["model"]["image_size"]),
        valid_min_db=float(data_config["valid_min_db"]),
        valid_max_db=float(data_config["valid_max_db"]),
        nodata=float(data_config["nodata"]),
        seed=seed,
        augment=augment,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    mode,
    tv_weight,
    gradient_clip,
    training,
    active_prior_classes,
) -> dict[str, float]:
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "tv": 0.0}
    count = 0
    amp_enabled = device.type == "cuda"
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            sources = batch["sources"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            land_cover = batch["land_cover"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                fused, diagnostics = model(
                    sources,
                    masks,
                    land_cover,
                    mode=mode,
                    sample_dirichlet=training if mode == "dg_vit" else False,
                    active_prior_classes=active_prior_classes,
                    return_diagnostics=True,
                )
                reliability = (
                    diagnostics["pixel_prior"] if mode == "dg_vit" else None
                )
                if reliability is not None and active_prior_classes is not None:
                    activation = diagnostics["pixel_prior_activation"]
                    uniform = torch.full_like(
                        reliability,
                        1.0 / reliability.shape[1],
                    )
                    reliability = activation * reliability + (1.0 - activation) * uniform
                loss, components = reconstruction_loss(
                    fused,
                    sources,
                    masks,
                    source_weights=reliability,
                    tv_weight=tv_weight,
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            for key in totals:
                totals[key] += float(components[key].item())
            count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.lambda_prior is not None:
        config["model"]["lambda_prior"] = float(args.lambda_prior)
    seed = int(config["data"]["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    active_prior_classes = (
        tuple(
            int(code)
            for code in args.active_prior_classes.split(",")
            if code.strip()
        )
        or None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.json")

    source_names = config["source_names"]
    prior_weights = load_prior_table(args.prior_csv, source_names)
    manifest = load_manifest(args.manifest, source_names)
    train_frame, validation_frame = split_manifest(
        manifest,
        float(config["data"]["train_fraction"]),
    )
    train_dataset = make_dataset(
        train_frame,
        config,
        args.land_cover,
        int(config["data"]["samples_per_epoch"]),
        True,
        seed,
    )
    validation_dataset = make_dataset(
        validation_frame,
        config,
        args.land_cover,
        int(config["data"]["validation_samples"]),
        False,
        seed + 1_000_000,
    )
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = build_model(config, prior_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_validation = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_validation = float(state.get("best_validation", best_validation))

    log_path = output_dir / "training_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "learning_rate",
                    "train_loss",
                    "train_reconstruction",
                    "train_tv",
                    "validation_loss",
                    "validation_reconstruction",
                    "validation_tv",
                ]
            )

    patience = int(config["training"]["patience"])
    stale_epochs = 0
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.mode,
            float(config["training"]["tv_weight"]),
            float(config["training"]["gradient_clip"]),
            True,
            active_prior_classes,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            optimizer,
            scaler,
            device,
            args.mode,
            float(config["training"]["tv_weight"]),
            float(config["training"]["gradient_clip"]),
            False,
            active_prior_classes,
        )
        scheduler.step(validation_metrics["loss"])
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d} | "
            f"train={train_metrics['loss']:.6f} | "
            f"val={validation_metrics['loss']:.6f} | "
            f"lr={learning_rate:.2e}"
        )
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    epoch + 1,
                    learning_rate,
                    train_metrics["loss"],
                    train_metrics["reconstruction"],
                    train_metrics["tv"],
                    validation_metrics["loss"],
                    validation_metrics["reconstruction"],
                    validation_metrics["tv"],
                ]
            )

        checkpoint = {
            "epoch": epoch,
            "mode": args.mode,
            "best_validation": best_validation,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "source_names": source_names,
            "prior_weights": prior_weights,
            "active_prior_classes": active_prior_classes,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            checkpoint["best_validation"] = best_validation
            torch.save(checkpoint, output_dir / "best.pt")
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break


if __name__ == "__main__":
    main()

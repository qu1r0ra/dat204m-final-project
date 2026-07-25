"""PyTorch LSTM Sequence Classifier for Cryptocurrency Direction Prediction.

Implements SequenceDataset for symbol-isolated sliding window sequence creation,
LSTMClassifier architecture, training loop with early stopping, threshold tuning,
prediction utilities, and artifact serialization routines.
"""

import json
import logging
import time
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.config import FEATURE_COLS
from src.exceptions import DataValidationError
from src.models.evaluation import calculate_metrics, save_metrics_json
from src.utils.seed import get_provenance_metadata, set_seed

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """PyTorch Dataset that constructs symbol-isolated sequence windows.

    To prevent data leakage across different cryptocurrency assets, sequences
    never cross symbol boundaries.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str = "target",
        seq_len: int = 60,
        scaler: StandardScaler | None = None,
    ) -> None:
        """Initializes SequenceDataset with Polars DataFrame and parameters."""
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
        self.target_col = target_col
        self.seq_len = seq_len

        # Ensure input DataFrame is sorted chronologically per symbol
        df_sorted = df.sort(["symbol", "open_time"])

        # Extract features matrix
        raw_features = df_sorted.select(self.feature_cols).to_numpy()

        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(raw_features).astype(np.float32)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(raw_features).astype(np.float32)

        # Target values
        if self.target_col in df_sorted.columns:
            self.targets = df_sorted[self.target_col].to_numpy().astype(np.float32)
        else:
            self.targets = np.zeros(len(df_sorted), dtype=np.float32)

        # Build symbol boundary mask to guarantee no sequence crosses symbol boundaries
        symbols = df_sorted["symbol"].to_numpy()
        n_rows = len(df_sorted)

        if n_rows < self.seq_len:
            raise DataValidationError(
                f"Input DataFrame contains {n_rows} rows, which is less than "
                f"sequence length {self.seq_len}."
            )

        # Index i is a valid window end if i >= seq_len - 1 and the symbol
        # at index i matches the symbol at index i - seq_len + 1.
        valid_mask = symbols[self.seq_len - 1 :] == symbols[: n_rows - self.seq_len + 1]
        valid_end_indices = np.where(valid_mask)[0] + (self.seq_len - 1)

        self.valid_end_indices = valid_end_indices

    def __len__(self) -> int:
        return len(self.valid_end_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end_idx = self.valid_end_indices[idx]
        start_idx = end_idx - self.seq_len + 1

        x_seq = torch.from_numpy(self.features[start_idx : end_idx + 1])
        y_val = torch.tensor(self.targets[end_idx], dtype=torch.float32)

        return x_seq, y_val


class LSTMClassifier(nn.Module):
    """2-layer PyTorch LSTM Classifier for binary sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        logits = self.fc(last_step).squeeze(-1)  # (batch_size,)
        return logits


def _append_jsonl(filepath: Path, record: dict[str, Any]) -> None:
    """Appends a single JSON record as one line to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def train_lstm(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "target",
    seq_len: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 2048,
    max_epochs: int = 20,
    patience: int = 3,
    seed: int = 42,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    config_name: str = "default",
    log_filepath: Path | str | None = None,
) -> tuple[LSTMClassifier, StandardScaler, float, dict[str, Any]]:
    """Trains PyTorch LSTM Classifier with early stopping and post-hoc threshold tuning.

    Args:
        config_name: Human-readable name for this training configuration (e.g.
            ``"Config A (baseline)"``). Written into JSONL log records for
            multi-config sweep identification.
        log_filepath: Path to a JSONL file where per-epoch training metrics are
            appended. Each line is a self-contained JSON object with timestamp,
            elapsed time, losses, and validation metrics. ``None`` disables
            file logging.
    """
    set_seed(seed)
    features = feature_cols if feature_cols is not None else FEATURE_COLS

    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        selected_device = torch.device(device)
    else:
        selected_device = device

    log_path = Path(log_filepath) if log_filepath is not None else None
    training_start_time = time.time()

    logger.info(
        f"Training LSTM Classifier [{config_name}] on device: {selected_device} (seed={seed})"
    )

    # Build sequence datasets
    train_ds = SequenceDataset(
        train_df, feature_cols=features, target_col=target_col, seq_len=seq_len
    )
    scaler = train_ds.scaler
    val_ds = SequenceDataset(
        val_df,
        feature_cols=features,
        target_col=target_col,
        seq_len=seq_len,
        scaler=scaler,
    )

    use_cuda = selected_device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=num_workers,
    )

    model = LSTMClassifier(
        input_size=len(features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(selected_device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    stopping_reason = "max_epochs"
    history: dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_roc_auc": [],
        "val_balanced_acc": [],
    }

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Training pass
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"[{config_name}] Epoch {epoch}/{max_epochs} Train",
            leave=False,
            unit="batch",
        )
        for x_batch, y_batch in train_pbar:
            x_batch = x_batch.to(selected_device)
            y_batch = y_batch.to(selected_device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            train_count += len(y_batch)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_count if train_count > 0 else 0.0

        # Validation pass — collect all predictions for real metric computation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        epoch_val_probs: list[np.ndarray] = []
        epoch_val_targets: list[np.ndarray] = []

        val_pbar = tqdm(
            val_loader,
            desc=f"[{config_name}] Epoch {epoch}/{max_epochs} Val",
            leave=False,
            unit="batch",
        )
        with torch.no_grad():
            for x_batch, y_batch in val_pbar:
                x_batch = x_batch.to(selected_device)
                y_batch = y_batch.to(selected_device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss_sum += loss.item() * len(y_batch)
                probs = torch.sigmoid(logits)
                epoch_val_probs.append(probs.cpu().numpy())
                epoch_val_targets.append(y_batch.cpu().numpy())
                val_count += len(y_batch)

        val_loss = val_loss_sum / val_count if val_count > 0 else 0.0

        # Compute real validation metrics per epoch
        all_val_probs = np.concatenate(epoch_val_probs) if epoch_val_probs else np.array([])
        all_val_targets = np.concatenate(epoch_val_targets) if epoch_val_targets else np.array([])
        val_preds = (all_val_probs >= 0.5).astype(int)
        epoch_metrics = calculate_metrics(all_val_targets, val_preds, all_val_probs)

        val_acc = epoch_metrics["accuracy"]
        val_precision = epoch_metrics["precision"]
        val_recall = epoch_metrics["recall"]
        val_f1 = epoch_metrics["f1"]
        val_bal_acc = epoch_metrics["balanced_accuracy"]
        val_roc_auc = epoch_metrics.get("roc_auc", 0.5)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_precision"].append(float(val_precision))
        history["val_recall"].append(float(val_recall))
        history["val_f1"].append(float(val_f1))
        history["val_roc_auc"].append(float(val_roc_auc))
        history["val_balanced_acc"].append(float(val_bal_acc))

        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - training_start_time
        is_best = val_loss < best_val_loss

        logger.info(
            f"Epoch {epoch}/{max_epochs} - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | Prec: {val_precision:.4f} | "
            f"Rec: {val_recall:.4f} | F1: {val_f1:.4f} | "
            f"Bal Acc: {val_bal_acc:.4f} | AUC: {val_roc_auc:.4f} | "
            f"Epoch Time: {epoch_elapsed:.1f}s"
        )

        # Write JSONL log record
        if log_path is not None:
            from datetime import datetime

            log_record = {
                "config_name": config_name,
                "epoch": epoch,
                "max_epochs": max_epochs,
                "timestamp": datetime.now(UTC).isoformat(),
                "epoch_elapsed_sec": round(epoch_elapsed, 2),
                "total_elapsed_sec": round(total_elapsed, 2),
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "val_acc": round(float(val_acc), 6),
                "val_precision": round(float(val_precision), 6),
                "val_recall": round(float(val_recall), 6),
                "val_f1": round(float(val_f1), 6),
                "val_balanced_acc": round(float(val_bal_acc), 6),
                "val_roc_auc": round(float(val_roc_auc), 6),
                "is_best_epoch": is_best,
                "patience_counter": patience_counter + (0 if is_best else 1),
                "hparams": {
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "lr": lr,
                    "batch_size": batch_size,
                },
            }
            _append_jsonl(log_path, log_record)

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopping_reason = "early_stopping"
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Post-hoc threshold tuning on validation set
    model.eval()
    val_probs = np.zeros(len(val_ds), dtype=np.float32)
    val_targets = np.zeros(len(val_ds), dtype=np.float32)
    ptr = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            b = len(y_batch)
            x_batch = x_batch.to(selected_device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).ravel().cpu().numpy()
            val_probs[ptr : ptr + b] = probs
            val_targets[ptr : ptr + b] = y_batch.ravel().cpu().numpy()
            ptr += b

    val_probs = val_probs[:ptr]
    val_targets = val_targets[:ptr]

    best_threshold = 0.5
    best_bal_acc = -1.0
    threshold_grid = np.arange(0.40, 0.605, 0.005)
    threshold_history = []

    for thresh in threshold_grid:
        thresh_val = float(thresh)
        preds = (val_probs >= thresh_val).astype(int)
        t_metrics = calculate_metrics(val_targets, preds, val_probs)
        threshold_history.append(
            {
                "threshold": thresh_val,
                "accuracy": t_metrics["accuracy"],
                "precision": t_metrics["precision"],
                "recall": t_metrics["recall"],
                "f1": t_metrics["f1"],
                "balanced_accuracy": t_metrics["balanced_accuracy"],
            }
        )
        if t_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = t_metrics["balanced_accuracy"]
            best_threshold = thresh_val

    history["best_epoch"] = best_epoch
    history["epochs_trained"] = len(history["epoch"])
    history["stopping_reason"] = stopping_reason
    history["best_val_loss"] = float(best_val_loss)
    history["best_threshold"] = best_threshold
    history["best_val_balanced_acc"] = float(best_bal_acc)
    history["threshold_grid_search"] = threshold_history
    history["val_probs"] = val_probs
    history["val_targets"] = val_targets

    logger.info(
        f"Tuned Decision Threshold: {best_threshold:.3f} | Val Balanced Acc: {best_bal_acc:.4f}"
    )

    return model, scaler, best_threshold, history


def predict_lstm(
    model: LSTMClassifier,
    dataset: SequenceDataset,
    batch_size: int = 2048,
    num_workers: int = 0,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates prediction probabilities and default threshold predictions."""
    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        selected_device = torch.device(device)
    else:
        selected_device = device

    model.to(selected_device)
    model.eval()

    use_cuda = selected_device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=num_workers,
    )

    all_probs = []

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(selected_device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    y_probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([], dtype=np.float32)

    y_preds_default = (y_probs >= 0.5).astype(np.int32)
    return y_probs, y_preds_default


def save_lstm_artifacts(
    model: LSTMClassifier,
    scaler: StandardScaler,
    threshold: float,
    feature_cols: list[str],
    seq_len: int,
    hparams: dict[str, Any],
    filepath: Path | str,
    history: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    seed: int = 42,
) -> None:
    """Serializes PyTorch LSTM model state dict and associated metadata."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "hparams": hparams,
        "history": history if history is not None else {},
        "metrics": metrics if metrics is not None else {},
        "provenance": get_provenance_metadata(seed),
    }
    torch.save(artifacts, path)
    logger.info(f"Saved LSTM artifacts to {path}")

    # Export sidecar JSON metrics if provided
    if metrics is not None:
        metrics_json_path = path.with_name(f"{path.stem}_metrics.json")
        save_metrics_json(metrics, metrics_json_path)


def load_lstm_artifacts(
    filepath: Path | str,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Loads PyTorch LSTM model and metadata from serialized checkpoint."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found at {path}")

    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        selected_device = torch.device(device)
    else:
        selected_device = device

    # Note: weights_only=False is intentional because the checkpoint contains a
    # scikit-learn StandardScaler object alongside PyTorch model state dicts.
    artifacts = torch.load(path, map_location=selected_device, weights_only=False)

    feature_cols = artifacts["feature_cols"]
    hparams = artifacts.get("hparams", {})
    hidden_size = hparams.get("hidden_size", 64)
    num_layers = hparams.get("num_layers", 2)
    dropout = hparams.get("dropout", 0.3)

    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(artifacts["model_state_dict"])
    model.to(selected_device)
    model.eval()

    return {
        "model": model,
        "scaler": artifacts["scaler"],
        "threshold": artifacts["threshold"],
        "feature_cols": feature_cols,
        "seq_len": artifacts["seq_len"],
        "hparams": hparams,
    }

from __future__ import annotations
import torch
import numpy as np
import os
import polars as pl
from pathlib import Path
from typing import Iterable, List, Optional, Tuple,  Dict, Any
import sys
import json, datetime, platform
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from .preprocess import *

def pearson_corr_general(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr

def mean_pearson_correlation(loader, model, device, dim: int = 1) -> float:
    """Compute mean Pearson r across sequences (dim=1) or across targets (dim=0)."""
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yhat = model(xb)
            ys.append(yb)
            yhats.append(yhat)
    y = torch.cat(ys, dim=0)
    yhat = torch.cat(yhats, dim=0)

    y = y - y.mean(dim=dim, keepdim=True)
    yhat = yhat - yhat.mean(dim=dim, keepdim=True)
    num = (y * yhat).sum(dim=dim)
    den = torch.sqrt((y**2).sum(dim=dim) * (yhat**2).sum(dim=dim) + 1e-8)
    r = num / den
    return float(torch.nanmean(r).item())

def make_loaders_from_arrays(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    X_test: Optional[np.ndarray] = None,
    Y_test: Optional[np.ndarray] = None,
    batch_size: int = 64,
    val_frac: float = 0.1,
    use_test_as_val: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders from numpy arrays.
    - If use_test_as_val=True and X_test/Y_test are provided, the *test* arrays are used as validation.
    - Otherwise we split train into (train, val) by val_frac.
    """
    Xtr = torch.from_numpy(X_train).float()
    Ytr = torch.from_numpy(Y_train).float()

    if use_test_as_val and X_test is not None and Y_test is not None:
        Xv = torch.from_numpy(X_test).float()
        Yv = torch.from_numpy(Y_test).float()
        train_ds = TensorDataset(Xtr, Ytr)
        val_ds = TensorDataset(Xv, Yv)
    else:
        # carve validation out of training
        N = Xtr.shape[0]
        n_val = int(round(val_frac * N))
        idx = torch.randperm(N)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        train_ds = TensorDataset(Xtr[tr_idx], Ytr[tr_idx])
        val_ds   = TensorDataset(Xtr[val_idx], Ytr[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader

def save_training_metadata(
    out_dir: str | Path,
    split_meta: dict,
    model_hparams: dict,
    train_hparams: dict,
    history: dict,
    best_model_path: str,
):
    """
    Save a merged metadata file next to the model checkpoint.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "system": {"python": platform.python_version(), "torch": torch.__version__},
        "data_split": split_meta,             # copied from the split folder's meta.json
        "model_hparams": model_hparams,       # e.g., sequence_length, n_targets, architecture flags
        "training_hparams": train_hparams,    # e.g., batch_size, lr, weight_decay, loss, patience, epochs
        "history": history,                   # losses and correlations per epoch
        "best_model_path": str(best_model_path),
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(payload, f, indent=2)



def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
    early_stopping_patience: int,
    device: torch.device,
    output_dir: str | Path,
    model_name: str,
    sequence_length: int,
):
    """
    Standard train/validate loop with early stopping and pearson metrics.
    Returns: (history dict, best_model_path)
    """
    import torch, os
    best_corr_sum = -float('inf')
    patience_counter = 0
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_corr_seq": [], "val_corr_type": []
    }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{model_name}_{sequence_length}bp_.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_running += criterion(model(xb), yb).item()
        val_loss = val_running / max(1, len(val_loader))

        corr_seq  = mean_pearson_correlation(val_loader, model, device, dim=1)
        corr_type = mean_pearson_correlation(val_loader, model, device, dim=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_corr_seq"].append(corr_seq)
        history["val_corr_type"].append(corr_type)

        print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"r_seq {corr_seq:.4f} | r_type {corr_type:.4f}")

        corr_sum = corr_seq + corr_type
        if corr_sum > best_corr_sum:
            best_corr_sum = corr_sum
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping.")
                break

    return history, str(best_path)



META_COLS: set[str] = {
    "peak_id", "Peak", "chromosome", "end", "start", "sequence", "dataset"
}


def choose_holdout_chroms(chroms: np.ndarray, test_size: float,
                          seed: int = 42):
    """
    Pick a deterministic set of chromosomes whose row count sums to ~test_size of total.
    Returns a list of chromosome labels (as strings).
    """
    uniq, counts = np.unique(chroms, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    total = len(chroms)
    chosen, acc = [], 0
    for i in order:
        chosen.append(str(uniq[i]))
        acc += counts[i]
        if acc / total >= test_size:
            break
    return chosen
import polars as pl
def _build_output_root(base_name: str, output_dir: str | Path,
                       test_size: float, split_mode: str) -> Path:
    train_pct = int(round((1 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    mode_lbl = "chrom_split" if split_mode == "chromosome" else "random"
    root = Path(output_dir) / base_name / f"{train_pct}_{test_pct}_{mode_lbl}"
    root.mkdir(parents=True, exist_ok=True)
    return root

def target_cols_from_schema(schema: dict[str, pl.DataType],
                            exclude: Iterable[str] = META_COLS) -> List[str]:
    """All numeric columns except metadata columns."""
    excl = set(exclude)
    return [c for c, dt in schema.items() if c not in excl and _is_numeric(dt)]

from typing import Optional, List

# Reuse helpers from your ATAC pipeline
# (assumes you already have META_COLS, one_hot_batch, _build_output_root, target_cols_from_schema, _json_default, choose_holdout_chroms)
def prepare_rna_training_from_csv(
    csv_path: str | Path,
    pattern: str,
    output_dir: str | Path = "../../data/rna_training/",
    test_size: float = 0.2,
    random_state: int = 42,
    channel_first: bool = True,
    split_mode: str = "random",                 # "random" | "chromosome"
    holdout_chroms: Optional[List[str]] = None
) -> Path:
    """
    Prepare training arrays (X/Y) and metadata for RNA data from a CSV.

    - CSV must have a 'sequence' column and numeric expression columns.
    - Splitting:
        * random: row-wise random split.
        * chromosome: hold out selected chromosomes.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "sequence" not in df.columns:
        raise ValueError("CSV must contain a 'sequence' column.")

    base_name = pattern.replace("DATASET", "rna")
    root = _build_output_root(base_name, output_dir, test_size, split_mode)

    # Determine targets: all numeric columns except metadata, start, end, chrom
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    targets = [c for c in numeric_cols if c not in list(META_COLS) + ["start", "end"]]
    if not targets:
        raise ValueError("No numeric target columns found in RNA CSV.")

    seqs = df["sequence"].astype(str).tolist()
    Y = df[targets].to_numpy(dtype=np.float32)
    N = len(seqs)

    rng = np.random.default_rng(random_state)

    # Split indices
    if split_mode == "chromosome":
        if "chrom" not in df.columns:
            raise ValueError("split_mode='chromosome' requires a 'chrom' column in CSV.")
        chroms = df["chrom"].astype(str).to_numpy()
        if holdout_chroms is None:
            holdout_chroms = choose_holdout_chroms(chroms, test_size, random_state)
        test_mask = np.isin(chroms, holdout_chroms)
        train_mask = ~test_mask
        train_idx = np.nonzero(train_mask)[0]
        test_idx = np.nonzero(test_mask)[0]

        # Top up if test size undersampled
        need = int(round(test_size * N)) - len(test_idx)
        if need > 0 and len(train_idx) > 0:
            extra = rng.choice(train_idx, size=min(need, len(train_idx)), replace=False)
            test_idx = np.sort(np.concatenate([test_idx, extra]))
            train_idx = np.array(sorted(set(train_idx) - set(extra)))
    else:
        perm = rng.permutation(N)
        n_test = int(round(test_size * N))
        test_idx, train_idx = np.sort(perm[:n_test]), np.sort(perm[n_test:])

    # Slice and encode
    seq_train = [seqs[i] for i in train_idx]
    seq_test = [seqs[i] for i in test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    X_train = one_hot_batch(seqs=seq_train, channel_first=channel_first)
    X_test = one_hot_batch(seqs=seq_test, channel_first=channel_first)

    # Save arrays
    np.save(root / "X_train.npy", X_train)
    np.save(root / "Y_train.npy", Y_train)
    np.save(root / "X_test.npy", X_test)
    np.save(root / "Y_test.npy", Y_test)

    # Save targets & meta
    (root / "target_columns.txt").write_text("\n".join(targets))
    seq_len = int(df["sequence"].astype(str).str.len().max())  

    meta = {
        "base_name": base_name,
        "N": int(N),
        "seq_len": seq_len,
        "n_targets": int(len(targets)),
        "split_mode": split_mode,
        "holdout_chromosomes": [str(c) for c in (holdout_chroms or [])]
            if split_mode == "chromosome" else None,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "train_pct": int(round((1 - test_size) * 100)),
        "test_pct": int(round(test_size * 100)),
        "channel_first": bool(channel_first),
        "targets": targets,
        "random_state": int(random_state),
    }
    with open(root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return root

def prepare_training_from_df(
    df: pl.DataFrame,
    pattern: str,
    output_dir: str | Path = "../../data/training/",
    test_size: float = 0.2,
    random_state: int = 42,
    channel_first: bool = True,
    split_mode: str = "random",                 # "random" | "chromosome"
    holdout_chroms: Optional[List[str]] = None  # used when split_mode="chromosome"
) -> Path:
    """
    From a merged Polars DataFrame (embryo2life), produce X/Y numpy arrays and metadata.

    - Targets = all numeric non-metadata columns.
    - Inputs  = one-hot encodings of 'sequence'.
    - Splitting:
        * random: standard random split by rows.
        * chromosome: hold out all rows from selected chromosomes (no leakage).
          If holdout_chroms is None, chromosomes are auto-selected to cover ~test_size.

    Folder layout:
        output_dir/
          <base_name>/
            <trainpct>_<testpct>_<mode>/
              X_train.npy, Y_train.npy, X_test.npy, Y_test.npy,
              target_columns.txt, meta.json

    Returns:
        Path to the split subfolder.
    """
    if "sequence" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'sequence' column.")
    base_name = pattern.replace("DATASET", "merged")
    root = _build_output_root(base_name, output_dir, test_size, split_mode)

    # Determine targets and arrays
    targets = target_cols_from_schema(df.schema, exclude=META_COLS)
    if not targets:
        raise ValueError("No numeric target columns found.")
    seqs = df.get_column("sequence").to_list()
    Y = df.select(targets).to_numpy().astype(np.float32)

    N = len(seqs)
    rng = np.random.default_rng(random_state)

    # Split indices
    if split_mode == "chromosome":
        if "chromosome" not in df.columns:
            raise ValueError("split_mode='chromosome' requires a 'chromosome' column.")
        chroms = df.get_column("chromosome").to_numpy()
        if holdout_chroms is None:
            holdout_chroms = choose_holdout_chroms(chroms, test_size, random_state)
        test_mask = np.isin(chroms, holdout_chroms)
        train_mask = ~test_mask
        train_idx = np.nonzero(train_mask)[0]
        test_idx = np.nonzero(test_mask)[0]

        # If under target size, top-up randomly from remaining train rows
        need = int(round(test_size * N)) - len(test_idx)
        if need > 0 and len(train_idx) > 0:
            extra = rng.choice(train_idx, size=min(need, len(train_idx)), replace=False)
            test_idx = np.sort(np.concatenate([test_idx, extra]))
            train_idx = np.array(sorted(set(train_idx) - set(extra)))
    else:
        perm = rng.permutation(N)
        n_test = int(round(test_size * N))
        test_idx, train_idx = np.sort(perm[:n_test]), np.sort(perm[n_test:])

    # Slice & encode
    seq_train = [seqs[i] for i in train_idx]
    seq_test = [seqs[i] for i in test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    X_train = one_hot_batch(seqs=seq_train, channel_first=channel_first)
    X_test = one_hot_batch(seqs=seq_test, channel_first=channel_first)

    # Save arrays
    np.save(root / "X_train.npy", X_train)
    np.save(root / "Y_train.npy", Y_train)
    np.save(root / "X_test.npy", X_test)
    np.save(root / "Y_test.npy", Y_test)

    # Save targets & meta
    (root / "target_columns.txt").write_text("\n".join(targets))
    meta = {
        "base_name": base_name,
        "N": int(N),
        "seq_len": int(X_train.shape[2] if channel_first else X_train.shape[1]),
        "n_targets": int(len(targets)),
        "split_mode": split_mode,
        "holdout_chromosomes": [str(c) for c in (holdout_chroms or [])]
            if split_mode == "chromosome" else None,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "train_pct": int(round((1 - test_size) * 100)),
        "test_pct": int(round(test_size * 100)),
        "channel_first": bool(channel_first),
        "targets": targets,
        "random_state": int(random_state),
    }
    with open(root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)

    return root

def _json_default(obj):
    """Make numpy/polars scalars JSON-serializable for json.dump(default=_json_default)."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _is_numeric(dt: pl.DataType) -> bool:
    """Return True if a Polars dtype is integer or float (works across Polars versions)."""
    return any(isinstance(dt, t) for t in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ))

def load_split_folder(split_dir: str | Path) -> dict[str, np.ndarray | list[str] | dict]:
    """
    Convenience loader for a produced split folder.
    Returns a dict with arrays and metadata.
    """
    split_dir = Path(split_dir)
    X_train = np.load(split_dir / "X_train.npy", mmap_mode=None)
    Y_train = np.load(split_dir / "Y_train.npy", mmap_mode=None)
    X_test = np.load(split_dir / "X_test.npy", mmap_mode=None)
    Y_test = np.load(split_dir / "Y_test.npy", mmap_mode=None)
    targets = (split_dir / "target_columns.txt").read_text().strip().splitlines()
    meta = json.loads((split_dir / "meta.json").read_text())
    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_test": X_test, "Y_test": Y_test,
        "targets": targets, "meta": meta,
    }

import os, json, torch, numpy as np
from pathlib import Path

from zebra_dev.models import *


def plot_history(
    history: dict,
    out_dir: str | Path,
    *,
    dataset: str,
    model_name: str,
    run_id: str,
    train_hparams: dict,
):
    """
    Save two plots:
      - losses.png  : Train vs Val loss per epoch
      - correlations.png : r_seq and r_type per epoch
    Titles include dataset, model, run_id, and key hyperparams.
    Also writes history.csv for quick inspection.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # history → DataFrame for convenience
    df = pd.DataFrame(history)
    df.index.name = "epoch"
    df.to_csv(out_dir / "history.csv")

    # Build a compact title suffix with key hparams
    hp = train_hparams
    hp_str = (f"loss={hp.get('loss','')}, bs={hp.get('batch_size','')}, "
              f"lr={hp.get('lr','')}, wd={hp.get('weight_decay','')}, "
              f"epochs={hp.get('num_epochs','')}, patience={hp.get('early_stopping_patience','')}, "
              f"{hp.get('val_from','')}")

    title_prefix = f"{dataset} • {model_name}\n{run_id}\n{hp_str}"

    # --- Loss plot ---
    plt.figure(figsize=(8,5))
    plt.plot(df["train_loss"], label="Train loss")
    plt.plot(df["val_loss"],   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}\nLoss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "losses.png", dpi=150)
    plt.close()

    # --- Correlation plot ---
    if "val_corr_seq" in df.columns and "val_corr_type" in df.columns:
        plt.figure(figsize=(8,5))
        plt.plot(df["val_corr_seq"],  label="r (per-sequence)")
        plt.plot(df["val_corr_type"], label="r (per-celltype)")
        plt.xlabel("Epoch")
        plt.ylabel("Pearson r")
        plt.title(f"{title_prefix}\nValidation correlations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "correlations.png", dpi=150)
        plt.close()

from pathlib import Path
import torch
from pathlib import Path
import json, hashlib, datetime
import torch

def _fmt_float(x: float) -> str:
    # nice, stable float formatting for ids (1e-3, 5e-4, 0.001)
    s = f"{x:.1e}"
    # strip trailing zeros in mantissa (e.g., 1.0e-03 -> 1e-03)
    s = s.replace("0e", "e").replace(".e", "e")
    return s

def _now_stamp() -> str:
    # UTC timestamp for sorting; avoids collisions
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _short_hash(payload: dict, n: int = 8) -> str:
    # stable hash over the full metadata/hparams to prevent collisions
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]

def _find_repo_root(start: Path | None = None, max_up: int = 6) -> Path | None:
    """
    Walk upward looking for a project root indicator (pyproject.toml or .git).
    Returns the path if found, else None.
    """
    here = (start or Path.cwd()).resolve()
    for _ in range(max_up):
        if (here / "pyproject.toml").exists() or (here / ".git").exists():
            return here
        if here.parent == here:
            break
        here = here.parent
    return None

def _default_out_root() -> Path:
    """
    Priority:
      1) ZEBRA_DEV_OUTPUTS env var
      2) <repo>/runs (if repo root can be found)
      3) ~/.zebra_dev/runs
    """
    env = os.getenv("ZEBRA_DEV_OUTPUTS")
    if env:
        return Path(env).expanduser().resolve()

    repo = _find_repo_root()
    if repo is not None:
        return (repo / "runs").resolve()

    # site-packages / unknown context
    return (Path.home() / ".zebra_dev" / "runs").resolve()

def train_on_split(
    split_dir: str | Path,
    dataset: str,
    *,
    model_name: str = "dilated_baseline_model",
    loss_name: str = "mse",
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_test_as_val: bool = True,
    val_frac: float = 0.1,
    num_epochs: int = 300,
    early_stopping_patience: int = 10,
    out_root: str | Path | None = None,   # via _default_out_root()
):
    """
    CWD-agnostic training that writes to:
        <repo>/src/models/outputs/<dataset>/<model_name>/<run_id>/
    with a compact, unique run_id that includes split, loss, hparams, and a timestamp+hash.
    """
    # ---------- resolve repo & IO roots ----------
    split_dir = Path(split_dir).expanduser()
    if not split_dir.is_absolute():
        # anchor relative paths to detected repo root (best effort), else CWD
        repo = _find_repo_root()
        anchor = repo if repo is not None else Path.cwd()
        split_dir = (anchor / split_dir).resolve()


    if out_root is None:
        out_root = _default_out_root()
    else:
        out_root = Path(out_root).expanduser()
        if not out_root.is_absolute():
            repo = _find_repo_root()
            anchor = repo if repo is not None else Path.cwd()
            out_root = (anchor / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------- load data split ----------
    bundle = load_split_folder(split_dir)
    X_train, Y_train = bundle["X_train"], bundle["Y_train"]
    X_test,  Y_test  = bundle["X_test"],  bundle["Y_test"]
    split_meta = bundle["meta"]

    n_targets = Y_train.shape[1]
    seq_len   = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[-1]

    # ---------- data loaders ----------
    train_loader, val_loader = make_loaders_from_arrays(
        X_train, Y_train,
        X_test=X_test, Y_test=Y_test,
        batch_size=batch_size,
        use_test_as_val=use_test_as_val,
        val_frac=val_frac,
    )

    # ---------- model + loss ----------
    model = ATACSeqCNN(sequence_length=seq_len, num_targets=n_targets)
    if loss_name.lower() == "mse":
        criterion = torch.nn.MSELoss()
        loss_tag = "mse"
    elif loss_name.lower() == "poisson_log":
        criterion = torch.nn.PoissonNLLLoss(log_input=True)  # model outputs log-rate
        loss_tag = "poisslog"
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------- build a compact, unique run_id ----------
    split_mode = split_meta.get("split_mode", "random")
    train_pct  = split_meta.get("train_pct")
    test_pct   = split_meta.get("test_pct")
    split_tag  = f"{train_pct}_{test_pct}_{'chrom' if split_mode=='chromosome' else 'random'}"

    val_tag = "val=test" if use_test_as_val else f"val={int(round(val_frac*100))}%"
    id_core = {
        "dataset": dataset,
        "model_name": model_name,
        "split": split_tag,
        "loss": loss_tag,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_from": val_tag,
        "num_epochs": num_epochs,
        "patience": early_stopping_patience,
        "seq_len": int(seq_len),
        "n_targets": int(n_targets),
        # input lineage:
        "split_dir": str(split_dir),
        "meta_random_state": split_meta.get("random_state"),
        "holdout_chromosomes": split_meta.get("holdout_chromosomes"),
        "targets": split_meta.get("targets"),
    }
    h = _short_hash(id_core, n=8)
    stamp = _now_stamp()
    run_id = f"{split_tag}__{loss_tag}__bs{batch_size}_lr{_fmt_float(lr)}_wd{_fmt_float(weight_decay)}__{val_tag}__{stamp}_{h}"

    # ---------- output dir ----------
    out_dir = out_root / dataset / model_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

     # ---------- train ----------
    history, best_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        device=device,
        output_dir=out_dir,
        model_name=model_name,
        sequence_length=seq_len,
    )

    # ---- save best model with unique identifier
    short_id = h[:4]  # first 4 chars of the run hash
    ckpt_name = f"{model_name}_bs{batch_size}_lr{_fmt_float(lr)}_{loss_tag}_{short_id}.pth"
    final_best = out_dir / ckpt_name
    import shutil
    shutil.copy2(best_path, final_best)

    # ---------- metadata ----------
    model_hparams = {"sequence_length": int(seq_len), "num_targets": int(n_targets), "arch": model_name}
    train_hparams = {
        "batch_size": int(batch_size),
        "optimizer": "AdamW",
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "num_epochs": int(num_epochs),
        "early_stopping_patience": int(early_stopping_patience),
        "loss": "MSELoss" if loss_tag=="mse" else "PoissonNLLLoss(log_input=True)",
        "val_from": "test_set" if use_test_as_val else f"train_split(val_frac={val_frac})",
    }
    save_training_metadata(
        out_dir=out_dir,
        split_meta=split_meta,
        model_hparams=model_hparams,
        train_hparams=train_hparams,
        history=history,
        best_model_path=str(final_best),
    )

    # ---------- plots + history.csv ----------
    plot_history(
        history=history,
        out_dir=out_dir,
        dataset=dataset,
        model_name=model_name,
        run_id=run_id,
        train_hparams=train_hparams,
    )

    print(f"[✓] Saved to: {out_dir}")
    return out_dir

### __________________________________________________________________
### __________________________________________________________________
###             Combine ATAC + RNA
### __________________________________________________________________
### A) Expand sequence tensor of TSS flanking regions (RNA) with ATAC


# ---------- helpers ----------

def _one_hot_seq(seq: str, channel_first: bool = True) -> np.ndarray:
    """Return 4×L (or L×4) float32 one-hot for A,C,G,T; others -> all zeros."""
    seq = (seq or "").upper()
    L = len(seq)
    arr = np.zeros((4, L), dtype=np.float32)
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, b in enumerate(seq):
        j = m.get(b)
        if j is not None:
            arr[j, i] = 1.0
    return arr if channel_first else arr.T

def _parse_peak_series(peak: str) -> Tuple[str, int, int]:
    """
    Parse Peak strings like 'chr4:1234-5678' or '4:1234-5678'.
    Returns (chrom, start, end), 1-based inclusive.
    """
    s = str(peak).strip().strip('"').strip("'")
    m = re.search(r"(?:chr)?([0-9A-Za-z]+):(\d+)-(\d+)", s)
    if not m:
        raise ValueError(f"Unparsable peak: {peak}")
    chrom = m.group(1)
    start = int(m.group(2))
    end   = int(m.group(3))
    if end < start:
        start, end = end, start
    return chrom, start, end

def _detect_numeric_targets(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    exclude = set(exclude)
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in exclude]

# ----------------------------------------
# main constructor
# ----------------------------------------

def build_seq_atac_tensors(
    *,
    rna_df: pd.DataFrame,                  # must contain: chrom, start, end, strand, sequence, and RNA targets
    atac_df: pd.DataFrame,                 # must contain: Peak + K pseudobulk columns
    atac_peak_col: str = "Peak",
    atac_cols: Optional[List[str]] = None, # if None -> all numeric in atac_df
    expr_cols: Optional[List[str]] = None, # if None -> all numeric in rna_df (excluding coords)
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    strand_col: str = "strand",
    # How the TSS window for each row is computed:
    #   If before_bp+after_bp (or symmetric_bp) > 0:   region = [TSS-before, TSS+after-1]
    #   else:                                          region = [start, end]
    symmetric_bp: int = 0,
    before_bp: int = 0,
    after_bp: int = 0,
    use_seq_len_to_infer_span: bool = True,
    # aggregation if multiple peaks cover the same base:
    agg: str = "max",                      # "max" or "sum"
    channel_first: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build X and Y arrays:

      X shape: (N, C, L) if channel_first else (N, L, C)
               where C = K (ATAC pseudobulks) + 4 (one-hot DNA)
      Y shape: (N, T) where T = len(expr_cols)

    Assumptions:
      - rna_df['sequence'] is the exact DNA window used (TSS window or gene span).
      - atac_df[atac_peak_col] are 1-based, inclusive intervals. Values in `atac_cols`
        are the per-peak signals per pseudobulk (broadcast across the peak span).
      - Chromosomes: we compare 'chrom' in rna_df to the parsed peak chrom
        with 'chr' removed (i.e., 'chr4' and '4' are matched).
    """
    # resolve expansion parameters
    if symmetric_bp and (before_bp or after_bp):
        raise ValueError("Provide either symmetric_bp or (before_bp/after_bp), not both.")
    if symmetric_bp > 0:
        before_bp = after_bp = symmetric_bp

    # choose columns
    if atac_cols is None:
        atac_cols = atac_df.select_dtypes(include=[np.number]).columns.tolist()
    if not atac_cols:
        raise ValueError("No numeric ATAC columns (pseudobulks) found.")

    meta_cols_rna = {chrom_col, start_col, end_col, strand_col, "sequence"}
    if expr_cols is None:
        expr_cols = _detect_numeric_targets(rna_df, exclude=meta_cols_rna)
    if not expr_cols:
        raise ValueError("No numeric expression target columns found in RNA dataframe.")

    # ----- preprocess ATAC peaks per chromosome -----
    atac_df = atac_df.copy()
    parsed = atac_df[atac_peak_col].apply(_parse_peak_series)
    atac_df["_chrom"] = parsed.apply(lambda t: str(t[0]).replace("chr", "").strip())
    atac_df["_start"] = parsed.apply(lambda t: int(t[1]))
    atac_df["_end"]   = parsed.apply(lambda t: int(t[2]))
    atac_df.sort_values(["_chrom", "_start", "_end"], inplace=True)

    # dictionary: chrom -> array of (start,end,row_index)
    chrom2rows: Dict[str, np.ndarray] = {}
    for chrom, sub in atac_df.groupby("_chrom", sort=False):
        chrom2rows[chrom] = sub[["_start", "_end"]].to_numpy(dtype=np.int64)
    # keep a view to the counts matrix
    atac_counts = atac_df[atac_cols].to_numpy(dtype=np.float32)
    atac_row_offsets = (atac_df.reset_index().index.to_numpy())

    # ----- iterate genes and construct tensors -----
    N = rna_df.shape[0]
    # L differs across genes; enforce constant L by trusting sequence length.
    seq_lens = rna_df["sequence"].astype(str).str.len().to_numpy()
    if not np.all(seq_lens == seq_lens[0]):
        # variable-length windows -> you can bucket or pad; for now enforce equal
        raise ValueError("Sequences have varying lengths. Generate a fixed window (e.g., symmetric_bp) first.")
    L = int(seq_lens[0])

    K = len(atac_cols)
    C = K + 4

    # allocate
    if channel_first:
        X = np.zeros((N, C, L), dtype=np.float32)
    else:
        X = np.zeros((N, L, C), dtype=np.float32)
    Y = rna_df[expr_cols].to_numpy(dtype=np.float32)

    # fast grouping by chromosome
    for idx, row in rna_df.iterrows():
        # --- region window for this gene ---
        chrom = str(row[chrom_col]).strip()
        s = int(row[start_col]); e = int(row[end_col])
        strand = str(row.get(strand_col, "+")).strip()
        seq = str(row["sequence"])
        L_here = len(seq)
        if L_here != L:
            raise ValueError("Sequences must be equal length across rows.")

        if before_bp > 0 or after_bp > 0:
            # TSS-centered window
            tss = s if strand == "+" else e
            region_start = tss - before_bp
            region_end   = tss + after_bp - 1  # 1-based inclusive
            if use_seq_len_to_infer_span:
                # trust sequence length for the end (handles off-by-ones)
                region_end = region_start + L_here - 1
        else:
            # full gene span
            region_start = s
            region_end   = e
            if region_end < region_start:
                region_start, region_end = region_end, region_start
            if use_seq_len_to_infer_span and L_here > 0:
                # make end consistent with the sequence length
                region_end = region_start + L_here - 1

        # One-hot DNA (always last 4 channels)
        DNA = _one_hot_seq(seq, channel_first=True)  # 4×L
        if channel_first:
            X[idx, K:K+4, :] = DNA
        else:
            X[idx, :, K:K+4] = DNA.T

        # ATAC fill (first K channels)
        # find ATAC peaks in the same chromosome
        chr_key = chrom.replace("chr", "")
        if chr_key not in chrom2rows:
            # no peaks for this chrom
            continue

        peaks_arr = chrom2rows[chr_key]           # shape (M,2)
        # binary search to find first peak ending after region_start
        # (small, readable two-pointer scan works well too)
        # We'll do a coarse scan: select candidates that start <= region_end and end >= region_start
        # using vectorized boolean mask on peaks_arr (it's fast enough)
        mask = (peaks_arr[:, 0] <= region_end) & (peaks_arr[:, 1] >= region_start)
        if not mask.any():
            continue

        cand_idx = np.nonzero(mask)[0]
        # Map to original rows to fetch counts
        # We need the same selection on atac_df order
        # atac_row_offsets aligns with peaks_arr order within each chrom group after groupby?
        # Simpler: recompute mask on the grouped dataframe 'sub'
        # We'll recover the sub-slice directly from atac_df using index matching:

        # construct a boolean mask over atac_df for this chrom & overlap
        sub_mask = (atac_df["_chrom"] == chr_key) & (atac_df["_start"] <= region_end) & (atac_df["_end"] >= region_start)
        if not sub_mask.any():
            continue

        sub_rows = atac_df.loc[sub_mask, ["_start", "_end"]].to_numpy(dtype=np.int64)
        sub_counts = atac_df.loc[sub_mask, atac_cols].to_numpy(dtype=np.float32)

        # fill each overlapping peak
        for (p_start, p_end), counts in zip(sub_rows, sub_counts):
            # overlapping span in 1-based inclusive
            ovl_start = max(region_start, p_start)
            ovl_end   = min(region_end,   p_end)
            if ovl_end < ovl_start:
                continue
            # convert to 0-based half-open indices in the gene window
            left  = int(ovl_start - region_start)      # inclusive
            right = int(ovl_end   - region_start + 1)  # exclusive

            if channel_first:
                if agg == "max":
                    X_slice = X[idx, 0:K, left:right]
                    X[idx, 0:K, left:right] = np.maximum(X_slice, counts[:, None])
                else:  # sum
                    X[idx, 0:K, left:right] += counts[:, None]
            else:
                if agg == "max":
                    X_slice = X[idx, left:right, 0:K]
                    X[idx, left:right, 0:K] = np.maximum(X_slice, counts[None, :])
                else:
                    X[idx, left:right, 0:K] += counts[None, :]

    meta = {
        "n_genes": int(N),
        "seq_len": int(L),
        "n_atac_channels": int(K),
        "dna_channels": 4,
        "channels_first": bool(channel_first),
        "atac_cols": atac_cols,
        "expr_cols": expr_cols,
        "agg": agg,
        "window_mode": "tss" if (before_bp > 0 or after_bp > 0) else "gene_span",
        "before_bp": int(before_bp),
        "after_bp": int(after_bp),
    }
    return X, Y, meta

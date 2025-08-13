import torch
import numpy as np
import os

def pearson_corr_general(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr

def mean_pearson_correlation(loader, model, device, dim=1):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    if dim == 0:
        return np.mean([
            pearson_corr_general(all_preds[:, i], all_targets[:, i]).item()
            for i in range(all_preds.shape[1])
            if torch.std(all_targets[:, i]) > 0
        ])
    else:
        return np.mean([
            pearson_corr_general(all_preds[i], all_targets[i]).item()
            for i in range(all_preds.shape[0])
            if torch.std(all_targets[i]) > 0
        ])

def train_model(
    model, train_loader, val_loader, criterion, optimizer,
    num_epochs, early_stopping_patience, device, output_dir, model_name, sequence_length):
    best_corr_sum = -float('inf')
    patience_counter = 0
    train_losses, val_losses, val_corrs_seq, val_corrs_type = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_X), batch_y).item()
        val_loss /= len(val_loader)

        corr_seq = mean_pearson_correlation(val_loader, model, device, dim=1)
        corr_type = mean_pearson_correlation(val_loader, model, device, dim=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_corrs_seq.append(corr_seq)
        val_corrs_type.append(corr_type)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Corr (Seq): {corr_seq:.4f} - Corr (Type): {corr_type:.4f}")
        corr_sum = corr_seq + corr_type

        if corr_sum > best_corr_sum:
            best_corr_sum = corr_sum
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_{sequence_length}bp_best.pth"))
            print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses, val_corrs_seq, val_corrs_type



def choose_holdout_chroms(chroms: np.ndarray, test_size: float,
                          seed: int = 42) -> List[str]:
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
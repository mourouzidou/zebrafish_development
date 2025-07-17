import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TORCH_JIT_DISABLE_ONEDNN_FUSION"] = "1"

def one_hot_encode(sequence):
    # Convert sequence to one-hot encoding with N handling
    seq_array = np.array(list(sequence))[:, None]
    bases = np.array(list('ACGT'))
    one_hot = (seq_array == bases).astype(np.float32)
    
    # Handle N bases by setting equal probability for all bases
    n_mask = seq_array.flatten() == 'N'
    one_hot[n_mask] = 0.25
    
    return one_hot

def reverse_complement_tensor(x):
    rev_x = torch.flip(x, dims=[2])
    rev_x = rev_x[:, [3, 2, 1, 0], :]
    return rev_x

# ---------------- Model Definition ---------------- #
class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2*dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x))) + x

class ATACSeqCNN(nn.Module):
    def __init__(self, sequence_length, num_targets):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=19, padding=9),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
        )

        self.dilated_blocks = nn.Sequential(
            DilatedBlock(256, 1),
            DilatedBlock(256, 2),
            DilatedBlock(256, 4),
            DilatedBlock(256, 8)
        )

        self.conv_tail = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

        fc_input_dim = 256 * (sequence_length // (2**5))
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_targets),
            nn.Softplus()
        )

    def forward(self, x, reverse=False):
        if reverse:
            x = reverse_complement_tensor(x)
        x = self.stem(x)
        x = self.dilated_blocks(x)
        x = self.conv_tail(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)

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
    # if dim == 0:
    #     return np.mean([pearson_corr_general(all_preds[:, i], all_targets[:, i], dim=0).item() for i in range(all_preds.shape[1])])
    # else:
    #     return np.mean([pearson_corr_general(all_preds[i], all_targets[i], dim=0).item() for i in range(all_preds.shape[0])])

    if dim == 0:
        return np.mean([pearson_corr_general(all_preds[:, i], all_targets[:, i]).item() for i in range(all_preds.shape[1])])
    else:
        return np.mean([pearson_corr_general(all_preds[i], all_targets[i]).item() for i in range(all_preds.shape[0])])

def load_and_split_data(csv_path, sequence_length):
    df = pd.read_csv(csv_path)
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    y = df.iloc[:, 0:-4].to_numpy(dtype=np.float32)
    X = np.stack([one_hot_encode(seq) for seq in df['sequence']])
    X_tensor = torch.tensor(X).permute(0, 2, 1)
    y_tensor = torch.tensor(y)
    chroms = df['chromosome'].unique()
    np.random.shuffle(chroms)
    split_idx = int(len(chroms) * 0.8)
    train_mask = df['chromosome'].isin(chroms[:split_idx]).values
    val_mask = df['chromosome'].isin(chroms[split_idx:]).values
    return X_tensor[train_mask], X_tensor[val_mask], y_tensor[train_mask], y_tensor[val_mask]

# ---------------- Training Loop ---------------- #
if __name__ == "__main__":
    sequence_length = 2000
    model_name = "dilated_baseline_model_raw"
    dataset_path = f"data/embryo/processed/atac_peaks_with_sequences_{sequence_length}.csv"
    output_dir = f"src/models/outputs/{model_name}_{sequence_length}_raw"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    num_targets = df.shape[1] - 4
    model = ATACSeqCNN(sequence_length, num_targets)

    X_train, X_val, y_train, y_val = load_and_split_data(dataset_path, sequence_length)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Get initial validation metrics before training
    model.eval()
    initial_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            initial_val_loss += criterion(model(batch_X), batch_y).item()
    initial_val_loss /= len(val_loader)
    
    initial_corr_seq = mean_pearson_correlation(val_loader, model, device, dim=1)
    initial_corr_type = mean_pearson_correlation(val_loader, model, device, dim=0)

    best_corr_sum = -float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    num_epochs = 300

    # Initialize lists with initial values
    train_losses = []
    val_losses = [initial_val_loss]
    val_corrs_seq = [initial_corr_seq]
    val_corrs_type = [initial_corr_type]

    print(f"Initial - Val Loss: {initial_val_loss:.4f} - Corr (Seq): {initial_corr_seq:.4f} - Corr (Type): {initial_corr_type:.4f}")

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

    # Create x-axis for plots (including initial point at epoch 0)
    epochs_range = list(range(len(train_losses)))
    epochs_range_val = list(range(len(val_losses)))

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range_val, val_losses, label="Val Loss")
    plt.title(f"{model_name} | Loss Curves | {sequence_length}bp")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"training_curve_{sequence_length}bp.png"))

    plt.figure()
    plt.plot(epochs_range_val, val_corrs_seq, label="Corr (Sequence)")
    plt.plot(epochs_range_val, val_corrs_type, label="Corr (Cell Type)")
    plt.title(f"{model_name} | Correlation Curves | {sequence_length}bp")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"validation_correlation_curve_{sequence_length}bp.png"))
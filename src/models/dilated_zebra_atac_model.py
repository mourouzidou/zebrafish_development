import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Model ---------------- #
class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2*dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out + x  # Residual connection


class ATACSeqCNN(nn.Module):
    def __init__(self, sequence_length, num_targets):
        super(ATACSeqCNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=19, padding=9),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
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
            nn.MaxPool1d(2)
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
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, list):
                preds = self.forward(x[0].to('cpu'))  # <-- Force to CPU
            else:
                preds = self.forward(x.to('cpu'))
        return preds.cpu().numpy()  # <-- Force to CPU


def forward_with_reverse_avg(model, x):
    out_fwd = model(x, reverse=False)
    out_rev = model(x, reverse=True)
    return 0.5 * (out_fwd + out_rev)


def pearson_corr_general(x, y, dim):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=dim)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=dim)) * torch.sqrt((y_centered ** 2).sum(dim=dim))
    return (numerator / (denominator + 1e-8)).mean(dim=dim)

def mean_pearson_correlation(loader, model, device, dim=1):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if dim == 0:
        corrs = [pearson_corr_general(all_preds[:, i], all_targets[:, i], dim=0).item() for i in range(all_preds.shape[1])]
    else:
        corrs = [pearson_corr_general(all_preds[i], all_targets[i], dim=0).item() for i in range(all_preds.shape[0])]
    return np.mean(corrs)

def one_hot_encode(sequence):
    return np.array(list(sequence))[:, None] == np.array(list('ACGT'))

def reverse_complement_tensor(x):
    # x: (batch, 4, length)
    rev_x = torch.flip(x, dims=[2])  # Reverse the sequence
    rev_x = rev_x[:, [3, 2, 1, 0], :]  # Complement: A<->T (0<->3), C<->G (1<->2)
    return rev_x


def load_and_split_data(csv_path, sequence_length):
    df = pd.read_csv(csv_path)
    df = df[df['sequence'].apply(lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    y = df.iloc[:, 0:num_targets].to_numpy(dtype=np.float32)
    X = np.stack([one_hot_encode(seq) for seq in df['sequence']]).astype(np.float32)
    X_tensor = torch.tensor(X).permute(0, 2, 1)
    y_tensor = torch.tensor(y)

    chroms = df['chromosome'].unique()
    np.random.seed(42)
    np.random.shuffle(chroms)
    split_idx = int(len(chroms) * 0.8)
    train_mask = df['chromosome'].isin(chroms[:split_idx]).values
    val_mask = df['chromosome'].isin(chroms[split_idx:]).values

    return X_tensor[train_mask], X_tensor[val_mask], y_tensor[train_mask], y_tensor[val_mask]


# ---------------- Training ---------------- #
if __name__ == "__main__":
    stage_prefix = "dpf1"
    sequence_length = 2000
    modality = "ATAC"
    dataset_path = f"../scDataDevStages/{stage_prefix}_accessibility.csv"


    df = pd.read_csv(dataset_path)
    num_targets = df.shape[1] - 4
    model = ATACSeqCNN(sequence_length, num_targets)
    X_train, X_val, y_train, y_val = load_and_split_data(dataset_path, sequence_length)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # criterion = nn.MSELoss()
    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)

    train_losses, val_losses, val_corrs_seq, val_corrs_type = [], [], [], []
    best_corr_sum = -float('inf')
    patience_counter = 0
    num_epochs = 300
    early_stopping_patience = 10
    plt.hist(y_train.numpy().flatten(), bins=100)
    plt.title("Distribution of Raw Counts")
    plt.yscale("log")
    plt.show()

    initial_val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            initial_val_loss += loss.item()
    initial_val_loss /= len(val_loader)
    initial_corr_seq = mean_pearson_correlation(val_loader, model, device, dim=1)
    initial_corr_type = mean_pearson_correlation(val_loader, model, device, dim=0)
    print(f"Initial Val Loss: {initial_val_loss:.4f} - Corr (Per Sequence): {initial_corr_seq:.4f} - Corr (Per Cell Type): {initial_corr_type:.4f}")

    train_losses.append(None)
    val_losses.append(initial_val_loss)
    val_corrs_seq.append(initial_corr_seq)
    val_corrs_type.append(initial_corr_type)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        corr_seq = mean_pearson_correlation(val_loader, model, device, dim=1)
        corr_type = mean_pearson_correlation(val_loader, model, device, dim=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_corrs_seq.append(corr_seq)
        val_corrs_type.append(corr_type)

        corr_sum = corr_seq + corr_type
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Corr (Seq): {corr_seq:.4f} - Corr (Type): {corr_type:.4f}")

        if corr_sum > best_corr_sum:
            best_corr_sum = corr_sum
            patience_counter = 0
            torch.save(model.state_dict(), f"{stage_prefix}_dilated{modality}model{num_targets}.pth")
            print("Model saved")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    plt.figure()
    plt.plot(train_losses[1:], label='Train Loss')
    plt.plot(val_losses[1:], label='Validation Loss')
    plt.axhline(y=val_losses[0], linestyle='--', color='gray', label='Initial Val Loss')
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("training_curve.png")

    plt.figure()
    plt.plot(val_corrs_seq[1:], label='Val Corr (Per Sequence)')
    plt.plot(val_corrs_type[1:], label='Val Corr (Per Cell Type)')
    plt.axhline(y=val_corrs_seq[0], linestyle='--', color='gray', label='Initial Corr (Seq)')
    plt.axhline(y=val_corrs_type[0], linestyle='--', color='black', label='Initial Corr (Type)')
    plt.legend()
    plt.title("Validation Correlation Curves")
    plt.savefig("validation_correlation_curve.png")

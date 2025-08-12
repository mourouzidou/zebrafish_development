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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.environ["TORCH_JIT_DISABLE_ONEDNN_FUSION"] = "1"

# ---------------- Data Processing Functions ---------------- #
def one_hot_encode(sequence):
    """Convert DNA sequence to one-hot encoded format"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr

def reverse_complement_tensor(x):
    """Apply reverse complement to one-hot encoded sequence tensor"""
    rev_x = torch.flip(x, dims=[2])
    rev_x = rev_x[:, [3, 2, 1, 0], :]
    return rev_x

def reduce_gene_expression_to_300(gene_expression_matrix, method='pca'):
    """
    Reduce gene expression matrix from original dimensions to 300 features
    
    Args:
        gene_expression_matrix: numpy array of shape (n_genes, n_pseudobulks)
        method: 'pca' or 'top_variance'
    
    Returns:
        reduced_matrix: numpy array of shape (300, n_pseudobulks)
        reducer: fitted reducer object for future use
    """
    if method == 'pca':
        # Standardize the data (genes as features, pseudobulks as samples)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(gene_expression_matrix.T)  # Shape: (n_pseudobulks, n_genes)
        
        # Apply PCA
        pca = PCA(n_components=300)
        reduced_data = pca.fit_transform(scaled_data)  # Shape: (n_pseudobulks, 300)
        
        # Transpose back to (300, n_pseudobulks)
        reduced_matrix = reduced_data.T
        
        reducer = {'scaler': scaler, 'pca': pca, 'method': 'pca'}
        
    elif method == 'top_variance':
        # Select top 300 genes by variance
        gene_variances = np.var(gene_expression_matrix, axis=1)
        top_indices = np.argsort(gene_variances)[-300:]
        reduced_matrix = gene_expression_matrix[top_indices, :]
        
        reducer = {'top_indices': top_indices, 'method': 'top_variance'}
    
    return reduced_matrix, reducer

# ---------------- Model Components ---------------- #
class DilatedBlock(nn.Module):
    """Dilated convolution block with residual connection"""
    def __init__(self, channels, dilation, kernel_size=9):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                             padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual

class MultimodalATACSeqCNN(nn.Module):
    """
    Multimodal CNN that integrates DNA sequence and RNA expression data
    to predict ATAC-seq accessibility across multiple cell types
    """
    def __init__(self, sequence_length=2048, num_targets=55, num_genes=300):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_targets = num_targets
        self.num_genes = num_genes
        
        # First convolution layer (19bp kernels, 300 filters)
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 300, kernel_size=19, padding=9),
            nn.BatchNorm1d(300),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Second convolution layer (9bp kernels, 300 filters)
        self.conv2 = nn.Sequential(
            nn.Conv1d(300, 300, kernel_size=9, padding=4),
            nn.BatchNorm1d(300),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(4)  # Pool by 4 -> 512 length
        )
        
        # Dilated convolution layers (5 layers with increasing dilation)
        self.dilated_blocks = nn.ModuleList([
            DilatedBlock(300, dilation=2**(i+1)) for i in range(5)
        ])
        
        # Additional convolution layers (7bp kernels)
        self.conv_tail = nn.Sequential(
            nn.Conv1d(300, 300, kernel_size=7, padding=3),
            nn.BatchNorm1d(300),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(4),  # 512 -> 128
            
            nn.Conv1d(300, 300, kernel_size=7, padding=3),
            nn.BatchNorm1d(300),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(4),  # 128 -> 32
            
            nn.Conv1d(300, 300, kernel_size=7, padding=3),
            nn.BatchNorm1d(300),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(4),  # 32 -> 8
        )
        
        # Calculate the final sequence length after all pooling operations
        # Initial: 2048 -> after pool(4): 512 -> after 3x pool(4): 8
        final_seq_length = 8
        fc_input_dim = 300 * final_seq_length
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_targets),
            nn.Softplus()
        )
        
        # RNA expression processing
        # This will be used to modulate the first convolution output
        self.rna_projection = nn.Linear(num_genes, 300)
        
    def forward(self, sequence, rna_expression, reverse=False):
        """
        Forward pass of the multimodal model
        
        Args:
            sequence: DNA sequence tensor (batch_size, 4, sequence_length)
            rna_expression: RNA expression tensor (batch_size, num_genes, num_targets)
            reverse: whether to apply reverse complement
        """
        if reverse:
            sequence = reverse_complement_tensor(sequence)
        
        # First convolution
        x = self.conv1(sequence)  # (batch_size, 300, sequence_length)
        
        # RNA integration: Hadamard product with gene expression
        # rna_expression: (batch_size, num_genes, num_targets)
        # We need to create a modulation signal for each convolution filter
        
        batch_size = x.shape[0]
        
        # Process RNA expression for each target (pseudobulk)
        rna_modulations = []
        for target_idx in range(self.num_targets):
            # Get RNA expression for this target: (batch_size, num_genes)
            rna_target = rna_expression[:, :, target_idx]
            
            # Project to 300 dimensions (matching conv filters)
            rna_projected = self.rna_projection(rna_target)  # (batch_size, 300)
            
            # Apply sigmoid to create modulation weights
            rna_modulation = torch.sigmoid(rna_projected)  # (batch_size, 300)
            rna_modulations.append(rna_modulation)
        
        # Stack all modulations: (batch_size, 300, num_targets)
        rna_modulation_stack = torch.stack(rna_modulations, dim=2)
        
        # For now, we'll use the mean modulation across all targets
        # This is a simplification - you might want to modify this based on your specific needs
        mean_rna_modulation = rna_modulation_stack.mean(dim=2)  # (batch_size, 300)
        
        # Apply RNA modulation to convolution output
        # Expand dimensions to match x: (batch_size, 300, 1)
        rna_modulation_expanded = mean_rna_modulation.unsqueeze(2)
        
        # Apply Hadamard product (element-wise multiplication)
        x = x * rna_modulation_expanded  # (batch_size, 300, sequence_length)
        
        # Continue with the rest of the network
        x = self.conv2(x)
        
        # Apply dilated blocks
        for dilated_block in self.dilated_blocks:
            x = dilated_block(x)
        
        # Apply tail convolutions
        x = self.conv_tail(x)
        
        # Flatten and classify
        x = x.reshape(x.size(0), -1)
        output = self.classifier(x)
        
        return output

# ---------------- Training and Evaluation Functions ---------------- #
def pearson_corr_general(x, y, dim):
    """Calculate Pearson correlation coefficient"""
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=dim)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=dim)) * torch.sqrt((y_centered ** 2).sum(dim=dim))
    return (numerator / (denominator + 1e-8)).mean(dim=dim)

def mean_pearson_correlation(loader, model, device, dim=1):
    """Calculate mean Pearson correlation on validation set"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_rna, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_rna = batch_rna.to(device)
            batch_y = batch_y.to(device)
            
            preds = model(batch_X, batch_rna)
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    if dim == 0:
        return np.mean([pearson_corr_general(all_preds[:, i], all_targets[:, i], dim=0).item() 
                       for i in range(all_preds.shape[1])])
    else:
        return np.mean([pearson_corr_general(all_preds[i], all_targets[i], dim=0).item() 
                       for i in range(all_preds.shape[0])])

def load_and_split_multimodal_data(atac_csv_path, rna_csv_path, sequence_length=2048):
    """
    Load and split both ATAC-seq and RNA expression data
    
    Args:
        atac_csv_path: Path to ATAC-seq data CSV
        rna_csv_path: Path to RNA expression data CSV
        sequence_length: Expected sequence length
    
    Returns:
        Training and validation datasets for both modalities
    """
    # Load ATAC-seq data
    atac_df = pd.read_csv(atac_csv_path)
    atac_df = atac_df[atac_df['sequence'].apply(
        lambda x: isinstance(x, str) and len(x) == sequence_length)].reset_index(drop=True)
    
    # Load RNA expression data
    rna_df = pd.read_csv(rna_csv_path)
    
    # Assume RNA data has genes as rows and pseudobulks as columns
    # Extract gene expression matrix (exclude metadata columns if any)
    rna_matrix = rna_df.iloc[:, 1:].values  # Skip first column (gene names)
    
    # Reduce RNA matrix to 300 dimensions
    rna_reduced, rna_reducer = reduce_gene_expression_to_300(rna_matrix, method='pca')
    
    # Process ATAC data
    atac_y = np.log1p(atac_df.iloc[:, 0:-4].to_numpy(dtype=np.float32))
    atac_X = np.stack([one_hot_encode(seq) for seq in atac_df['sequence']])
    
    # Convert to tensors
    atac_X_tensor = torch.tensor(atac_X).permute(0, 2, 1)
    atac_y_tensor = torch.tensor(atac_y)
    
    # Create RNA tensor for each sequence
    # Assuming the RNA data corresponds to the same pseudobulks as ATAC
    num_sequences = len(atac_df)
    num_genes_reduced = rna_reduced.shape[0]
    num_targets = rna_reduced.shape[1]
    
    # Repeat RNA data for each sequence
    rna_tensor = torch.tensor(rna_reduced, dtype=torch.float32).unsqueeze(0).repeat(num_sequences, 1, 1)
    
    # Split by chromosome
    chroms = atac_df['chromosome'].unique()
    np.random.shuffle(chroms)
    split_idx = int(len(chroms) * 0.8)
    train_mask = atac_df['chromosome'].isin(chroms[:split_idx]).values
    val_mask = atac_df['chromosome'].isin(chroms[split_idx:]).values
    
    return (atac_X_tensor[train_mask], atac_X_tensor[val_mask],
            rna_tensor[train_mask], rna_tensor[val_mask],
            atac_y_tensor[train_mask], atac_y_tensor[val_mask],
            rna_reducer)

# ---------------- Main Training Script ---------------- #
if __name__ == "__main__":
    # Configuration
    sequence_length = 2048
    num_targets = 55
    num_genes = 300
    model_name = "multimodal_atac_rna_model"
    
    # Data paths (you'll need to update these)
    atac_dataset_path = f"data/embryo/processed/atac_peaks_with_sequences_{sequence_length}.csv"
    rna_dataset_path = "data/embryo/processed/rna_expression_pseudobulks.csv"
    
    output_dir = f"src/models/outputs/{model_name}_{sequence_length}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data
    print("Loading and processing data...")
    X_train, X_val, rna_train, rna_val, y_train, y_val, rna_reducer = load_and_split_multimodal_data(
        atac_dataset_path, rna_dataset_path, sequence_length
    )
    
    print(f"Training data shapes:")
    print(f"  ATAC sequences: {X_train.shape}")
    print(f"  RNA expression: {rna_train.shape}")
    print(f"  ATAC targets: {y_train.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, rna_train, y_train)
    val_dataset = TensorDataset(X_val, rna_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = MultimodalATACSeqCNN(sequence_length, num_targets, num_genes)
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training parameters
    best_corr_sum = -float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    num_epochs = 300
    
    train_losses, val_losses, val_corrs_seq, val_corrs_type = [], [], [], []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_rna, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_rna = batch_rna.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(batch_X, batch_rna), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_rna, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_rna = batch_rna.to(device)
                batch_y = batch_y.to(device)
                val_loss += criterion(model(batch_X, batch_rna), batch_y).item()
        
        val_loss /= len(val_loader)
        
        # Calculate correlations
        corr_seq = mean_pearson_correlation(val_loader, model, device, dim=1)
        corr_type = mean_pearson_correlation(val_loader, model, device, dim=0)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_corrs_seq.append(corr_seq)
        val_corrs_type.append(corr_type)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Corr (Seq): {corr_seq:.4f} - Corr (Type): {corr_type:.4f}")
        
        # Early stopping and model saving
        corr_sum = corr_seq + corr_type
        if corr_sum > best_corr_sum:
            best_corr_sum = corr_sum
            patience_counter = 0
            
            # Save model and reducer
            torch.save(model.state_dict(), 
                      os.path.join(output_dir, f"{model_name}_{sequence_length}bp_best.pth"))
            torch.save(rna_reducer, 
                      os.path.join(output_dir, f"rna_reducer_{sequence_length}bp.pth"))
            
            print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"{model_name} | Loss Curves | {sequence_length}bp")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_corrs_seq, label="Corr (Sequence)")
    plt.plot(val_corrs_type, label="Corr (Cell Type)")
    plt.title(f"{model_name} | Correlation Curves | {sequence_length}bp")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_curves_{sequence_length}bp.png"))
    plt.show()
    
    print(f"Training completed. Best correlation sum: {best_corr_sum:.4f}")
    print(f"Model and results saved to: {output_dir}")
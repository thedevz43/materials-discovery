import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_materials_json
from featurizers import MaterialsDataset
from models.gnn import CGCNN

# -------------------------
# Training & Evaluation Utils
# -------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        # Ensure batch.y shape matches out shape
        target = batch.y.view(-1, 1)
        loss = torch.nn.functional.l1_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses), losses

def eval_epoch(model, loader, device, return_preds=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    if return_preds:
        return mae, rmse, preds, targets
    return mae, rmse

# -------------------------
# Ensemble Training
# -------------------------
def train_ensemble(args, dataset, device, n_ensemble=5):
    """
    Train an ensemble of CGCNN models.
    """
    n = len(dataset)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    train_idx = idxs[:int(0.7*n)]
    val_idx = idxs[int(0.7*n):int(0.85*n)]
    test_idx = idxs[int(0.85*n):]
    train_set = [dataset.get(i) for i in train_idx]
    val_set = [dataset.get(i) for i in val_idx]
    test_set = [dataset.get(i) for i in test_idx]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    maes, rmses = [], []
    for i in range(n_ensemble):
        print(f"\n=== Training ensemble model {i+1}/{n_ensemble} ===")
        model = CGCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_mae = float('inf')

        train_losses = []
        for epoch in range(args.epochs):
            train_loss, batch_losses = train_epoch(model, train_loader, optimizer, device)
            train_losses.append(train_loss)
            val_mae, val_rmse = eval_epoch(model, val_loader, device)
            print(f"Model {i+1} Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val MAE={val_mae:.4f}, Val RMSE={val_rmse:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), f"cgcnn_{i}.pt")

        # Plot training loss curve
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Curve (Model {i+1})')
        plt.legend()
        plt.savefig(f'train_loss_curve_{i+1}.png')
        plt.close()

        # Test set predictions vs actual
        test_mae, test_rmse, preds, targets = eval_epoch(model, test_loader, device, return_preds=True)
        maes.append(test_mae)
        rmses.append(test_rmse)
        print(f"Model {i+1} Test MAE={test_mae:.4f}, Test RMSE={test_rmse:.4f}")

        plt.figure()
        plt.scatter(targets, preds, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Ideal')
        plt.xlabel('Actual Volume')
        plt.ylabel('Predicted Volume')
        plt.title(f'Predicted vs Actual (Model {i+1})')
        plt.legend()
        plt.savefig(f'pred_vs_actual_{i+1}.png')
        plt.close()

    print("\n=== Ensemble Results ===")
    print(f"Ensemble Test MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"Ensemble Test RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

# -------------------------
# Main Script
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CGCNN on materials data")
    parser.add_argument("--dataset_path", type=str, default="data/materials.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--checkpoint", type=str, default="cgcnn.pt")
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble of models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    materials = load_materials_json(args.dataset_path)
    dataset = MaterialsDataset(materials, cutoff=args.cutoff)

    if args.ensemble:
        train_ensemble(args, dataset, device, n_ensemble=5)
    else:
        # Standard single-model training
        n = len(dataset)
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        train_idx = idxs[:int(0.7*n)]
        val_idx = idxs[int(0.7*n):int(0.85*n)]
        test_idx = idxs[int(0.85*n):]
        train_set = [dataset.get(i) for i in train_idx]
        val_set = [dataset.get(i) for i in val_idx]
        test_set = [dataset.get(i) for i in test_idx]
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        model = CGCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_mae = float('inf')
        train_losses = []
        for epoch in range(args.epochs):
            train_loss, batch_losses = train_epoch(model, train_loader, optimizer, device)
            train_losses.append(train_loss)
            val_mae, val_rmse = eval_epoch(model, val_loader, device)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val MAE={val_mae:.4f}, Val RMSE={val_rmse:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), args.checkpoint)
                print(f"Checkpoint saved to {args.checkpoint}")

        # Plot training loss curve
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig('train_loss_curve.png')
        plt.close()

        # Test set predictions vs actual
        test_mae, test_rmse, preds, targets = eval_epoch(model, test_loader, device, return_preds=True)
        print(f"Test MAE={test_mae:.4f}, Test RMSE={test_rmse:.4f}")

        plt.figure()
        plt.scatter(targets, preds, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Ideal')
        plt.xlabel('Actual Volume')
        plt.ylabel('Predicted Volume')
        plt.title('Predicted vs Actual')
        plt.legend()
        plt.savefig('pred_vs_actual.png')
        plt.close()

if __name__ == "__main__":
    main()

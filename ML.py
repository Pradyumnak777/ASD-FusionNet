import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import torch.optim as optim


class FusionDataset(Dataset):
    def __init__(self, conn_dict, pheno_dict, label_dict):
        self.conn_dict = conn_dict
        self.pheno_dict = pheno_dict
        self.label_dict = label_dict

        self.keys = list(set(conn_dict.keys()) & set(pheno_dict.keys()) & set(label_dict.keys())) #all keys are same anyways
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        sub_id = self.keys[idx]
        # for fc values
        conn_vals = torch.tensor(self.conn_dict[sub_id], dtype=torch.float32)
        # for pheno features
        pheno_vals = torch.tensor(self.pheno_dict[sub_id], dtype=torch.float32)
        label = torch.tensor(self.label_dict[sub_id], dtype=torch.long)

        return (conn_vals, pheno_vals), label
    

class MultiBranchNet(nn.Module):
    def __init__(self, conn_dim=19900, pheno_dim=6, embed_conn=256, embed_pheno=16):
        super().__init__()

        # fc branch
        self.branch_conn = nn.Sequential(
            nn.Linear(conn_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embed_conn),
            nn.ReLU()
        )

        # pheno branch
        self.branch_pheno = nn.Sequential(
            nn.Linear(pheno_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, embed_pheno),
            nn.ReLU()
        )

        # classifer
        self.classifier = nn.Sequential(
            nn.Linear(embed_conn + embed_pheno, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, conn_x, pheno_x):
        out_conn = self.branch_conn(conn_x)      # (batch, embed_conn)
        out_pheno = self.branch_pheno(pheno_x)   # (batch, embed_pheno)
        fused = torch.cat([out_conn, out_pheno], dim=1)  #  (batch, embed_conn+embed_pheno)
        logits = self.classifier(fused).squeeze(1)       # (batch,)
        return logits 

def train(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for (conn, pheno), labels in dataloader:
        conn, pheno, labels = conn.to(device), pheno.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(conn, pheno)  # shape: (batch,)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

        # Accuracy calc
        # preds = (logits > 0).long()
        # correct += preds.eq(labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / total
    return avg_loss

def evaluate(model, dataloader, device):
    # ipdb.set_trace()
    model.eval()
    y_true, y_pred = [], []
    y_scores = []

    with torch.no_grad():
        for (conn, pheno), labels in dataloader:
            conn, pheno = conn.to(device), pheno.to(device)
            logits = model(conn, pheno)  # Only use logits
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()

            y_scores.extend(probs.cpu().numpy())
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=0)
    rec = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred, pos_label=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[1, 1], cm[1, 0]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
      auc = roc_auc_score(y_true, y_scores)
    except ValueError as e:

      # print("AUC calculation error:", e)
      # print("y_true:", np.unique(y_true, return_counts=True))
      # print("y_scores (unique):", np.unique(y_scores))
      auc = float('nan')

    return acc, prec, rec, specificity, f1, auc

def cross_validate_fusion(conn_dict, pheno_dict, label_dict,
                          device='cuda', n_splits=5, epochs=20,
                          save_dir='/content/drive/MyDrive/project-2', eval_only=False):
    os.makedirs(save_dir, exist_ok=True)

    dataset = FusionDataset(conn_dict, pheno_dict, label_dict)
    labels = [label_dict[k] for k in dataset.keys]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.keys, labels)):
        print(f"\n Fold {fold + 1}/{n_splits}")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=10, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=10)

        # Detect dimensions
        conn_dim = len(conn_dict[dataset.keys[0]])
        pheno_dim = len(pheno_dict[dataset.keys[0]])

        model = MultiBranchNet(conn_dim=conn_dim, pheno_dim=pheno_dim).to(device)
        model_path = os.path.join(save_dir, f'latest_fold{fold + 1}.pth')

        if not eval_only:
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

            for epoch in range(epochs):
                avg_loss = train(model, train_loader, optimizer, device)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")

        acc, prec, rec, spec, f1, auc = evaluate(model, val_loader, device)
        fold_metrics.append((acc, prec, rec, spec, f1, auc))

        print(f"Fold {fold + 1} Results:")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Precision   : {prec:.4f}")
        print(f"  Recall      : {rec:.4f}")
        print(f"  Specificity : {spec:.4f}")
        print(f"  F1 Score    : {f1:.4f}")
        print(f"  AUC         : {auc:.4f}")

    accs, precs, recs, specs, f1s, aucs = zip(*fold_metrics)
    print("\nFinal Summary")
    print(f"Avg Accuracy    : {np.mean(accs):.4f}")
    print(f"Avg Precision   : {np.mean(precs):.4f}")
    print(f"Avg Recall      : {np.mean(recs):.4f}")
    print(f"Avg Specificity : {np.mean(specs):.4f}")
    print(f"Avg F1 Score    : {np.mean(f1s):.4f}")
    print(f"Avg AUC         : {np.mean(aucs):.4f}")

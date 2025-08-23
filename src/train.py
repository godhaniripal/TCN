import argparse, os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import TCN

def load_classes(path):
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

class NpzDataset(Dataset):
    def __init__(self, files, mean=None, std=None):
        self.files = files
        self.mean = mean
        self.std = std

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        seq = d["keypoints"].astype(np.float32)  # (T,F)
        mask = d["mask"].astype(np.float32)      # (T,)
        label = int(d["label"])
        if self.mean is not None and self.std is not None:
            # standardize valid frames only
            seq[mask==1] = (seq[mask==1] - self.mean) / self.std
            # keep padded frames as zeros (helps masked pooling)
            seq[mask==0] = 0.0
        # return as (F,T)
        x = torch.from_numpy(seq.transpose(1,0))  # (F,T)
        m = torch.from_numpy(mask)                # (T,)
        y = torch.tensor(label, dtype=torch.long)
        return x, m, y

def compute_stats(files):
    # per-dimension mean/std over valid frames only
    sum_vec = None
    sq_sum_vec = None
    count = 0.0
    for fp in files:
        d = np.load(fp)
        seq = d["keypoints"].astype(np.float32)  # (T,F)
        mask = d["mask"].astype(np.float32)      # (T,)
        if sum_vec is None:
            F = seq.shape[1]
            sum_vec = np.zeros((F,), np.float64)
            sq_sum_vec = np.zeros((F,), np.float64)
        valid = mask[:, None]  # (T,1)
        sum_vec += (seq * valid).sum(axis=0)
        sq_sum_vec += ((seq**2) * valid).sum(axis=0)
        count += float(mask.sum())
    mean = (sum_vec / max(count, 1.0)).astype(np.float32)
    var = (sq_sum_vec / max(count, 1.0)) - (mean.astype(np.float64)**2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std

def split_files(npz_dir, classes, seed=42):
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    per_class = {i: [] for i in range(len(classes))}
    for fp in files:
        cid = int(np.load(fp)["label"])
        per_class[cid].append(fp)
    random.Random(seed).shuffle(files)
    train, val, test = [], [], []
    for cid, lst in per_class.items():
        random.Random(seed).shuffle(lst)
        n = len(lst); n_tr = int(0.8*n); n_val = int(0.1*n)
        train += lst[:n_tr]
        val += lst[n_tr:n_tr+n_val]
        test += lst[n_tr+n_val:]
    return train, val, test

def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x, m)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        total_acc += float((preds == y).sum().item())
        n += x.size(0)
    return total_loss/n, total_acc/n

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        total_acc += float((preds == y).sum().item())
        n += x.size(0)
    return total_loss/n, total_acc/n

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hardcoded paths for new data and classes
    classes = load_classes(r"outputs/models/classes.txt")
    npz_dir = r"outputs/models/npz2"
    train_files, val_files, test_files = split_files(npz_dir, classes, seed=args.seed)

    mean, std = compute_stats(train_files)
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    np.savez(os.path.join(args.out_dir, "models", "stats.npz"), mean=mean, std=std)

    ds_tr = NpzDataset(train_files, mean, std)
    ds_va = NpzDataset(val_files, mean, std)
    ds_te = NpzDataset(test_files, mean, std)

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, num_workers=0)

    model = TCN(in_ch=ds_tr[0][0].shape[0], n_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_va, best_path, patience_ctr = 0.0, None, 0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, device, criterion)
        va_loss, va_acc = eval_epoch(model, dl_va, device, criterion)
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_va:
            best_va = va_acc
            best_path = os.path.join(args.out_dir, "models", "tcn_best2.pt")
            torch.save(model.state_dict(), best_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("Early stopping.")
                break

    # evaluate best model on test
    if best_path:
        model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc = eval_epoch(model, dl_te, device, criterion)
    print(f"Test: loss {te_loss:.4f} acc {te_acc:.3f}")
    # save classes next to model
    with open(os.path.join(args.out_dir, "models", "classes.txt"), "w") as f:
        for c in classes: f.write(c + "\n")
    print(f"Saved model to {best_path}")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
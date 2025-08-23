import argparse, os, time, collections
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from features import HandFeatureExtractor, FEATS_PER_FRAME
from model import TCN

def load_classes(path):
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes)
    stats = np.load(args.stats)
    mean, std = stats["mean"], stats["std"]

    model = TCN(in_ch=FEATS_PER_FRAME, n_classes=len(classes))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    cap = cv2.VideoCapture(0)
    extractor = HandFeatureExtractor()
    T = args.T
    buf = collections.deque(maxlen=T)

    def standardize(seq, mask):
        seq = seq.copy()
        seq[mask==1] = (seq[mask==1] - mean) / std
        seq[mask==0] = 0.0
        return seq

    EMA = None
    alpha = 0.6  # smoothing
    conf_thresh = 0.6

    while True:
        ok, frame = cap.read()
        if not ok: break
        feat = extractor.frame_to_feature(frame)
        buf.append(feat)

        # build (T,F) and mask
        if len(buf) < T:
            seq = np.zeros((T, FEATS_PER_FRAME), np.float32)
            mask = np.zeros((T,), np.float32)
            seq[:len(buf)] = np.array(buf, dtype=np.float32)
            mask[:len(buf)] = 1.0
        else:
            seq = np.array(buf, dtype=np.float32)
            mask = np.ones((T,), np.float32)

        seq = standardize(seq, mask)
        x = torch.from_numpy(seq.transpose(1,0)).unsqueeze(0).to(device)  # (1,F,T)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)                 # (1,T)

        with torch.no_grad():
            logits = model(x, m)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        # EMA smoothing
        if EMA is None: EMA = probs
        else: EMA = alpha*probs + (1-alpha)*EMA

        pred_id = int(np.argmax(EMA))
        pred_p = float(EMA[pred_id])

        label = classes[pred_id] if pred_p >= conf_thresh else "…"
        txt = f"{label} ({pred_p:.2f})"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.imshow("ISL Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    extractor.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="outputs/models/tcn_best.pt")
    ap.add_argument("--stats", type=str, default="outputs/models/stats.npz")
    ap.add_argument("--classes", type=str, default="outputs/models/classes.txt")
    ap.add_argument("--T", type=int, default=48)
    args = ap.parse_args()
    main(args)
import argparse, os, glob, json
import numpy as np
from tqdm import tqdm
from features import HandFeatureExtractor, FEATS_PER_FRAME

def main(args):
    classes = args.classes.split(',')
    classes = [c.strip() for c in classes if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    # Support saving to npz2 if specified in out_dir
    npz_dirname = "npz2" if "npz2" in args.out_dir else "npz"
    npz_dir = os.path.join(args.out_dir, npz_dirname)
    os.makedirs(npz_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "classes.txt"), "w") as f:
        for c in classes: f.write(c + "\n")

    extractor = HandFeatureExtractor()

    idx = 0

    class_patterns = {
        "hello": "hello_*.mp4",
        "thanks": "ThankYou_*.mp4",
        "no": "No_*.mp4",
        "ask": "ask_*.mp4",
        "about": "about_*.mp4"
    }
    for cid, cname in enumerate(classes):
        pattern = class_patterns.get(cname, f"{cname}_*.mp4")
        files = sorted(glob.glob(os.path.join(args.raw_dir, pattern)))
        print(f"{cname}: {len(files)} videos")
        for vp in tqdm(files):
            seq, mask = extractor.video_to_sequence(vp, target_len=args.T)
            out_path = os.path.join(npz_dir, f"{cname}_{idx:05d}.npz")
            np.savez_compressed(out_path,
                keypoints=seq.astype(np.float32),  # (T,94)
                mask=mask.astype(np.float32),      # (T,)
                label=np.int64(cid),
                classname=cname,
                src=os.path.basename(vp)
            )
            idx += 1

    extractor.close()
    print(f"Saved {idx} samples to {npz_dir}")
    print(f"Feature dim per frame: {FEATS_PER_FRAME}, T={args.T}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--classes", type=str, default="hello,thanks,yes")
    ap.add_argument("--T", type=int, default=48)
    args = ap.parse_args()
    main(args)
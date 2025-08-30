import os
import cv2
import time
from datetime import datetime

# Directory to save collected videos
def get_save_dir(word):
    save_dir = os.path.join("data", word)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def get_next_index(save_dir):
    files = [f for f in os.listdir(save_dir) if f.endswith('.mp4')]
    if not files:
        return 1
    nums = [int(f.split('_')[1].split('.')[0]) for f in files if '_' in f]
    return max(nums, default=0) + 1

def record_video(word, idx, duration=5):
    save_dir = get_save_dir(word)
    filename = f"{word}_{idx:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(save_dir, filename)
    cap = cv2.VideoCapture(0)
    fps = 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Preparation countdown (no saving yet)
    prep_time = 2
    for t in range(prep_time, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        disp = frame.copy()
        cv2.putText(disp, f"Get Ready: {t}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        cv2.imshow('Recording', disp)
        cv2.waitKey(1000)
    # Start recording after countdown
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    print(f"Recording {filename} for {duration} seconds...")
    print(f"Recording {filename} for {duration} seconds...")
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = time.time() - start
        # Draw countdown (bottom right, small)
        disp = frame.copy()
        countdown = max(0, int(duration - elapsed) + 1)
        h, w = disp.shape[:2]
        cv2.putText(disp, f"{countdown}", (w-60, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        out.write(frame)
        cv2.imshow('Recording', disp)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Recording cancelled.")
            break
        if elapsed >= duration:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved: {filepath}")

def print_stats(word):
    save_dir = get_save_dir(word)
    files = [f for f in os.listdir(save_dir) if f.endswith('.mp4')]
    print(f"Current samples for '{word}': {len(files)}")

def main():
    word = input("Enter the word/gesture to collect: ").strip()
    n_samples = int(input("How many samples do you want to record? ").strip())
    duration = 5  # seconds, compulsory static duration
    print(f"\nReady to collect {n_samples} samples for '{word}'. Each will be {duration} seconds.")
    for i in range(n_samples):
        print(f"\nSample {i+1}/{n_samples}")
        idx = get_next_index(get_save_dir(word))
        input("Press Enter to start recording...")
        record_video(word, idx, duration)
        print_stats(word)
    print(f"\nCollection complete for '{word}'.")

if __name__ == "__main__":
    main()

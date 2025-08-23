# CNN Experiment: Hand Gesture Recognition

This project is a complete pipeline for hand gesture recognition using video data. It leverages MediaPipe for hand landmark extraction, processes video frames into feature sequences, and trains a Temporal Convolutional Network (TCN) to classify gestures. The workflow is designed for easy extension to new gestures and datasets.

## Project Structure

```
cnn-experiment/
├── data/                # Raw video files (.mp4)
├── outputs/             # Preprocessed data, models, and results
├── src/                 # Source code
│   ├── features.py      # Feature extraction using MediaPipe
│   ├── preprocess.py    # Preprocessing script
│   ├── model.py         # TCN model definition
│   ├── train.py         # Training script
│   └── live.py          # Live prediction script
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

## Step-by-Step Guide

### 1. Install Dependencies

Create and activate a conda environment (recommended):

```sh
conda create -n myenv python=3.10 -y
conda activate myenv
pip install -r requirements.txt
```

### 2. Prepare Your Data

- Place your gesture videos in the `data/` folder.
- Name files according to their class, e.g.:
  - `hello_*.mp4` for "hello"
  - `thanks_*.mp4` or `ThankYou_*.mp4` for "thanks"
  - `no_*.mp4` or `No_*.mp4` for "no"
  - `ask_*.mp4` for "ask"
  - `about_*.mp4` for "about"

### 3. Preprocess the Data

Run the preprocessing script to extract features and save them as `.npz` files:

```sh
python src/preprocess.py --raw_dir data --out_dir outputs/npz2 --classes "hello,thanks,no,ask,about" --T 48
```
- `--raw_dir`: Folder with your videos
- `--out_dir`: Where to save processed data
- `--classes`: Comma-separated list of gesture classes
- `--T`: Number of frames per sample (sequence length)

### 4. Train the Model

Make sure your processed data and class list are in the correct locations:
- Processed `.npz` files: `outputs/models/npz2/`
- Class list: `outputs/models/classes.txt`

Train the model (the script is hardcoded for these paths):

```sh
python src/train.py --out_dir outputs/models --epochs 60 --bs 32
```
- The best model will be saved as `outputs/models/models/tcn_best2.pt`

### 5. Live Prediction

Run the live prediction script using your trained model:

```sh
python src/live.py --model outputs/models/models/tcn_best2.pt --stats outputs/models/models/stats.npz --classes outputs/models/models/classes.txt --T 48
```

### 6. Customization & Tips
- To add new gestures, add new videos with the correct naming pattern and update the `--classes` argument.
- You can adjust the sequence length `T`, batch size, and number of epochs as needed.
- All intermediate and output files (models, stats, classes) are saved in the `outputs/models/models/` directory by default.

## Project Highlights
- **Preprocessing**: Uses MediaPipe for robust hand landmark extraction.
- **Model**: Temporal Convolutional Network (TCN) for sequence modeling.
- **Extensible**: Easily add new gestures or more data.
- **Live Prediction**: Real-time gesture recognition from webcam or video.

---


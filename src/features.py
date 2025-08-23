import cv2, numpy as np, mediapipe as mp

IDX_WRIST, IDX_INDEX_MCP, IDX_PINKY_MCP = 0, 5, 17
FEATS_PER_FRAME = 94  # 84 local + 4 wrists + 2 inter + 4 scales/presence

def _palm_width(pts):
    return float(np.linalg.norm(pts[IDX_INDEX_MCP] - pts[IDX_PINKY_MCP]) + 1e-6)

def _normalize_local(pts):
    # center at wrist, scale by palm width
    wrist = pts[IDX_WRIST]
    scale = _palm_width(pts)
    local = (pts - wrist) / scale
    return local, scale

class HandFeatureExtractor:
    def __init__(self, max_num_hands=2, det_conf=0.5, track_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )

    def close(self):
        self.hands.close()

    def frame_to_feature(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        right_xy = np.zeros((21,2), dtype=np.float32)
        left_xy  = np.zeros((21,2), dtype=np.float32)
        r_present = 0.0
        l_present = 0.0

        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = hd.classification[0].label  # 'Left' or 'Right'
                pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
                if label == 'Right':
                    right_xy = pts; r_present = 1.0
                else:
                    left_xy = pts;  l_present = 1.0

        # per-hand local shape
        if r_present:
            right_local, r_scale = _normalize_local(right_xy)
            r_wrist = right_xy[IDX_WRIST]
        else:
            right_local = np.zeros((21,2), np.float32); r_scale = 0.0
            r_wrist = np.array([0.0, 0.0], np.float32)

        if l_present:
            left_local, l_scale = _normalize_local(left_xy)
            l_wrist = left_xy[IDX_WRIST]
        else:
            left_local = np.zeros((21,2), np.float32); l_scale = 0.0
            l_wrist = np.array([0.0, 0.0], np.float32)

        inter = l_wrist - r_wrist

        feat = np.concatenate([
            right_local.flatten(),      # 42*2
            left_local.flatten(),       # 42*2
            r_wrist, l_wrist,           # 4
            inter,                      # 2
            np.array([r_scale, l_scale, r_present, l_present], np.float32)  # 4
        ], axis=0)
        return feat  # (94,)

    def video_to_sequence(self, video_path, target_len=48, resize_width=640):
        cap = cv2.VideoCapture(video_path)
        seq = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            if resize_width is not None:
                h, w = frame.shape[:2]
                new_h = int(h * (resize_width / w))
                frame = cv2.resize(frame, (resize_width, new_h))
            feat = self.frame_to_feature(frame)
            seq.append(feat)
        cap.release()

        if len(seq) == 0:
            seq = [np.zeros((FEATS_PER_FRAME,), np.float32)]

        seq = np.array(seq, dtype=np.float32)  # (T_raw, 94)
        T = len(seq)
        if T >= target_len:
            seq = seq[:target_len]
            mask = np.ones(target_len, dtype=np.float32)
        else:
            pad = np.zeros((target_len - T, seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
            mask = np.concatenate([np.ones(T, dtype=np.float32),
                                   np.zeros(target_len - T, dtype=np.float32)])
        return seq, mask  # (T,94), (T,)
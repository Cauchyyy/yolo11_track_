#!/usr/bin/env python3
# run_botsort_softreid.py
import os, sys, argparse, time
proj = os.path.abspath(os.path.dirname(__file__))
if proj not in sys.path:
    sys.path.insert(0, proj)

import numpy as np, cv2
from ultralytics import YOLO

from ultralytics.trackers.botsort_softreid import BoTSORTSoftReID
try:
    from reid.extract_embeddings import ReIDEncoder
except Exception:
    ReIDEncoder = None

def run_yolo_on_frame(model, frame, conf_thresh=0.25):
    try:
        res = model.predict(frame, conf=conf_thresh, verbose=False)
        r = res[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            out = np.concatenate([xyxy, confs.reshape(-1,1), clss.reshape(-1,1)], axis=1)
            return out
    except Exception as e:
        print("YOLO error:", e)
    return np.zeros((0,6))

def normalize_dets(dets, W,H):
    if dets is None or dets.size==0:
        return np.zeros((0,6), dtype=float)
    arr = np.asarray(dets, dtype=float)
    if arr.max() <= 1.01:
        arr[:,0] = np.clip(arr[:,0]*W, 0, W-1)
        arr[:,2] = np.clip(arr[:,2]*W, 0, W-1)
        arr[:,1] = np.clip(arr[:,1]*H, 0, H-1)
        arr[:,3] = np.clip(arr[:,3]*H, 0, H-1)
    arr[:,0] = np.clip(arr[:,0], 0, W-1); arr[:,1] = np.clip(arr[:,1], 0, H-1)
    arr[:,2] = np.clip(arr[:,2], 0, W-1); arr[:,3] = np.clip(arr[:,3], 0, H-1)
    return arr[:, :6]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/home/waas/yolo11_track/fish_video/rgb_165354_part9.mp4")
    parser.add_argument("--out",  default="/home/waas/yolo11_track/track_result/_1016_1_part9.mp4")
    parser.add_argument("--out_npy", default="/home/waas/yolo11_track/track_out_npy/_1016_1.npy")
    parser.add_argument("--weights", default="/home/waas/yolo11_track/train_0818_/exp/weights/best.pt", help="yolo weights")
    parser.add_argument("--reid_model", default="/home/waas/yolo11_track/reid_weight/best_model_resnet18.pth", help="reid pth (optional)")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--w_app", type=float, default=0.35)
    parser.add_argument("--w_motion", type=float, default=0.55)
    parser.add_argument("--w_iou", type=float, default=0.10)
    parser.add_argument("--reserve_frames", type=int, default=150)
    parser.add_argument("--stable_frames", type=int, default=5)
    parser.add_argument("--tracklet_len", type=int, default=10)
    parser.add_argument("--C_cls", type=float, default=0.60)
    parser.add_argument("--C_emb", type=float, default=0.65)
    parser.add_argument("--diagnostics_csv", default="/home/waas/yolo11_track/track_csv/track.csv")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open", args.video); return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
    print("Video opened:", args.video, "fps", fps, "size", (W,H))

    model = None
    if args.weights is not None:
        try:
            model = YOLO(args.weights)
            print("Loaded YOLO:", args.weights)
        except Exception as e:
            print("Failed loading YOLO:", e)
            model = None

    encoder = None
    if args.reid_model is not None:
        if ReIDEncoder is None:
            print("ReIDEncoder not found; REID disabled")
        else:
            try:
                encoder = ReIDEncoder(args.reid_model, device='cuda')
                print("Loaded ReID model:", args.reid_model)
            except Exception as e:
                print("Failed to load ReID:", e)
                encoder = None

    tracker = BoTSORTSoftReID(encoder=encoder,
                              w_app=args.w_app, w_motion=args.w_motion, w_iou=args.w_iou,
                              reserve_frames=args.reserve_frames, max_age=args.reserve_frames*2,
                              allow_new_tracks=True,
                              stable_frames=args.stable_frames, tracklet_len=args.tracklet_len,
                              C_cls=args.C_cls, C_emb=args.C_emb,
                              diagnostics_csv=args.diagnostics_csv)

    frame_idx = 0
    outputs_per_frame = []
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = np.zeros((0,6))
        if model is not None:
            dets = run_yolo_on_frame(model, frame, conf_thresh=args.conf_thresh)
        dets_px = normalize_dets(dets, W, H)
        outs = tracker.update(dets_px, frame)
        vis = frame.copy()
        # draw raw detections lightly
        for d in dets_px:
            x1,y1,x2,y2 = int(d[0]),int(d[1]),int(d[2]),int(d[3])
            cv2.rectangle(vis, (x1,y1), (x2,y2), (200,200,200), 1)
        if outs is not None and len(outs) > 0:
            for r in outs:
                x1,y1,x2,y2, display_id, score, fish_label = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), float(r[5]), int(r[6])
                color = ((display_id*37)%255, (display_id*91)%255, (display_id*53)%255)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                label_txt = f"ID:{display_id}"
                if fish_label >= 0:
                    label_txt += f" f{fish_label+1}"
                cv2.putText(vis, label_txt, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(vis)
        outputs_per_frame.append(outs if outs is not None else np.zeros((0,7)))
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames, elapsed {time.time()-t0:.1f}s")
    cap.release(); writer.release()
    np.save(args.out_npy, np.array(outputs_per_frame, dtype=object))
    print("Done. Saved:", args.out, args.out_npy)
    return 0

if __name__ == "__main__":
    main()

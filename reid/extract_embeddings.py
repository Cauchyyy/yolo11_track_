# reid/extract_embeddings.py
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from reid.models import ReIDModel

# transform must match training transform (except RandomErasing)
transform = transforms.Compose([
    transforms.Resize((256,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class ReIDEncoder:
    """
    Wrapper around trained ReIDModel.
    Usage:
      enc = ReIDEncoder('/path/to/reid_best.pth', device='cuda')
      feats = enc(frame, bboxes)  # returns (N, emb_dim) numpy
      labels, probs = enc.predict_labels(frame, bboxes)  # returns (N,), (N,)
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # instantiate model with same arch as training
        # num_classes is only needed for loading classifier - doesn't affect embedding head shape
        self.model = ReIDModel(num_classes=9, emb_dim=256, pretrained_backbone=None, use_imagenet_pretrained=False)
        state = torch.load(model_path, map_location='cpu')
        # accept ckpt with 'model_state' or direct state_dict
        if isinstance(state, dict) and 'model_state' in state:
            sd = state['model_state']
        else:
            sd = state
        # load state dict
        try:
            self.model.load_state_dict(sd, strict=False)
        except Exception as e:
            # attempt to handle if state is full checkpoint
            try:
                self.model.load_state_dict(sd, strict=False)
            except Exception as e2:
                print("Warning: loading reid model state failed:", e2)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _crop_and_transform(self, frame, bboxes):
        crops = []
        for box in bboxes:
            x1,y1,x2,y2 = int(max(0, box[0])), int(max(0, box[1])), int(max(0, box[2])), int(max(0, box[3]))
            if x2 <= x1 or y2 <= y1:
                # empty patch
                # create black image of target size
                img = Image.new('RGB', (128,256), (0,0,0))
            else:
                patch = frame[y1:y2, x1:x2, ::-1]  # BGR -> RGB
                img = Image.fromarray(patch)
            img_t = transform(img)
            crops.append(img_t)
        if len(crops) == 0:
            return None
        batch = torch.stack(crops, dim=0)
        return batch

    def __call__(self, frame_bgr, bboxes):
        """
        frame_bgr: HxWx3 BGR
        bboxes: list of [x1,y1,x2,y2]
        returns: np.ndarray (N, emb_dim)
        """
        if len(bboxes) == 0:
            return np.zeros((0, 256), dtype=np.float32)
        batch = self._crop_and_transform(frame_bgr, bboxes)
        batch = batch.to(self.device)
        with torch.no_grad():
            logits, feats = self.model(batch)
            feats = feats.cpu().numpy()
        return feats

    def predict_labels(self, frame_bgr, bboxes):
        """
        Predict per-box class label (0..8) and probability (softmax max)
        returns: labels (N,), probs (N,)
        """
        if len(bboxes) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        batch = self._crop_and_transform(frame_bgr, bboxes)
        batch = batch.to(self.device)
        with torch.no_grad():
            logits, feats = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            preds = preds.cpu().numpy().astype(int)
            confs = confs.cpu().numpy().astype(float)
        return preds, confs

    def predict_single(self, frame_bgr, bbox):
        preds, confs = self.predict_labels(frame_bgr, [bbox])
        if len(preds) == 0:
            return -1, 0.0
        return int(preds[0]), float(confs[0])

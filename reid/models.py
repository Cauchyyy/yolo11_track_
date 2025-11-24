# reid/models.py
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import re

try:
    # new torchvision API
    from torchvision.models import ResNet50_Weights
    HAVE_TORCHVISION_WEIGHTS = True
except Exception:
    ResNet50_Weights = None
    HAVE_TORCHVISION_WEIGHTS = False

class ReIDModel(nn.Module):
    def __init__(self, num_classes=9, emb_dim=256, pretrained_backbone=None, use_imagenet_pretrained=True):
        """
        num_classes: classification classes for CE loss (here 9)
        emb_dim: output embedding dim (e.g. 256)
        pretrained_backbone: optional path to a checkpoint that contains backbone weights
        use_imagenet_pretrained: if True and pretrained_backbone is None, use torchvision ImageNet weights
        """
        super().__init__()
        # build resnet50 backbone
        if use_imagenet_pretrained:
            try:
                if HAVE_TORCHVISION_WEIGHTS:
                    resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                else:
                    resnet = torchvision.models.resnet50(pretrained=True)
            except Exception:
                # fallback
                resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained=False)

        # remove avgpool and fc; keep conv layers up to layer4
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # optional: load user provided backbone checkpoint (only matching keys)
        if pretrained_backbone is not None:
            try:
                self._load_backbone_weights(pretrained_backbone)
            except Exception as e:
                print("Warning: failed to load pretrained_backbone:", e)

        # pooling and heads
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(emb_dim, num_classes)

        # init heads
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0.0)

    def _load_backbone_weights(self, pretrained_backbone_path):
        """
        Robustly load backbone params from a checkpoint (handles various key prefixes).
        Only loads keys that match shapes with self.backbone.state_dict()
        """
        sd = torch.load(pretrained_backbone_path, map_location='cpu')
        if isinstance(sd, dict) and ('model_state' in sd or 'model_state_dict' in sd):
            sd = sd.get('model_state', sd.get('model_state_dict', sd))
        if not isinstance(sd, dict):
            raise RuntimeError("pretrained_backbone file does not contain a state_dict-like mapping")
        # normalize keys
        new_sd = {}
        for k, v in sd.items():
            newk = k
            newk = re.sub(r'^module\.', '', newk)
            newk = re.sub(r'^(backbone\.|model\.|net\.|encoder\.)', '', newk)
            new_sd[newk] = v
        bk_sd = self.backbone.state_dict()
        filtered = {}
        for k, v in new_sd.items():
            if k in bk_sd and bk_sd[k].shape == v.shape:
                filtered[k] = v
        # load
        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        print(f"[load_backbone_weights] loaded {len(filtered)} backbone keys. missing keys: {len(missing)} unexpected: {len(unexpected)}")

    def forward(self, x):
        """
        Returns: logits, normalized_embedding
        """
        x = self.backbone(x)  # (B,2048,H',W')
        x = self.gap(x).view(x.size(0), -1)  # (B,2048)
        feat = self.fc(x)  # (B, emb_dim)
        feat = self.bn(feat)
        feat_norm = F.normalize(feat, p=2, dim=1)
        logits = self.classifier(feat)
        return logits, feat_norm

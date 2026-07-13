"""
Model Service — encapsulates everything related to the Computer Vision model:
SegFormer inference, bi-temporal difference map, and raw damage statistics.
Does NOT know anything about RAG, Gemini, or HTTP — pure model logic.
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

DECISION_SUPPORT = {
    "green": {"recommended_action": "Bantuan logistik dasar (non-darurat)",
              "evacuation_radius_km": 0.0,
              "required_logistics": ["Air bersih", "Bahan makanan"]},
    "yellow": {"recommended_action": "Asesmen lapangan lanjutan dalam 48-72 jam",
               "evacuation_radius_km": 0.5,
               "required_logistics": ["Tim asesmen struktur bangunan", "Tenda darurat"]},
    "orange": {"recommended_action": "Siapkan evakuasi sebagian & buka shelter terdekat",
               "evacuation_radius_km": 1.0,
               "required_logistics": ["Shelter sementara", "Tim medis", "Alat berat ringan"]},
    "red": {"recommended_action": "Immediate evacuation — evakuasi segera",
            "evacuation_radius_km": 2.0,
            "required_logistics": ["Tim SAR penuh", "Alat berat", "Shelter darurat", "Suplai medis darurat"]},
}


class ModelService:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True
        ).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _predict_tile_grid(self, img_rgb, tile_size=512):
        h, w = img_rgb.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        conf_values = []
        with torch.no_grad():
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    tile = img_rgb[y:y+tile_size, x:x+tile_size]
                    th, tw = tile.shape[:2]
                    if th < tile_size or tw < tile_size:
                        tile = cv2.copyMakeBorder(tile, 0, tile_size-th, 0, tile_size-tw,
                                                   cv2.BORDER_CONSTANT, value=0)
                    inp = self.transform(image=tile)['image'].unsqueeze(0).to(self.device)
                    out = self.model(pixel_values=inp)
                    logits = nn.functional.interpolate(out.logits, size=(tile_size, tile_size),
                                                        mode="bilinear", align_corners=False)
                    probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                    pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                    full_mask[y:y+th, x:x+tw] = pred[:th, :tw]
                    bp = probs[:th, :tw][pred[:th, :tw] == 1]
                    if bp.size:
                        conf_values.append(bp)
        mean_conf = float(np.concatenate(conf_values).mean()) if conf_values else 0.0
        return full_mask, mean_conf

    @staticmethod
    def _bucket(score):
        if score <= 20: return "green"
        if score <= 50: return "yellow"
        if score <= 80: return "orange"
        return "red"

    def analyze(self, pre_bgr, post_bgr):
        pre_rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)
        post_rgb = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB)

        mask_pre, _ = self._predict_tile_grid(pre_rgb)
        mask_post, conf_post = self._predict_tile_grid(post_rgb)

        diff = np.zeros_like(mask_pre, dtype=np.uint8)
        diff[(mask_pre == 1) & (mask_post == 1)] = 1
        diff[(mask_pre == 1) & (mask_post == 0)] = 2

        total_bldg = int(np.sum(mask_pre == 1))
        damaged = int(np.sum(diff == 2))
        ratio = damaged / max(total_bldg, 1)
        score = round(ratio * 100, 2)
        bucket = self._bucket(score)
        ds = DECISION_SUPPORT[bucket]

        stats = {
            "total_building_pixels": total_bldg,
            "damaged_pixels": damaged,
            "damage_ratio": round(ratio, 4),
            "damage_percentage": score,
            "triage_score": score,
            "priority": bucket.upper(),
            "confidence": round(conf_post, 3),
            "recommended_action": ds["recommended_action"],
            "evacuation_radius_km": ds["evacuation_radius_km"],
            "required_logistics": ds["required_logistics"],
        }
        return mask_pre, mask_post, diff, stats

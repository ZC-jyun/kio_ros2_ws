"""Grounding DINO open-vocabulary object detector."""
import numpy as np

GRASPABLE_CATEGORIES = [
    "medicine box", "towel", "door handle",
    "takeout bag", "cup", "bottle",
]


class ObjectDetector:
    def __init__(self, model_path, config_path, device="cuda"):
        from groundingdino.util.inference import load_model
        self.model = load_model(config_path, model_path)
        self.device = device
        self.text_prompt = " . ".join(GRASPABLE_CATEGORIES)

    def detect(self, rgb_image: np.ndarray, box_threshold=0.25,
               text_threshold=0.20, caption: str = "") -> list[dict]:
        """Run detection on an RGB image.

        Args:
            rgb_image: (H, W, 3) uint8 RGB.
            caption: text prompt to detect (defaults to GRASPABLE_CATEGORIES).
        """
        import torch
        from groundingdino.util.inference import predict

        prompt = caption or self.text_prompt

        # Convert numpy (H,W,3) uint8 RGB to torch (3,H,W) float32 [0,1]
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0

        boxes, logits, phrases = predict(
            model=self.model,
            image=img_tensor,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        results = []
        for box, conf, phrase in zip(boxes, logits, phrases):
            results.append({
                "class_name": phrase,
                "bbox": box.tolist(),
                "confidence": float(conf),
            })
        return results

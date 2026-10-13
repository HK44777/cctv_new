import numpy as np
import cv2
import torch
import torchreid
from src.utils.gpu_guard import get_device


class AppearanceExtractor:
    def __init__(self, weights_path="src/pipeline/extractors/osnet_x1_0_market1501.pth", strict_gpu=True):
        # strict_gpu=True  → require GPU (ingestion phase on Colab)
        # strict_gpu=False → allow CPU fallback (query server on local machine)
        self.device = get_device(strict=strict_gpu)

        # Step 5: Appearance Extraction (OSNet via torchreid)
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=False
        )
        self.model.to(self.device)

        try:
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
            print(f"Initialized Appearance Extractor (OSNet loaded from {weights_path})")
        except Exception as e:
            print(f"Warning: Could not load OSNet weights from {weights_path}. Using ImageNet weights instead. Error: {e}")
            self.model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=1000,
                pretrained=True
            )
            self.model.to(self.device)

        self.model.eval()

    def extract(self, person_crop):
        """
        Returns a 512-dimensional vector describing clothing/appearance.
        """
        if person_crop.size == 0:
            return None

        # OSNet Preprocessing (Standard Torchvision pipeline)
        # 1. Resize to 256x128 (Height x Width)
        img = cv2.resize(person_crop, (128, 256))

        # 2. BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Convert to float32 and scale to [0, 1]
        img = img.astype(np.float32) / 255.0

        # 4. Normalize using ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std

        # 5. Convert to tensor: HWC → CHW → Add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            features = self.model(img_tensor)  # Returns [1, 512] tensor

        # Postprocess: squeeze → CPU → numpy
        vector = features.squeeze().cpu().numpy()

        # L2 Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def extract_batch(self, person_crops):
        """
        Accepts a list of images, processes them all at once on the GPU,
        and returns a list of 512-dimensional vectors.
        """
        valid_crops = [crop for crop in person_crops if crop is not None and crop.size > 0]
        if not valid_crops:
            return []

        processed_imgs = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for crop in valid_crops:
            img = cv2.resize(crop, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std
            processed_imgs.append(img)

        # Create batch tensor
        batch_tensor = torch.from_numpy(np.array(processed_imgs)).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            features = self.model(batch_tensor)  # [N, 512]

        # L2 Normalize each vector in the batch
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy().tolist()

    def extract_with_parts(self, person_crop):
        """
        Extracts both global 512-d embedding and 3-part horizontal stripes (B, 512, 3)
        Part 0 = Head / Upper Torso
        Part 1 = Mid Torso / Waist
        Part 2 = Lower Legs / Feet
        """
        if person_crop is None or person_crop.size == 0:
            return None, None
            
        img = cv2.resize(person_crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fmap = self.model.featuremaps(img_tensor)  # (1, 512, 16, 8)
            # Global pooled
            global_feat = torch.nn.functional.adaptive_avg_pool2d(fmap, (1, 1)).view(1, 512)
            global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=1)
            # 3 parts pooled (horizontal stripes)
            parts = torch.nn.functional.adaptive_avg_pool2d(fmap, (3, 1)).squeeze(3)
            parts = torch.nn.functional.normalize(parts, p=2, dim=1)

        g_vec = global_feat.squeeze().cpu().numpy().tolist()
        p_vecs = [parts[0, :, i].cpu().numpy().tolist() for i in range(3)]
        return g_vec, p_vecs


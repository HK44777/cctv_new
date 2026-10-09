import cv2
import numpy as np
import torch
import torch.nn as nn
from src.utils.gpu_guard import get_device


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSConv, self).__init__()
        # Depthwise convolution
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=in_channels, bias=True)
        # Pointwise convolution
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                    padding=0, bias=True)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = DSConv(3, number_f, 3, 1, 1)
        self.e_conv2 = DSConv(number_f, number_f, 3, 1, 1)
        self.e_conv3 = DSConv(number_f, number_f, 3, 1, 1)
        self.e_conv4 = DSConv(number_f, number_f, 3, 1, 1)
        self.e_conv5 = DSConv(number_f * 2, number_f, 3, 1, 1)
        self.e_conv6 = DSConv(number_f * 2, number_f, 3, 1, 1)
        self.e_conv7 = DSConv(number_f * 2, 24, 3, 1, 1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image + r5 * (torch.pow(enhance_image, 2) - enhance_image)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        return enhance_image


class ZeroDCEEnhancer:
    def __init__(self, weights_path: str = "src/pipeline/Epoch99.pth", strict_gpu: bool = True):
        # strict_gpu=True  → require GPU (ingestion phase on Colab)
        # strict_gpu=False → allow CPU fallback (query server on local machine)
        self.device = get_device(strict=strict_gpu)

        self.model = enhance_net_nopool().to(self.device)
        self.is_active = True

        if weights_path:
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"[Enhancer] ✅ Loaded Zero-DCE weights from {weights_path}")
            except Exception as e:
                print(f"[Enhancer] ❌ Error loading weights: {e}. Enhancer will be bypassed.")
                self.is_active = False
        else:
            print("[Enhancer] ⚠️ No weights path provided. Enhancer will be bypassed.")
            self.is_active = False

        self.model.eval()

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Step 1: Low-Light Enhancement using PyTorch Zero-DCE.
        """
        if not self.is_active:
            return frame

        # Only enhance dark frames (mean pixel value < threshold)
        if np.mean(frame) < 80:
            # Preprocess: BGR → RGB → Normalize [0,1] → Tensor → CHW → Batch
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = np.asarray(img) / 255.0
            img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                enhanced_tensor = self.model(img)

            # Postprocess: Tensor → HWC → [0,255] → uint8 → BGR
            enhanced_img = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            enhanced_img = np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8)
            return cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

        return frame

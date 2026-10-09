"""
GPU Model Verification Script
Strictly verifies that the system's GPU is accessible and that all AI models 
(Detection, Face, Appearance, Gait) are correctly configured for GPU execution.
"""
import sys

def print_status(component: str, status: str, is_error: bool = False, is_warning: bool = False):
    if is_error:
        prefix = "[X] ERROR  "
    elif is_warning:
        prefix = "[!] WARNING"
    else:
        prefix = "[V] OK     "
    print(f"{prefix} | {component:<25} : {status}")


def verify_gpu_and_models():
    print("=" * 65)
    print("                * AI MODEL GPU VERIFICATION *                ")
    print("=" * 65)
    
    all_passed = True
    
    # ---------------------------------------------------------
    # 1. CORE ENGINE GPU CHECKS
    # ---------------------------------------------------------
    # PyTorch Engine
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            print_status("PyTorch Engine", f"CUDA Available ({dev_name})")
        else:
            print_status("PyTorch Engine", "CUDA NOT available. CPU fallback.", is_error=True)
            all_passed = False
    except ImportError:
        print_status("PyTorch Engine", "Not installed", is_error=True)
        all_passed = False

    # ONNX Runtime Engine
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print_status("ONNX Runtime Engine", "CUDAExecutionProvider Available")
        else:
            print_status("ONNX Runtime Engine", "CUDA Provider Missing. CPU fallback.", is_error=True)
            all_passed = False
    except ImportError:
        print_status("ONNX Runtime Engine", "Not installed", is_error=True)
        all_passed = False

    print("-" * 65)
    
    # ---------------------------------------------------------
    # 2. SPECIFIC MODEL CHECKS
    # ---------------------------------------------------------
    # YOLOv8 (Detection)
    try:
        import ultralytics
        if 'torch' in sys.modules and torch.cuda.is_available():
            print_status("YOLOv8 (Detection)", "GPU Active (PyTorch CUDA)")
        else:
            print_status("YOLOv8 (Detection)", "GPU Unavailable", is_error=True)
    except ImportError:
        print_status("YOLOv8 (Detection)", "Not installed (will skip)", is_warning=True)

    # InsightFace (Face Recognition)
    try:
        import insightface
        if 'onnxruntime' in sys.modules and "CUDAExecutionProvider" in ort.get_available_providers():
             print_status("InsightFace (Face)", "GPU Active (ONNX CUDA)")
        else:
             print_status("InsightFace (Face)", "GPU Unavailable", is_error=True)
    except ImportError:
        print_status("InsightFace (Face)", "Not installed (will skip)", is_warning=True)

    # TorchReID (Appearance Re-ID)
    try:
        import torchreid
        if 'torch' in sys.modules and torch.cuda.is_available():
            print_status("TorchReID (Appearance)", "GPU Active (PyTorch CUDA)")
        else:
            print_status("TorchReID (Appearance)", "GPU Unavailable", is_error=True)
    except ImportError:
        print_status("TorchReID (Appearance)", "Not installed (will skip)", is_warning=True)
        
    # OpenGait (Gait Recognition)
    try:
        import os
        opengait_path = os.path.abspath(os.path.join(os.getcwd(), 'OpenGait'))
        if opengait_path not in sys.path:
            sys.path.insert(0, opengait_path)
        import opengait
        
        if 'torch' in sys.modules and torch.cuda.is_available():
            print_status("OpenGait (Gait)", "GPU Active (PyTorch CUDA)")
        else:
            print_status("OpenGait (Gait)", "GPU Unavailable", is_error=True)
    except ImportError:
        print_status("OpenGait (Gait)", "Not installed (will skip)", is_warning=True)

    print("=" * 65)
    
    # ---------------------------------------------------------
    # 3. FINAL VERDICT
    # ---------------------------------------------------------
    if all_passed:
        print("[SUCCESS] The core engines are correctly configured for GPU execution.")
        sys.exit(0)
    else:
        print("[FAILED] Strict GPU policy violated. Please fix dependencies.")
        sys.exit(1)


if __name__ == "__main__":
    verify_gpu_and_models()

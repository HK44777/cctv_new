import torch


def get_device(strict: bool = True) -> torch.device:
    """
    Get the compute device for model inference.

    strict=True  (default) — Raises RuntimeError immediately if no CUDA GPU is available.
                             Use this for the ingestion phase (Google Colab).
    strict=False           — Falls back to CPU silently if no GPU is found.
                             Use this for the query/retrieval server (local machine).

    Returns:
        torch.device — either 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU GUARD] ✅ GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        return device

    if strict:
        raise RuntimeError(
            "\n\n[GPU GUARD] ❌ No CUDA GPU detected!\n"
            "This pipeline requires a GPU to run.\n"
            "If you are on Google Colab, go to: Runtime → Change runtime type → GPU\n"
        )

    print("[GPU GUARD] ⚠️  No GPU detected — falling back to CPU (query/retrieval mode).")
    return torch.device("cpu")


def require_gpu() -> torch.device:
    """Convenience alias for get_device(strict=True). Always raises if no GPU."""
    return get_device(strict=True)

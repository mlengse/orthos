# orthos.py - Auto-detection wrapper for ML/CPU implementations
# Secara otomatis mendeteksi ketersediaan backend dan memilih implementasi yang sesuai
# Priority: ML (PyTorch) -> CPU

def _check_ml_available():
    """Check if PyTorch/ML backend is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _check_gpu_pytorch():
    """Check if PyTorch GPU (CUDA) is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Detect available backends
ML_AVAILABLE = _check_ml_available()
GPU_PYTORCH = _check_gpu_pytorch()

# Select implementation based on priority: ML -> CPU
# Note: orthos_gpu (numba/cupy) is deprecated due to previous issues
if ML_AVAILABLE:
    device_info = "GPU (PyTorch CUDA)" if GPU_PYTORCH else "CPU (PyTorch)"
    print(f"ML backend detected. Using OrthosML on {device_info}.")
    from orthos_ml import OrthosML as Orthos
    BACKEND = 'ml'
else:
    print("No ML backend. Using OrthosCPU implementation.")
    from orthos_cpu import OrthosCPU as Orthos
    BACKEND = 'cpu'


def create_orthos(checkpoint_dir: str = './checkpoints'):
    """
    Factory function to create appropriate Orthos instance.
    
    Args:
        checkpoint_dir: Directory for checkpoints (used by ML backend)
        
    Returns:
        Orthos instance (OrthosML or OrthosCPU)
    """
    if BACKEND == 'ml':
        return Orthos(checkpoint_dir=checkpoint_dir)
    else:
        return Orthos()


def is_ml_available():
    """Check if ML (PyTorch) implementation is available."""
    return ML_AVAILABLE


def is_gpu_available():
    """Check if GPU is available for current backend."""
    if BACKEND == 'ml':
        return GPU_PYTORCH
    return False


def get_backend():
    """Get current backend name."""
    return BACKEND


def get_device():
    """Get current device (cuda/cpu)."""
    if BACKEND == 'ml' and GPU_PYTORCH:
        return 'cuda'
    return 'cpu'


if __name__ == "__main__":
    print(f"ML Available: {ML_AVAILABLE}")
    print(f"GPU (PyTorch CUDA): {GPU_PYTORCH}")
    print(f"Backend: {BACKEND}")
    print(f"Device: {get_device()}")
    print(f"Using class: {Orthos.__name__}")

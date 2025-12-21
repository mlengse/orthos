# orthos.py - Auto-detection wrapper for GPU/CPU implementations
# Secara otomatis mendeteksi ketersediaan GPU dan memilih implementasi yang sesuai

def _check_gpu_available():
    """Check if GPU is available and accessible."""
    try:
        import cupy as cp
        from numba import cuda
        
        # Try to get current device
        cuda.get_current_device()
        
        # Try simple cupy operation
        _ = cp.array([1, 2, 3])
        
        return True
    except ImportError:
        return False
    except Exception:
        return False


# Detect GPU availability
GPU_AVAILABLE = _check_gpu_available()

if GPU_AVAILABLE:
    print("GPU detected. Using OrthosGPU implementation.")
    from orthos_gpu import OrthosGPU as Orthos
else:
    print("No GPU detected. Using OrthosCPU implementation.")
    from orthos_cpu import OrthosCPU as Orthos


def create_orthos():
    """Factory function to create appropriate Orthos instance."""
    return Orthos()


def is_gpu_available():
    """Check if GPU implementation is being used."""
    return GPU_AVAILABLE


if __name__ == "__main__":
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Using class: {Orthos.__name__}")

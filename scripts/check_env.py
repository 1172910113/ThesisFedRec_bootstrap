"""Report the Python, PyTorch, CUDA, and GPU environment."""

import platform

import torch


def main() -> None:
    """Print environment information needed for experiment diagnostics."""
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'not available'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    for index in range(gpu_count):
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        print(f"GPU {index}: {name} (compute capability {major}.{minor})")


if __name__ == "__main__":
    main()

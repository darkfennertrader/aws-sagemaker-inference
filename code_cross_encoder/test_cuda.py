import torch
import gc, torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import GPUtil as GPU


def test_cuda():
    print("\n")
    print("-" * 50)
    print(f"torch version: {torch.__version__}")
    print(f"Is AI models using GPU?:  {torch.cuda.is_available()}")
    print("-" * 50)
    print("\n")

    gc.collect()
    torch.cuda.empty_cache()

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    free_vram = info.free / 1048576000
    print("There is a GPU with " + str(free_vram) + "GB of free VRAM")

    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    print(
        "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(
            gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal
        )
    )


if __name__ == "__main__":
    test_cuda()

import torch

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"GPU Memory - Allocated: {allocated / 1024**3:.2f} GB, Reserved: {reserved / 1024**3:.2f} GB")
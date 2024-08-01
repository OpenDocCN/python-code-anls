# `.\DB-GPT-src\dbgpt\util\model_utils.py`

```py
import logging  # 导入日志模块

from dataclasses import dataclass  # 导入用于定义数据类的模块
from typing import List, Tuple  # 导入用于类型提示的模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


def _clear_model_cache(device="cuda"):
    try:
        # 尝试清理 torch 缓存
        import torch

        _clear_torch_cache(device)
    except ImportError:
        logger.warn("Torch not installed, skip clear torch cache")  # 如果导入失败，则记录警告日志
    # TODO clear other cache  # TODO 注释：清理其他缓存的功能待实现


def _clear_torch_cache(device="cuda"):
    import gc  # 导入垃圾回收模块

    import torch  # 导入 PyTorch 模块

    gc.collect()  # 手动触发垃圾回收
    if device != "cpu":
        if torch.has_mps:
            try:
                from torch.mps import empty_cache  # 导入清理 MPS 缓存的函数

                empty_cache()  # 清理 MPS 缓存
            except Exception as e:
                logger.warn(f"Clear mps torch cache error, {str(e)}")  # 记录清理 MPS 缓存时的错误日志
        elif torch.has_cuda:
            device_count = torch.cuda.device_count()  # 获取 CUDA 设备的数量
            for device_id in range(device_count):
                cuda_device = f"cuda:{device_id}"
                logger.info(f"Clear torch cache of device: {cuda_device}")  # 记录正在清理的 CUDA 设备的日志
                with torch.cuda.device(cuda_device):
                    torch.cuda.empty_cache()  # 清理当前 CUDA 设备的缓存
                    torch.cuda.ipc_collect()  # 收集 CUDA IPC 内存
        else:
            logger.info("No cuda or mps, not support clear torch cache yet")  # 如果没有找到支持的 CUDA 或 MPS，记录日志


@dataclass
class GPUInfo:
    total_memory_gb: float
    allocated_memory_gb: float
    cached_memory_gb: float
    available_memory_gb: float


def _get_current_cuda_memory() -> List[GPUInfo]:
    try:
        import torch  # 尝试导入 torch 模块
    except ImportError:
        logger.warn("Torch not installed")  # 如果导入失败，则记录警告日志
        return []
    if torch.cuda.is_available():  # 如果 CUDA 可用
        num_gpus = torch.cuda.device_count()  # 获取 CUDA 设备数量
        gpu_infos = []
        for gpu_id in range(num_gpus):
            with torch.cuda.device(gpu_id):
                device = torch.cuda.current_device()  # 获取当前 CUDA 设备的 ID
                gpu_properties = torch.cuda.get_device_properties(device)  # 获取当前 CUDA 设备的属性
                total_memory = round(gpu_properties.total_memory / (1.0 * 1024**3), 2)  # 计算总内存（GB）
                allocated_memory = round(
                    torch.cuda.memory_allocated() / (1.0 * 1024**3), 2
                )  # 计算已分配内存（GB）
                cached_memory = round(
                    torch.cuda.memory_reserved() / (1.0 * 1024**3), 2
                )  # 计算缓存内存（GB）
                available_memory = total_memory - allocated_memory  # 计算可用内存（GB）
                gpu_infos.append(
                    GPUInfo(
                        total_memory_gb=total_memory,
                        allocated_memory_gb=allocated_memory,
                        cached_memory_gb=cached_memory,
                        available_memory_gb=available_memory,
                    )
                )  # 将 GPU 信息添加到列表中
        return gpu_infos  # 返回 GPU 信息列表
    else:
        logger.warn("CUDA is not available.")  # 如果 CUDA 不可用，则记录警告日志
        return []  # 返回空列表
```
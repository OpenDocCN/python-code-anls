# `.\integrations\tpu.py`

```py
# 导入 DataLoader 类从 torch.utils.data 模块
# 导入 is_torch_xla_available 函数从上级模块 ..utils 中
from torch.utils.data import DataLoader
from ..utils import is_torch_xla_available

# 定义函数 tpu_spmd_dataloader，接收一个 DataLoader 对象作为参数
def tpu_spmd_dataloader(dataloader: DataLoader):
    # 如果 Torch XLA 可用
    if is_torch_xla_available():
        # 导入 torch_xla.distributed.parallel_loader 模块，并重命名为 pl
        import torch_xla.distributed.parallel_loader as pl
        
        # 断言确保 dataloader 是一个 torch_xla.distributed.parallel_loader.MpDeviceLoader 对象
        assert isinstance(
            dataloader, pl.MpDeviceLoader
        ), "The dataloader must be a `torch_xla.distributed.parallel_loader.MpDeviceLoader`."
        
        # 注释：这段代码支持通过 SPMD 实现 PyTorch/XLA FSDP。
        # 在这里，我们将输入数据的第 0 维在 fsdp 轴上进行分片。
        
        # 导入 torch_xla.distributed.spmd 模块，并重命名为 xs
        import torch_xla.distributed.spmd as xs
        
        # 获取全局网格并创建一个 ShardingSpec 对象，指定 fsdp 轴上的分片规范
        sharding_spec = xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None))
        
        # 将 input_sharding 参数设置为上述分片规范
        dataloader._parallel_loader_kwargs["input_sharding"] = sharding_spec
        
        # 返回修改后的 dataloader 对象
        return dataloader
    else:
        # 如果 Torch XLA 不可用，则直接返回原始的 dataloader 对象
        return dataloader
```
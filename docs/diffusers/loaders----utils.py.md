# `.\diffusers\loaders\utils.py`

```py
# 版权声明，表明此文件的所有权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第2.0版（“许可证”）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按现状”提供的，
# 不附带任何明示或暗示的保证或条件。
# 请参阅许可证以了解与权限和
# 限制相关的特定语言。

# 导入类型字典
from typing import Dict

# 导入 PyTorch 库
import torch

# 定义一个名为 AttnProcsLayers 的类，继承自 torch.nn.Module
class AttnProcsLayers(torch.nn.Module):
    # 初始化方法，接受一个字典作为参数
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        # 调用父类的初始化方法
        super().__init__()
        # 将字典的值存储到一个 ModuleList 中
        self.layers = torch.nn.ModuleList(state_dict.values())
        # 创建一个映射字典，将索引与键关联
        self.mapping = dict(enumerate(state_dict.keys()))
        # 创建反向映射字典，将键与索引关联
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # 定义用于分割的关键字列表，分别用于处理器和自注意力
        self.split_keys = [".processor", ".self_attn"]

        # 定义一个将状态字典映射到模块的函数
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}  # 创建一个新的状态字典
            # 遍历状态字典中的每个键值对
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 提取数字部分，0 总是“layers”
                # 根据映射生成新的键
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value  # 存储到新字典中

            return new_state_dict  # 返回新状态字典

        # 定义一个用于重新映射键的函数
        def remap_key(key, state_dict):
            # 遍历分割关键字
            for k in self.split_keys:
                if k in key:  # 如果键包含分割关键字
                    return key.split(k)[0] + k  # 返回处理后的键

            # 如果没有找到匹配的分割关键字，抛出错误
            raise ValueError(
                f"There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}."
            )

        # 定义一个将状态字典映射回模块的函数
        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())  # 获取所有键
            # 遍历所有键
            for key in all_keys:
                replace_key = remap_key(key, state_dict)  # 重新映射键
                # 生成新的键
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]  # 更新状态字典
                del state_dict[key]  # 删除旧的键

        # 注册状态字典钩子以实现映射
        self._register_state_dict_hook(map_to)
        # 注册加载状态字典前的钩子以实现映射
        self._register_load_state_dict_pre_hook(map_from, with_module=True)
```
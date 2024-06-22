# `.\transformers\utils\model_parallel_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 从 math 模块导入 ceil 函数
from math import ceil

# 检查设备映射是否正确
def assert_device_map(device_map, num_blocks):
    # 创建一个包含 0 到 num_blocks-1 的列表
    blocks = list(range(0, num_blocks))

    # 将设备映射中的所有值合并为一个列表
    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # 检查重复的块
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # 找出缺失的块
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    # 如果有重复的块，则引发 ValueError
    if len(duplicate_blocks) != 0:
        raise ValueError(
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device."
            " These attention blocks were specified more than once: " + str(duplicate_blocks)
        )
    # 如果有缺失的块，则引发 ValueError
    if len(missing_blocks) != 0:
        raise ValueError(
            "There are attention blocks for this model that are not specified in the device_map. Add these attention "
            "blocks to a device on the device_map: " + str(missing_blocks)
        )
    # 如果设备映射包含多余的块，则引发 ValueError
    if len(extra_blocks) != 0:
        raise ValueError(
            "The device_map contains more attention blocks than this model has. Remove these from the device_map:"
            + str(extra_blocks)
        )

# 返回一个字典，将层均匀分布在所有设备上
def get_device_map(n_layers, devices):
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = [layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks)]

    return dict(zip(devices, layers_list))
```
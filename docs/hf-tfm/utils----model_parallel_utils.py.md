# `.\utils\model_parallel_utils.py`

```
# coding=utf-8
# 上面是代码文件的编码声明和版权信息，标识使用了Apache许可证版本2.0

from math import ceil  # 导入math库中的ceil函数，用于向上取整操作


def assert_device_map(device_map, num_blocks):
    # 创建一个从0到num_blocks-1的整数列表，表示模型中的注意力块编号
    blocks = list(range(0, num_blocks))

    # 将device_map字典中所有值（即所有分配的注意力块）组成一个单层列表
    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # 检查是否有重复的注意力块分配
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)

    # 找出未分配的注意力块编号
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    # 找出额外被分配的注意力块编号（超出了模型中的块数）
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    # 如果有重复的注意力块分配，抛出错误
    if len(duplicate_blocks) != 0:
        raise ValueError(
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device."
            " These attention blocks were specified more than once: " + str(duplicate_blocks)
        )
    # 如果有未分配的注意力块，抛出错误
    if len(missing_blocks) != 0:
        raise ValueError(
            "There are attention blocks for this model that are not specified in the device_map. Add these attention "
            "blocks to a device on the device_map: " + str(missing_blocks)
        )
    # 如果有额外的注意力块被分配，抛出错误
    if len(extra_blocks) != 0:
        raise ValueError(
            "The device_map contains more attention blocks than this model has. Remove these from the device_map:"
            + str(extra_blocks)
        )


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    # 创建层编号列表，从0到n_layers-1
    layers = list(range(n_layers))
    # 计算每个设备上分配的层数目
    n_blocks = int(ceil(n_layers / len(devices)))
    # 将层编号分组，使每个设备上的层尽量平均
    layers_list = [layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks)]

    # 返回一个字典，键为设备标识，值为该设备上分配的层编号列表
    return dict(zip(devices, layers_list))
```
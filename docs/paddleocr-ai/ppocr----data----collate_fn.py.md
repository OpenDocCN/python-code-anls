# `.\PaddleOCR\ppocr\data\collate_fn.py`

```
# 导入 paddle 模块
import paddle
# 导入 numbers 模块
import numbers
# 导入 numpy 模块，并重命名为 np
import numpy as np
# 导入 defaultdict 类
from collections import defaultdict

# 定义 DictCollator 类
class DictCollator(object):
    """
    data batch
    """

    # 定义 __call__ 方法
    def __call__(self, batch):
        # todo：支持批处理操作
        # 创建一个默认值为列表的字典
        data_dict = defaultdict(list)
        # 存储需要转换为张量的键
        to_tensor_keys = []
        # 遍历批次中的样本
        for sample in batch:
            # 遍历样本中的键值对
            for k, v in sample.items():
                # 检查值是否为 ndarray、Tensor 或数字
                if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
                    # 如果键不在需要转换为张量的键列表中，则添加到列表中
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                # 将值添加到对应键的列表中
                data_dict[k].append(v)
        # 将需要转换为张量的键转换为张量
        for k in to_tensor_keys:
            data_dict[k] = paddle.to_tensor(data_dict[k])
        # 返回数据字典
        return data_dict

# 定义 ListCollator 类
class ListCollator(object):
    """
    data batch
    """

    # 定义 __call__ 方法
    def __call__(self, batch):
        # todo：支持批处理操作
        # 创建一个默认值为列表的字典
        data_dict = defaultdict(list)
        # 存储需要转换为张量的索引
        to_tensor_idxs = []
        # 遍历批次中的样本
        for sample in batch:
            # 遍历样本中的值
            for idx, v in enumerate(sample):
                # 检查值是否为 ndarray、Tensor 或数字
                if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
                    # 如果索引不在需要转换为张量的索引列表中，则添加到列表中
                    if idx not in to_tensor_idxs:
                        to_tensor_idxs.append(idx)
                # 将值添加到对应索引的列表中
                data_dict[idx].append(v)
        # 将需要转换为张量的索引转换为张量
        for idx in to_tensor_idxs:
            data_dict[idx] = paddle.to_tensor(data_dict[idx])
        # 返回数据字典的值列表
        return list(data_dict.values())

# 定义 SSLRotateCollate 类
class SSLRotateCollate(object):
    """
    # 定义一个包含多个元素的列表，每个元素是一个包含两个元素的列表，第一个元素是形状为(4*3xH*W)的数组，第二个元素是形状为(4,)的数组
    bach: [
        [(4*3xH*W), (4,)]
        [(4*3xH*W), (4,)]
        ...
    ]
    """

    # 定义一个方法，接受一个batch参数
    def __call__(self, batch):
        # 将batch中的每个元素按列拼接起来，形成一个新的数组
        output = [np.concatenate(d, axis=0) for d in zip(*batch)]
        # 返回拼接后的数组
        return output
class DyMaskCollator(object):
    """
    batch: [
        image [batch_size, channel, maxHinbatch, maxWinbatch]
        image_mask [batch_size, channel, maxHinbatch, maxWinbatch]
        label [batch_size, maxLabelLen]
        label_mask [batch_size, maxLabelLen]
        ...
    ]
    """

    # 定义一个类，用于处理数据批次
    def __call__(self, batch):
        # 初始化最大宽度、最大高度和最大长度
        max_width, max_height, max_length = 0, 0, 0
        # 获取批次大小和通道数
        bs, channel = len(batch), batch[0][0].shape[0]
        # 存储符合条件的数据项
        proper_items = []
        # 遍历批次中的每个数据项
        for item in batch:
            # 如果图片的宽度或高度超过阈值，则跳过该数据项
            if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[
                    2] * max_height > 1600 * 320:
                continue
            # 更新最大高度和宽度
            max_height = item[0].shape[1] if item[0].shape[
                1] > max_height else max_height
            max_width = item[0].shape[2] if item[0].shape[
                2] > max_width else max_width
            # 更新最大长度
            max_length = len(item[1]) if len(item[
                1]) > max_length else max_length
            # 将符合条件的数据项添加到列表中
            proper_items.append(item)

        # 初始化图像和图像掩码数组
        images, image_masks = np.zeros(
            (len(proper_items), channel, max_height, max_width),
            dtype='float32'), np.zeros(
                (len(proper_items), 1, max_height, max_width), dtype='float32')
        # 初始化标签和标签掩码数组
        labels, label_masks = np.zeros(
            (len(proper_items), max_length), dtype='int64'), np.zeros(
                (len(proper_items), max_length), dtype='int64')

        # 遍历符合条件的数据项
        for i in range(len(proper_items)):
            _, h, w = proper_items[i][0].shape
            # 将图像数据和掩码数据填充到数组中
            images[i][:, :h, :w] = proper_items[i][0]
            image_masks[i][:, :h, :w] = 1
            l = len(proper_items[i][1])
            # 将标签数据和掩码数据填充到数组中
            labels[i][:l] = proper_items[i][1]
            label_masks[i][:l] = 1

        # 返回处理后的图像、图像掩码、标签和标签掩码数组
        return images, image_masks, labels, label_masks
```
# `.\YOLO-World\yolo_world\datasets\transformers\mm_transforms.py`

```
# 导入所需的库
import json
import random
from typing import Tuple

import numpy as np
from mmyolo.registry import TRANSFORMS

# 注册 RandomLoadText 类为 TRANSFORMS 模块
@TRANSFORMS.register_module()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        # 初始化 RandomLoadText 类的属性
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        # 如果指定了 text_path，则读取对应文件内容
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

# 注册 LoadText 类为 TRANSFORMS 模块
@TRANSFORMS.register_module()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        # 初始化 LoadText 类的属性
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        # 如果指定了 text_path，则读取对应文件内容
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    # 定义 __call__ 方法，用于处理结果字典
    def __call__(self, results: dict) -> dict:
        # 检查结果字典中是否包含 'texts' 键或者类属性中是否包含 'class_texts'
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        # 获取类属性中的 'class_texts' 或者结果字典中的 'texts'
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        texts = []
        # 遍历类别文本列表，处理每个类别文本
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        # 将处理后的文本列表存入结果字典中的 'texts' 键
        results['texts'] = texts

        return results
```
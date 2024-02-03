# `.\PaddleOCR\ppocr\data\imaug\randaugment.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 导入 PIL 库中的 Image、ImageEnhance、ImageOps 模块
from PIL import Image, ImageEnhance, ImageOps
# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 random 模块
import random
# 导入 six 模块
import six

# 定义 RawRandAugment 类
class RawRandAugment(object):
    # 定义 __call__ 方法
    def __call__(self, img):
        # 获取可用操作名称列表
        avaiable_op_names = list(self.level_map.keys())
        # 循环执行指定次数的数据增强操作
        for layer_num in range(self.num_layers):
            # 随机选择一个操作名称
            op_name = np.random.choice(avaiable_op_names)
            # 根据操作名称和对应的级别执行数据增强操作
            img = self.func[op_name](img, self.level_map[op_name])
        return img

# 定义 RandAugment 类，继承自 RawRandAugment 类
class RandAugment(RawRandAugment):
    """ RandAugment 包装器，自动适配不同的图像类型 """

    # 定义初始化方法
    def __init__(self, prob=0.5, *args, **kwargs):
        self.prob = prob
        # 如果是 Python 2 版本
        if six.PY2:
            super(RandAugment, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    # 定义 __call__ 方法
    def __call__(self, data):
        # 根据概率决定是否执行数据增强操作
        if np.random.rand() > self.prob:
            return data
        # 获取图像数据
        img = data['image']
        # 如果图像不是 PIL 图像类型，则转换为 numpy 数组
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        # 如果是 Python 2 版本
        if six.PY2:
            img = super(RandAugment, self).__call__(img)
        else:
            img = super().__call__(img)

        # 如果图像是 PIL 图像类型，则转换为 numpy 数组
        if isinstance(img, Image.Image):
            img = np.asarray(img)
        data['image'] = img
        return data
```